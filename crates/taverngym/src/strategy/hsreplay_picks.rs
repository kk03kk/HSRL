use crate::cards::{CardId, CardKind, db};
use crate::content::trinket_offers;
use crate::state::{Pending, Player};
use crate::types::{Kw, Races};
use crate::{Action, Game};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

use super::{apm, guide_bot::Build, quilboar::is_choose_one};

#[derive(Default)]
pub struct Rules {
    pub hero: bool,
    pub trinket: bool,
}

#[derive(Default)]
pub struct Decisions {
    pub hero_dbf: u32,
    pub hero_best_dbf: u32,
    pub hero_rated: bool,
    pub hero_changed: bool,
    pub lesser: [u32; 3],
    pub greater: [u32; 3],
}

#[derive(Deserialize)]
struct Snapshot {
    heroes: Vec<(u32, f64)>,
    trinkets: Vec<(u32, String, f64, f64)>,
}

#[derive(Clone, Copy)]
struct TrinketStats {
    pick: f64,
    rank: f64,
}

struct Stats {
    heroes: HashMap<u32, f64>,
    trinkets: HashMap<(u32, u16), TrinketStats>,
}

fn stats() -> &'static Stats {
    static LOADED: OnceLock<Stats> = OnceLock::new();
    LOADED.get_or_init(|| {
        let snapshot: Snapshot = serde_json::from_str(include_str!("hsreplay_37_0_3.json"))
            .expect("HSReplay statistics snapshot");
        Stats {
            heroes: snapshot.heroes.into_iter().collect(),
            trinkets: snapshot
                .trinkets
                .into_iter()
                .map(|(dbf, race, pick, rank)| {
                    (
                        (dbf, Races::from_name(&race).0),
                        TrinketStats { pick, rank },
                    )
                })
                .collect(),
        }
    })
}

fn trinket_stats(game: &Game, seat: usize, card: CardId) -> Option<TrinketStats> {
    let def = db().get(card);
    let data = stats();
    if let Some(&entry) = data.trinkets.get(&(def.dbf, 0)) {
        return Some(entry);
    }
    if !matches!(
        def.id.as_str(),
        "BG30_MagicItem_426" | "BG30_MagicItem_426t"
    ) {
        return None;
    }
    let player = &game.players[seat];
    let stored = player
        .counters
        .iter()
        .find(|(id, _)| *id == card)
        .map_or(0, |(_, value)| *value);
    if stored > 0 {
        return data.trinkets.get(&(def.dbf, stored as u16)).copied();
    }
    // The simulator resolves Compass ties at random. Use the mean of eligible variants
    // while the offer has no assigned type, without consuming game RNG.
    let races: Vec<_> = player
        .board
        .iter()
        .chain(player.hand.iter().filter_map(|h| h.item.as_minion()))
        .map(|m| m.races)
        .collect();
    let lobby = game.lobby_races;
    let common: Vec<_> = lobby
        .iter()
        .filter(|r| races.iter().any(|m| *m != Races::ALL && m.intersects(*r)))
        .map(|r| (r, races.iter().filter(|m| m.intersects(r)).count()))
        .collect();
    let most = common.iter().map(|(_, count)| *count).max().unwrap_or(0);
    let mut pick = 0.0;
    let mut rank = 0.0;
    let mut count = 0;
    for race in lobby
        .iter()
        .filter(|r| most == 0 || common.contains(&(*r, most)))
    {
        if let Some(value) = data.trinkets.get(&(def.dbf, race.0)) {
            pick += value.pick;
            rank += value.rank;
            count += 1;
        }
    }
    (count > 0).then(|| TrinketStats {
        pick: pick / f64::from(count),
        rank: rank / f64::from(count),
    })
}

fn has(p: &Player, id: &str) -> bool {
    let card = db().id(id);
    p.board
        .iter()
        .chain(p.hand.iter().filter_map(|h| h.item.as_minion()))
        .any(|m| m.is_card_id(card))
}

/// Context adjustments distilled from the HSReplay trinket guide. Attached effects and
/// minion progress are read from the visible board and hand.
fn guide_fit(p: &Player, build: Build, card: CardId, price: i32, danger: bool) -> f64 {
    let def = db().get(card);
    let count = |race| p.board.iter().filter(|m| m.is(race)).count();
    let fit = |enabled| if enabled { 3.5 } else { -4.0 };
    let brann = has(p, "BG_LOE_077");
    let pirate_line = matches!(build, Build::Apm(_)) || has(p, apm::ROGERS) || has(p, apm::ESCAPEE);
    let bounty = has(p, apm::ROGERS)
        || has(p, "BG33_821")
        || p.hand
            .iter()
            .any(|h| crate::content::is_bounty(h.item.card()));
    let gem_engine =
        p.stats.gem_atk + p.stats.gem_hp >= 3 || has(p, "BG31_326") || has(p, "BG33_885");
    let gem_supply = gem_engine
        || has(p, "BG36_370")
        || has(p, "BG34_684")
        || p.hand.iter().any(|h| h.item.card() == db().id("BG20_GEM"));
    let rewind = has(p, "BG26_174") || has(p, "BG32_873");
    let tavern_growth = p.stats.tavern_atk + p.stats.tavern_hp + p.stats.tavern_elemental_hp >= 8
        || p.shop
            .iter()
            .filter_map(|s| s.item.as_minion())
            .any(|m| m.atk + m.hp >= 32);
    let tier = p.tier;
    let racial_fit = trinket_offers::types(card).iter().any(|r| count(r) >= 3);
    let direct = match def.id.as_str() {
        "BG30_MagicItem_301" | "BG36_MagicItem_216" => fit(p
            .board
            .iter()
            .any(|m| m.is_card_id(db().id("BG25_008")) && m.atk + m.hp >= 16)),
        "BG30_MagicItem_962" => fit(has(p, "BG21_015")),
        "BG32_MagicItem_363" => fit(count(Races::DRAGON) >= 3 && has(p, "BG29_813")),
        "BG30_MagicItem_437"
        | "BG32_MagicItem_360"
        | "BG30_MagicItem_547"
        | "BG30_MagicItem_547t"
        | "BG32_MagicItem_270t" => fit(p.stats.undead_atk >= 4
            || (count(Races::UNDEAD) >= 3 && has(p, "BG25_008"))
            || build == Build::Apm(apm::Route::Undead)),
        "BG32_MagicItem_808" | "BG32_MagicItem_808t" | "BG30_MagicItem_411" => {
            fit(gem_engine)
                + if gem_engine && has(p, "BG35_883") {
                    1.5
                } else {
                    0.0
                }
        }
        "BG30_MagicItem_988" | "BG36_MagicItem_307" => fit(gem_supply),
        "BG36_MagicItem_303" | "BG36_MagicItem_303t" => {
            if build == Build::Quilboar || p.board.iter().any(|m| is_choose_one(m.card)) {
                3.0
            } else if price <= 1 {
                1.25
            } else {
                0.0
            }
        }
        "BG35_MagicItem_741" => fit(has(p, "BG36_851")
            || has(p, "BG36_367")
            || count(Races::MECH) >= 3
            || build == Build::Apm(apm::Route::Mech)),
        "BG35_MagicItem_301"
        | "BG36_MagicItem_423"
        | "BG36_MagicItem_423t"
        | "BG35_MagicItem_743" => {
            fit(count(Races::MECH) >= 2 || build == Build::Apm(apm::Route::Mech))
        }
        "BG35_MagicItem_710" | "BG35_MagicItem_890" => fit(has(p, apm::PRIVATEER) && bounty),
        "BG36_MagicItem_363" | "BG32_MagicItem_232" | "BG32_MagicItem_957"
        | "BG35_MagicItem_711" => fit(pirate_line || bounty),
        "BG35_MagicItem_712" => {
            if bounty || pirate_line {
                3.5
            } else {
                -1.5
            }
        }
        "BG30_MagicItem_868" => {
            fit(!rewind && has(p, "BGS_004") || p.hero == Some(db().id("TB_BaconShop_HERO_37")))
        }
        "BG32_MagicItem_821" | "BG32_MagicItem_822" | "BG30_MagicItem_701" => fit(rewind),
        "BG35_MagicItem_156" | "BG35_MagicItem_862" | "BG35_MagicItem_154" => fit(tavern_growth),
        "BG36_MagicItem_830" => fit(has(p, "BG36_762") && has(p, "BG35_883")),
        "BG30_MagicItem_431" | "BG35_MagicItem_850" => {
            if build == Build::Azerite || count(Races::ELEMENTAL) >= 3 {
                3.5
            } else {
                -1.2
            }
        }
        "BG35_MagicItem_840t"
        | "BG35_MagicItem_755"
        | "BG32_MagicItem_888"
        | "BG35_MagicItem_750"
        | "BG35_MagicItem_753" => fit(brann),
        "BG30_MagicItem_418"
            if !brann && matches!(build, Build::Azerite | Build::Tea | Build::Apm(_)) =>
        {
            2.5
        }
        "BG36_MagicItem_414" => {
            if has(p, "BG26_ICC_901") {
                3.5
            } else if danger {
                -2.0
            } else {
                0.0
            }
        }
        "BG30_MagicItem_434" => {
            if p.hand
                .iter()
                .any(|h| h.item.card() == db().id("TB_BaconShop_Triples_01"))
            {
                3.0
            } else {
                0.0
            }
        }
        "BG30_MagicItem_423" => {
            if (4..=5).contains(&tier) && !danger {
                3.0
            } else {
                -2.0
            }
        }
        "BG35_MagicItem_931" => fit(tier == 4),
        "BG35_MagicItem_814" => fit(tier >= 5 && !danger),
        "BG32_MagicItem_858" | "BG35_MagicItem_815" | "BG30_MagicItem_430"
        | "BG35_MagicItem_840"
            if p.board.len() <= 5 =>
        {
            2.5
        }
        "BG30_MagicItem_902" | "BG36_MagicItem_206" if danger => 2.5,
        "BG30_MagicItem_847" | "BG32_MagicItem_428" | "BG36_MagicItem_220"
        | "BG32_MagicItem_271" => {
            if danger {
                0.5
            } else {
                1.8
            }
        }
        _ => 0.0,
    };
    direct
        + f64::from(i32::from(racial_fit))
        + if danger
            && p.board.iter().filter(|m| m.has(Kw::DIVINE_SHIELD)).count() == 0
            && def.id == "BG30_MagicItem_902"
        {
            0.75
        } else {
            0.0
        }
}

fn trinket_value(game: &Game, seat: usize, build: Build, card: CardId, free: bool) -> f64 {
    let p = &game.players[seat];
    let life = p.health + p.armor;
    let danger = life <= 16 || life <= 25 && p.stats.last_damage <= -8;
    let price = if free {
        0
    } else {
        game.trinket_price(seat, card)
    };
    let historical = trinket_stats(game, seat, card)
        .map_or(0.0, |s| 8.0 * (4.5 - s.rank) + 0.05 * (s.pick - 25.0));
    let budget = p.gold - price;
    let can_keep_curve = !free
        && !danger
        && (6..=10).contains(&game.turn)
        && p.tier < 6
        && budget >= game.upgrade_price(seat);
    let tempo_budget = danger && !free && budget >= game.minion_price(seat);
    historical + guide_fit(p, build, card, price, danger)
        - f64::from(price) * if danger { 1.0 } else { 0.62 }
        + if can_keep_curve { 1.7 } else { 0.0 }
        + if tempo_budget { 1.2 } else { 0.0 }
}

pub fn select(
    game: &Game,
    seat: usize,
    build: Build,
    base: Action,
    rules: &Rules,
    decisions: &mut Decisions,
) -> Action {
    let Some(menu) = game.players[seat].pending.last() else {
        return base;
    };
    let legal = |i| game.option_legal(seat, i);
    match menu {
        Pending::HeroSelect { options } => {
            let rates = &stats().heroes;
            let mut action = base;
            let best = options
                .iter()
                .enumerate()
                .filter(|(i, _)| legal(*i))
                .filter_map(|(i, &card)| rates.get(&db().get(card).dbf).map(|rate| (i, *rate)))
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)));
            if let Some((i, _)) = best {
                decisions.hero_best_dbf = db().get(options[i]).dbf;
                if rules.hero {
                    action = Action::Choose(i);
                }
            }
            if let Action::Choose(i) = action
                && let Some(&card) = options.get(i)
            {
                decisions.hero_dbf = db().get(card).dbf;
                decisions.hero_rated = rates.contains_key(&decisions.hero_dbf);
                decisions.hero_changed = action != base;
            }
            action
        }
        Pending::Trinket { options, free, .. } => {
            let mut action = base;
            if rules.trinket
                && let Some((i, _)) = options
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| legal(*i))
                    .map(|(i, &card)| (i, trinket_value(game, seat, build, card, *free)))
                    .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            {
                action = Action::Choose(i);
            }
            if let Action::Choose(i) = action
                && let Some(&card) = options.get(i)
            {
                let counts = if db().get(card).kind == CardKind::TrinketGreater {
                    &mut decisions.greater
                } else {
                    &mut decisions.lesser
                };
                counts[0] += 1;
                counts[1] += u32::from(trinket_stats(game, seat, card).is_some());
                counts[2] += u32::from(action != base);
            }
            action
        }
        _ => base,
    }
}
