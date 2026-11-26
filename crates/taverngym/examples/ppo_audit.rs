#[allow(dead_code)]
#[path = "strategy/apm_report.rs"]
mod apm_report;
#[allow(dead_code)]
#[path = "strategy/azerite_report.rs"]
mod azerite_report;
#[allow(dead_code)]
#[path = "strategy/catalog_report.rs"]
mod catalog_report;
#[allow(dead_code)]
#[path = "strategy/demon_report.rs"]
mod demon_report;
#[allow(dead_code)]
#[path = "strategy/quilboar_report.rs"]
mod quilboar_report;
#[allow(dead_code)]
#[path = "strategy/tea_report.rs"]
mod tea_report;

#[allow(dead_code)]
#[path = "strategy/undead_apm_report.rs"]
mod undead_apm_report;

use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, Write};
use taverngym::types::{Kw, Races};
use taverngym::{
    Action, Game,
    bc::Episode,
    cards::db,
    engine::Phase,
    strategy::{apm, azerite, catalog, demon, quilboar, tea, undead_apm},
};

#[derive(Default)]
struct Formations {
    azerite: azerite_report::Formation,
    tea: tea_report::Formation,
    quilboar: quilboar_report::Formation,
    demon: demon_report::Formation,
    apm: apm_report::Formation,
    catalog: Vec<(catalog::Route, catalog_report::Log)>,
}

fn route(core: Option<u32>, active: Option<u32>, full: Option<u32>, terminal: bool) -> Value {
    json!({"core":core,"active":active,"full":full,"terminal":terminal})
}

impl Formations {
    fn report(&self, races: Races) -> BTreeMap<&'static str, Value> {
        let mut out = BTreeMap::new();
        let a = &self.azerite;
        let t = &self.tea;
        let q = &self.quilboar;
        let d = &self.demon;
        if races.contains(Races::ELEMENTAL.union(Races::MURLOC)) {
            out.insert(
                "azerite",
                route(
                    a.core_turn,
                    a.working_turn,
                    a.full_turn,
                    a.last.as_ref().is_some_and(|b| b.pieces == azerite::FULL),
                ),
            );
        }
        out.insert(
            "tea",
            route(
                t.core_turn,
                t.working_turn,
                t.four_turn,
                t.last
                    .as_ref()
                    .is_some_and(|b| b.pieces & tea::GUIDE_FOUR == tea::GUIDE_FOUR),
            ),
        );
        if races.contains(Races::QUILBOAR) {
            out.insert(
                "quilboar",
                route(
                    q.core_turn,
                    q.working_turn,
                    q.five_turn,
                    q.last
                        .as_ref()
                        .is_some_and(|b| b.pieces == quilboar::GUIDE_FIVE),
                ),
            );
        }
        if races.contains(Races::DEMON) {
            out.insert(
                "demon",
                route(
                    d.engine_turn,
                    d.working_turn,
                    d.six_turn,
                    d.last
                        .as_ref()
                        .is_some_and(|b| b.pieces & demon::SIX == demon::SIX),
                ),
            );
        }
        for r in apm::Route::ALL {
            if races.contains(r.required()) {
                let c = &self.apm.routes[r.index()];
                out.insert(r.name(), route(c.core, c.working, c.full, c.final_full));
            }
        }
        for (r, log) in &self.catalog {
            let c = &log.routes[r.index()];
            out.insert(r.spec().key, route(c.core, c.started, c.full, c.final_full));
        }
        out
    }

    fn end_recruit(&mut self, game: &Game, seat: usize, turn: u32, tier: u8) {
        self.azerite.record_turn(game, seat, turn, tier);
        self.tea.record_turn(game, seat, turn, tier);
        self.quilboar.record_turn(game, seat, turn, tier);
        self.demon.record_turn(game, seat, turn, tier);
        self.apm.record_turn(game, seat, turn, tier);
    }
}

fn audit(record: &Value) -> Result<Value, String> {
    let expected = &record["game"];
    let seed = expected["seed"].as_u64().unwrap();
    let mut episode = Episode::new(seed, record["field"] == "mixed")?;
    let seat = episode.seat;
    let races = episode.game.lobby_races;
    let mut formation = Formations {
        catalog: catalog::Route::ALL
            .into_iter()
            .filter(|r| r.full_pool(races))
            .map(|r| (r, catalog_report::Log::default()))
            .collect(),
        ..Default::default()
    };
    let mut count = BTreeMap::<&str, u32>::new();
    let mut rounds = Vec::new();
    let mut undead = undead_apm_report::Proof::default();
    let mut streak = 0;
    let mut max_streak = 0;
    let mut previous_freeze = None;
    let mut unattributed = Vec::new();
    let mut watch_freeze: BTreeSet<u32> = BTreeSet::new();
    let mut bought_frozen = false;
    let mut last_recruit_board = json!([]);
    for id in record["actions"].as_array().unwrap() {
        let id = id.as_u64().unwrap() as usize;
        let game = &episode.game;
        let player = &game.players[seat];
        let action =
            Action::decode(id).ok_or_else(|| format!("seed {seed}: unknown action {id}"))?;
        let mask = game.legal_mask(seat);
        if !mask[id] || matches!(action, Action::Cancel | Action::Move { .. }) {
            return Err(format!("seed {seed}: prohibited action {action:?}"));
        }
        let name = match action {
            Action::EndTurn => "end",
            Action::Refresh => "refresh",
            Action::Freeze => "freeze",
            Action::Upgrade => "upgrade",
            Action::HeroPower(_) => "power",
            Action::DarkGift => "dark_gift",
            Action::Buy(_) => "buy",
            Action::Sell(_) => "sell",
            Action::Activate(_) => "activate",
            Action::Choose(_) => "choose",
            Action::Play { .. } => "play",
            Action::Magnetize { .. } => "magnetize",
            Action::Cancel | Action::Move { .. } => unreachable!(),
        };
        *count.entry(name).or_default() += 1;
        let turn = game.turn;
        let tier = player.tier;
        let gold = player.gold;
        let free = player.free_refreshes;
        let frozen = !player.shop.is_empty() && player.shop.iter().all(|s| s.frozen);
        let life = player.health + player.armor;
        let refresh_price = (action == Action::Refresh).then(|| game.refresh_price(seat));
        let health_refresh =
            refresh_price.is_some_and(|price| price > 0) && game.malchezaar(seat).is_some();
        let budget = player.actions_this_turn;
        let cap = game.action_cap_of(seat);
        let end = game.phase == Phase::Recruit && action == Action::EndTurn;
        if game.phase == Phase::Recruit && budget >= cap {
            if player.pending.is_empty() && !end {
                return Err(format!("seed {seed}: exceeded action cap in turn {turn}"));
            }
            if !end {
                *count.entry("post_cap_menu").or_default() += 1;
            }
        }
        if action == Action::Refresh {
            streak += 1;
            max_streak = max_streak.max(streak);
        } else {
            if streak >= 8 {
                unattributed.push(json!({"turn":turn,"kind":"refresh_streak","length":streak,"tier":tier,"gold_after":gold}));
            }
            streak = 0;
        }
        if let Action::Buy(i) = action
            && player
                .shop
                .get(i)
                .is_some_and(|slot| watch_freeze.contains(&slot.sid))
        {
            bought_frozen = true;
            *count.entry("buy_prior_frozen").or_default() += 1;
        }
        let inspect_shop = matches!(action, Action::Freeze | Action::Refresh);
        let mut shops = inspect_shop.then(|| player.shop.iter().map(|s| s.sid).collect::<Vec<_>>());
        if end {
            if !watch_freeze.is_empty() {
                *count.entry("freeze_followup_turns").or_default() += 1;
                *count.entry("freeze_followup_bought").or_default() += u32::from(bought_frozen);
            }
            watch_freeze = player
                .shop
                .iter()
                .filter(|s| s.frozen)
                .map(|s| s.sid)
                .collect();
            bought_frozen = false;
            rounds.push(
                json!({"turn":turn,"gold":gold,"tier":tier,"life":player.health + player.armor,
                "actions":budget,"forced":budget >= cap,"frozen":!watch_freeze.is_empty(),
                "hand":player.hand.len(),"board":player.board.len(),"bank":free}),
            );
        }
        let az = azerite_report::Witness::before_action(game, seat, action);
        let tea = tea_report::Cast::before_action(game, seat, action);
        let quill = quilboar_report::Choice::before_action(game, seat, action);
        let demon = demon_report::Witness::before_action(game, seat, action);
        let apm = apm_report::Witness::before_action(game, seat, action);
        let undead_step = undead_apm_report::Witness::before(game, seat, action);
        let witness: Vec<_> = formation
            .catalog
            .iter_mut()
            .map(|(r, log)| log.begin(*r, game, seat, action))
            .collect();
        episode.step(id, mask.iter().filter(|&&x| x).count() > 1)?;
        let game = &episode.game;
        let player = &game.players[seat];
        if let Some(w) = az {
            formation.azerite.record_action(w, game, seat);
        }
        if let Some(w) = tea {
            formation.tea.record_cast(w, player);
        }
        if let Some(w) = quill {
            formation.quilboar.record_choice(w, player);
        }
        if let Some(w) = demon {
            w.record(&mut formation.demon, player);
        }
        if let Some(w) = undead_step {
            w.apply(&mut undead, player, turn);
        }
        if let Some(w) = apm {
            w.record(&mut formation.apm, player, turn);
        }
        for ((_, log), w) in formation.catalog.iter_mut().zip(witness) {
            if let Some(w) = w {
                log.finish(w, player);
            }
        }
        if end {
            last_recruit_board = json!(player.last_board.iter().map(|m| {
                let card = db().get(m.card);
                json!({"name":card.name_zh,"id":card.id,"golden":m.golden,"attack":m.atk,
                    "health":m.hp,"races":m.races.0,"reborn":m.has(Kw::REBORN),"shield":m.has(Kw::DIVINE_SHIELD)})
            }).collect::<Vec<_>>());
            formation.end_recruit(game, seat, turn, tier);
            undead.record_turn(player, turn, tier);
        }
        if game.turn != turn || game.is_over() {
            formation.apm.after_combat(player, turn);
            for (_, log) in &mut formation.catalog {
                log.after_combat(game, seat, turn);
            }
        }
        if action == Action::Freeze {
            let same_shop = shops
                .as_ref()
                .unwrap()
                .iter()
                .copied()
                .eq(player.shop.iter().map(|s| s.sid));
            if frozen && same_shop && previous_freeze.as_ref() == shops.as_ref() {
                *count.entry("immediate_unfreeze").or_default() += 1;
                if unattributed.len() < 12 {
                    unattributed.push(json!({"turn":turn,"kind":"adjacent_unfreeze","actions":budget,"gold":gold}));
                }
            }
            *count
                .entry(if frozen { "freeze_off" } else { "freeze_on" })
                .or_default() += 1;
            previous_freeze = if frozen { None } else { shops.take() };
        } else {
            previous_freeze = None;
        }
        if action == Action::Refresh {
            if free > player.free_refreshes {
                *count.entry("refresh_free").or_default() += 1;
            } else if health_refresh {
                *count.entry("refresh_health").or_default() += 1;
            } else if refresh_price == Some(0) {
                *count.entry("refresh_other_free").or_default() += 1;
            } else {
                *count.entry("refresh_paid").or_default() += 1;
            }
            if shops
                .as_ref()
                .unwrap()
                .iter()
                .copied()
                .eq(player.shop.iter().map(|s| s.sid))
            {
                *count.entry("refresh_same_entities").or_default() += 1;
                if life == player.health + player.armor
                    && gold == player.gold
                    && free == player.free_refreshes
                {
                    *count.entry("refresh_same_no_resource_delta").or_default() += 1;
                }
                if unattributed.len() < 12 {
                    unattributed.push(json!({"turn":turn,"kind":"refresh_same_entities","actions":budget,"gold":gold,"shop":shops}));
                }
            }
        }
    }
    let player = &episode.game.players[seat];
    let hero = player.hero.map(|h| &db().get(h).name_zh);
    let global = json!({"undead_attack":player.stats.undead_atk,
        "tavern_attack":player.stats.tavern_atk,"tavern_health":player.stats.tavern_hp,
        "apm_peak_spend":formation.apm.peak_spend,"apm_bounties_cast":formation.apm.bounties_cast,
        "azerite_casts":formation.azerite.absorption_casts,"tea_targeted":formation.tea.targeted_casts,
        "quill_waves":formation.quilboar.gem_waves,"demon_safe_damage":formation.demon.safely_rewound});
    let undead = json!({"trio":undead.trio,"same_turn_loop":undead.same_turn_loop,
        "full_with_loop":undead.full_with_loop,"permanent_phantom_hp":undead.permanent_phantom_hp,
        "peak_phantom_gain":undead.peak_phantom_gain,"phantom_max_hp":undead.max_phantom_hp,
        "sacrifice_growth":undead.sacrifice_growth,"best":undead.tallest_phantom});
    let actual = serde_json::to_value(episode.result()?).unwrap();
    if &actual != expected {
        return Err(format!(
            "seed {seed}: Python and Rust complete results differ"
        ));
    }
    Ok(
        json!({"field":record["field"],"agent":record["agent"],"mode":record["mode"],
        "game":actual,"hero":hero,"turns":rounds,"action_counts":count,"max_refresh_streak":max_streak,
        "events":unattributed,"final_board":last_recruit_board,"global":global,"undead_proof":undead,
        "formations":formation.report(races)}),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let record = serde_json::from_str(&line?)?;
        serde_json::to_writer(&mut out, &audit(&record)?)?;
        writeln!(out)?;
    }
    Ok(())
}
