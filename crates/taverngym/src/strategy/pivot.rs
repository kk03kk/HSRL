use crate::cards::{CardId, db};
use crate::state::{Minion, Player};
use crate::types::{Kw, Races};
use crate::{Action, Game};

use super::{apm, cadence, catalog, guide_bot::Build};

pub const LOBBY: [&str; 5] = ["PIRATE", "UNDEAD", "MECHANICAL", "QUILBOAR", "ABERRATION"];
pub const LOBBY_MASK: Races = Races(
    Races::PIRATE.0 | Races::UNDEAD.0 | Races::MECH.0 | Races::QUILBOAR.0 | Races::ABERRATION.0,
);

pub fn bridge(build: Build) -> Option<catalog::Route> {
    match build {
        Build::Catalog(
            route @ (catalog::Route::Knight | catalog::Route::Sludge | catalog::Route::HandGems),
        ) => Some(route),
        _ => None,
    }
}

struct Package {
    build: Build,
    race: Races,
    pieces: &'static [&'static [&'static str]],
}

impl Package {
    fn fits(&self, game: &Game, p: &Player) -> bool {
        (self.race.is_none() || game.lobby_races.contains(self.race))
            && match self.build {
                Build::Catalog(catalog::Route::DeityDeath) => {
                    p.deity.is_some() && p.stats.tavern_spells_cast >= 8
                }
                Build::Catalog(catalog::Route::Barrage) => p.stats.gem_atk + p.stats.gem_hp >= 4,
                _ => true,
            }
    }

    fn deployed(&self, p: &Player) -> bool {
        self.pieces.iter().all(|group| has(&p.board, group))
    }

    fn can_replace_bridge(&self, p: &Player) -> bool {
        use catalog::Route::*;
        match self.build {
            Build::Catalog(UndeadAttack) => {
                let extra_attacks = p
                    .board
                    .iter()
                    .filter(|m| {
                        m.is(Races::UNDEAD)
                            && (m.has(Kw::REBORN)
                                || matches_group(
                                    catalog::id(m),
                                    &["BG25_010", "BG28_309", "BG25_009", "BG36_511", "BG36_515"],
                                ))
                    })
                    .count();
                extra_attacks >= 2
                    && (p.stats.undead_atk >= 8
                        || has(&p.board, &[catalog::TITUS, "BG36_511", "BG36_515"]))
            }
            Build::Catalog(MechSpell) => {
                let mut mechs = p.board.iter().filter(|m| m.is(Races::MECH));
                mechs.clone().take(2).count() >= 2 && mechs.any(|m| m.atk >= 25 || m.hp >= 25)
            }
            Build::Catalog(DeityDeath) => p.stats.tavern_spells_cast >= 12,
            _ => true,
        }
    }
}

const PACKAGES: [Package; 11] = {
    use Build::{Apm, Catalog, Quilboar, Tea};
    use apm::{PRIVATEER, ROGERS};
    use catalog::Route::{Barrage, DeityDeath, MechSpell, PirateGold, UndeadAttack};
    [
        Package {
            build: Catalog(UndeadAttack),
            race: Races::UNDEAD,
            pieces: &[&["BG32_324"], &["BG25_010", "BG28_309", "BG25_009"]],
        },
        Package {
            build: Catalog(Barrage),
            race: Races::QUILBOAR,
            pieces: &[&["BG34_682"], &["BG34_684", catalog::FELBOAR]],
        },
        Package {
            build: Catalog(DeityDeath),
            race: Races::ABERRATION,
            pieces: &[&["BG36_318"], &[catalog::TITUS, catalog::BIRD]],
        },
        Package {
            build: Catalog(MechSpell),
            race: Races::MECH,
            pieces: &[
                &["BG36_853"],
                &[catalog::BALINDA, catalog::DRONE, "BG36_854"],
            ],
        },
        Package {
            build: Quilboar,
            race: Races::QUILBOAR,
            pieces: &[&["BG31_323"], &["BG31_326", "BG36_331"]],
        },
        Package {
            build: Tea,
            race: Races::NONE,
            pieces: &[&["BG36_640"], &[catalog::BALINDA, "BG36_104"]],
        },
        Package {
            build: Apm(apm::Route::Undead),
            race: Races(Races::PIRATE.0 | Races::UNDEAD.0),
            pieces: &[&["BG34_692"], &[ROGERS], &[PRIVATEER]],
        },
        Package {
            build: Apm(apm::Route::Tavern),
            race: Races(Races::PIRATE.0 | Races::QUILBOAR.0),
            pieces: &[&[catalog::FELBOAR], &[ROGERS], &[PRIVATEER]],
        },
        Package {
            build: Apm(apm::Route::Mech),
            race: Races(Races::PIRATE.0 | Races::MECH.0),
            pieces: &[&["BG36_367"], &[ROGERS], &[PRIVATEER]],
        },
        Package {
            build: Apm(apm::Route::Deity),
            race: Races(Races::PIRATE.0 | Races::ABERRATION.0),
            pieces: &[&["BG36_109"], &[ROGERS], &[PRIVATEER]],
        },
        Package {
            build: Catalog(PirateGold),
            race: Races::PIRATE,
            pieces: &[&["BG36_344"], &[ROGERS], &[apm::ESCAPEE, PRIVATEER]],
        },
    ]
};

fn has(board: &[Minion], group: &[&str]) -> bool {
    group.iter().any(|card| catalog::has_key(board, card))
}

fn matches_group(card: CardId, group: &[&str]) -> bool {
    group.iter().any(|key| card == db().id(key))
}

fn has_in_hand(p: &Player, group: &[&str]) -> bool {
    p.hand.iter().any(|h| {
        h.locked == 0
            && h.item
                .as_minion()
                .is_some_and(|m| matches_group(catalog::id(m), group))
    })
}

#[derive(Clone, Copy)]
struct Offer {
    index: usize,
    cost: i32,
}

#[derive(Clone, Copy, Default)]
struct Readiness {
    deployed: i32,
    owned: i32,
    cost: i32,
    buy: Option<usize>,
}

fn readiness(game: &Game, seat: usize, package: &Package) -> Option<Readiness> {
    let p = &game.players[seat];
    if !package.fits(game, p) {
        return None;
    }
    let mut result = Readiness::default();
    for group in package.pieces {
        if has(&p.board, group) {
            result.deployed += 1;
            result.owned += 1;
            continue;
        }
        if has_in_hand(p, group) {
            result.owned += 1;
            continue;
        }
        if p.hand.len() >= game.cfg.hand_size {
            return None;
        }
        let offer = p
            .shop
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                let m = slot.item.as_minion()?;
                if !matches_group(catalog::id(m), group) {
                    return None;
                }
                let (cost, health, _) = game.slot_price(seat, index)?;
                (!health).then_some(Offer { index, cost })
            })
            .min_by_key(|offer| offer.cost)?;
        result.cost += offer.cost;
        result.buy.get_or_insert(offer.index);
    }
    (result.cost <= p.gold).then_some(result)
}

#[derive(Clone, Copy)]
struct Commitment {
    index: usize,
    turn: u32,
}

#[derive(Default)]
pub struct State {
    committed: Option<Commitment>,
    pub buy: Option<Action>,
    pub preserve_carry: bool,
}

pub fn search_for_six(player: &Player, build: Build) -> bool {
    const SIX_SUPPORT: &[&str] = &[
        catalog::BALINDA,
        "BG36_104",
        apm::PRIVATEER,
        apm::ESCAPEE,
        "BG31_326",
        "BG36_331",
    ];
    bridge(build).is_some() && (has(&player.board, SIX_SUPPORT) || has_in_hand(player, SIX_SUPPORT))
}

impl State {
    pub fn select(&mut self, game: &Game, seat: usize, previous: Build, usual: Build) -> Build {
        self.buy = None;
        self.preserve_carry = false;
        let p = &game.players[seat];
        if game.turn < 9 || p.tier < 5 {
            return usual;
        }
        if let Some(hold) = self.committed.take() {
            let target = &PACKAGES[hold.index];
            if (usual == target.build
                || usual == Build::General
                || bridge(usual).is_some()
                || game.turn == hold.turn)
                && let Some(ready) = readiness(game, seat, target)
            {
                return self.commit(hold, game.turn, ready);
            }
        }
        let source = bridge(if usual == Build::General {
            previous
        } else {
            usual
        });
        let Some(source) = source else { return usual };
        // A shop-only direction is too speculative to discard a working board.
        let life = p.health + p.armor;
        let pressured = life <= 16 || life <= 23 && p.stats.last_damage <= -8;
        let candidate = PACKAGES
            .iter()
            .enumerate()
            .filter_map(|(index, target)| {
                let compatible = target.race.intersects(source.race());
                let ready = readiness(game, seat, target)?;
                if !target.can_replace_bridge(p)
                    || ready.owned == 0
                    || (pressured && (!compatible || ready.deployed == 0))
                    || (!compatible && ready.deployed == 0 && life < 24)
                {
                    return None;
                }
                let score = ready.deployed * 5 + ready.owned * 8 - ready.cost
                    + 7 * i32::from(compatible)
                    + 5 * i32::from(target.pieces.len() == 3);
                Some((index, score, ready))
            })
            .max_by_key(|(index, score, _)| (*score, std::cmp::Reverse(*index)));
        if let Some((index, _, ready)) = candidate {
            return self.commit(
                Commitment {
                    index,
                    turn: game.turn,
                },
                game.turn,
                ready,
            );
        }
        usual
    }

    fn commit(&mut self, hold: Commitment, turn: u32, ready: Readiness) -> Build {
        self.committed = Some(hold);
        self.buy = ready.buy.map(Action::Buy);
        let target = &PACKAGES[hold.index];
        self.preserve_carry = turn <= hold.turn + 1 && ready.deployed < target.pieces.len() as i32;
        target.build
    }
}

pub fn name(build: Build) -> &'static str {
    match build {
        Build::General => "通用",
        Build::Azerite => "活化艾泽里特",
        Build::Tea => "喝茶流",
        Build::Quilboar => "野猪人 - 抉择",
        Build::Demon => "恶魔 - 自伤",
        Build::Apm(route) => route.name(),
        Build::Catalog(route) => route.spec().name,
    }
}

#[derive(Default)]
pub struct Trace {
    pub knight_first: Option<u32>,
    source: Option<catalog::Route>,
    last_strength: f64,
    pub conversion: Option<Conversion>,
    pub knight_at_end: bool,
}

pub struct Conversion {
    pub from: catalog::Route,
    pub to: Build,
    pub turn: u32,
    pub after: f64,
    pub before: f64,
    pub won: Option<bool>,
}

impl Trace {
    pub fn recruit_end(&mut self, game: &Game, seat: usize, build: Build) {
        let p = &game.players[seat];
        let strength = cadence::strength(&p.board);
        if game.turn <= 8
            && let Some(route) =
                bridge(build).filter(|route| catalog::has(&p.board, route.spec().anchor))
        {
            self.source = Some(route);
            if route == catalog::Route::Knight {
                self.knight_first.get_or_insert(game.turn);
            }
        }
        self.knight_at_end = build == Build::Catalog(catalog::Route::Knight)
            && catalog::has_key(&p.board, "BG25_008");
        if game.turn >= 9
            && p.tier >= 5
            && self.conversion.is_none()
            && let Some(from) = self.source
            && PACKAGES.iter().any(|project| {
                project.build == build && project.fits(game, p) && project.deployed(p)
            })
        {
            self.conversion = Some(Conversion {
                from,
                to: build,
                turn: game.turn,
                after: strength,
                before: self.last_strength,
                won: None,
            });
        }
        self.last_strength = strength;
    }

    pub fn after_combat(&mut self, game: &Game, subject: usize, turn: u32) {
        if let Some(conversion) = &mut self.conversion
            && conversion.turn == turn
            && conversion.won.is_none()
            && let Some(combat) = game
                .last_combats
                .iter()
                .find(|c| c.turn == turn && (c.a == subject || c.b == Some(subject)))
        {
            conversion.won = Some(
                (combat.a == subject && combat.winner == 0)
                    || (combat.b == Some(subject) && combat.winner == 1),
            );
        }
    }

    pub fn had_bridge(&self) -> bool {
        self.source.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RulesConfig;
    use crate::state::{Item, ShopSlot};

    const KNIGHT: Build = Build::Catalog(catalog::Route::Knight);
    const ATTACK: Build = Build::Catalog(catalog::Route::UndeadAttack);

    fn late_knights() -> Game {
        let mut game = Game::new(51, RulesConfig::default());
        game.lobby_races = LOBBY_MASK;
        game.turn = 9;
        let p = &mut game.players[0];
        p.tier = 5;
        p.health = 30;
        p.armor = 0;
        p.gold = 3;
        p.board = ["BG25_008", "BG25_009", "BG28_309"]
            .into_iter()
            .enumerate()
            .map(|(i, card)| Minion::new(i as u32 + 1, db().id(card)))
            .collect();
        p.shop.clear();
        game
    }

    fn offer(game: &mut Game, card: &str) {
        game.players[0].shop.push(ShopSlot::new(
            10,
            Item::Minion(Minion::new(10, db().id(card))),
        ));
    }

    #[test]
    fn bridge_waits_for_high_tier_an_affordable_core_and_an_engine() {
        let mut game = late_knights();
        let mut state = State::default();
        offer(&mut game, "BG32_324");
        assert_eq!(state.select(&game, 0, KNIGHT, KNIGHT), KNIGHT);
        game.players[0].stats.undead_atk = 8;
        game.turn = 8;
        assert_eq!(state.select(&game, 0, KNIGHT, KNIGHT), KNIGHT);
        game.turn = 9;
        game.players[0].gold = 2;
        assert_eq!(state.select(&game, 0, KNIGHT, KNIGHT), KNIGHT);
        game.players[0].gold = 3;
        assert_eq!(state.select(&game, 0, KNIGHT, KNIGHT), ATTACK);
        assert_eq!(state.buy, Some(Action::Buy(0)));
        assert!(state.preserve_carry);

        let p = &mut game.players[0];
        let acquired = p.shop.pop().unwrap();
        p.board.push(acquired.item.as_minion().unwrap().clone());
        assert_eq!(state.select(&game, 0, ATTACK, KNIGHT), ATTACK);
        assert!(state.buy.is_none());
        assert!(!state.preserve_carry);
    }

    #[test]
    fn low_life_keeps_the_bridge_instead_of_starting_cross_tribe() {
        let mut game = late_knights();
        game.players[0]
            .board
            .push(Minion::new(4, db().id(catalog::BALINDA)));
        offer(&mut game, "BG36_640");
        game.players[0].health = 14;
        assert_eq!(State::default().select(&game, 0, KNIGHT, KNIGHT), KNIGHT);
        game.players[0].health = 30;
        assert_eq!(
            State::default().select(&game, 0, KNIGHT, KNIGHT),
            Build::Tea
        );
    }
}
