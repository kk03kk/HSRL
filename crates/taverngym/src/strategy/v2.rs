use crate::cards::CardId;
use crate::script::script;
use crate::state::{Continuation, Minion, Pending, Player};
use crate::types::{Kw, Races};
use crate::{Action, Game};

use super::{apm, azerite, cadence, catalog, catalog_plan, demon, guide_bot::Build, quilboar, tea};

pub struct Recruit<'a> {
    p: &'a Player,
    turn: u32,
    life: i32,
    tempo: f64,
    pressure: bool,
}

impl<'a> Recruit<'a> {
    pub fn new(game: &'a Game, seat: usize) -> Self {
        let p = &game.players[seat];
        let life = p.health + p.armor;
        let tempo = cadence::strength(&p.board);
        let behind = p.next_opponent.is_some_and(|opponent| {
            let (board, seen) = game.known_board(seat, opponent);
            seen > 0
                && game.turn.saturating_sub(seen) <= 2
                && cadence::strength(board) > tempo * 1.30 + 14.0
        });
        let pressure =
            game.turn >= 6 && (life <= 13 || life <= 24 && (p.stats.last_damage <= -7 || behind));
        Self {
            p,
            turn: game.turn,
            life,
            tempo,
            pressure,
        }
    }

    pub fn upgrade(&self, game: &Game, seat: usize, build: Build, prior: bool) -> bool {
        if self.turn < 7 || self.p.tier < 4 || self.p.tier >= 6 {
            return prior;
        }
        let price = game.upgrade_price(seat);
        let spare = self.p.gold - price;
        let can_buy = spare >= game.minion_price(seat);
        let board = self.p.board.len();
        if self.p.tier == 4 {
            return prior
                || self.turn == 7
                    && !self.pressure
                    && self.life >= 24
                    && board >= 5
                    && self.tempo >= 68.0
                    && (can_buy || self.p.stats.last_damage >= 0);
        }
        let needs_six = match build {
            Build::Catalog(route) => {
                catalog_plan::Plan::new(self.p, route, self.turn).needs_six(game.lobby_races)
            }
            Build::Azerite => azerite::pieces(&self.p.board) & azerite::AZERITE == 0,
            Build::Quilboar => quilboar::pieces(&self.p.board) & quilboar::RIDER == 0,
            Build::Tea => !tea::has_core(tea::pieces(&self.p.board)),
            Build::Demon => demon::needs_six(&self.p.board),
            Build::Apm(route) => !apm::economy(&self.p.board) || !route.payoff(&self.p.board),
            Build::General => true,
        };
        if self.pressure {
            return needs_six
                && self.turn >= 11
                && price <= 3
                && can_buy
                && board >= 6
                && self.tempo >= 125.0;
        }
        if !needs_six {
            return self.turn >= 11 && price <= 4 && can_buy && self.tempo >= 145.0;
        }
        prior
            || self.turn >= 9
                && self.life >= 25
                && board >= 5
                && self.tempo >= 95.0
                && (can_buy || self.tempo >= 130.0 && self.p.stats.last_damage >= 0)
    }

    pub fn save_triple(&self, m: &Minion, level_this_turn: bool) -> bool {
        m.golden
            && m.triple_reward
            && level_this_turn
            && matches!(self.p.tier, 3 | 4)
            && self.p.board.len() >= 4
            && !self.pressure
    }

    pub fn defer_gift(&self) -> bool {
        self.turn == 4
            && self.p.dark_gift_uses == 1
            && self.life >= 30
            && self.p.board.len() >= 3
            && self.tempo >= 33.0
            && self.p.stats.last_damage >= 0
    }

    pub fn keep_bonus(&self, m: &Minion) -> f64 {
        let key = catalog::key(m.card);
        let p = self.p;
        let copies = |key| p.board.iter().filter(|b| catalog::is(b, key)).count();
        let count = |race| p.board.iter().filter(|b| b.is(race)).count();
        let synergy = match key {
            "BG36_523" if self.turn <= 13 => 4.0 + 4.0 * f64::from(p.tier >= 5),
            "BG33_823" if copies("BG33_825") > 0 => 14.0,
            "BG33_825" if copies("BG33_823") > 0 => 12.0,
            "BG31_323" if p.stats.gem_atk + p.stats.gem_hp >= 4 => 9.0,
            "BG35_883" if copies("BG36_640") + copies("BG36_762") + copies("BG30_121") > 0 => 11.0,
            "BG34_692" if copies("BG33_823") > 0 => 10.0,
            "BG28_707" if self.turn <= 11 && count(Races::ELEMENTAL) >= 1 => 8.0,
            "BG26_ICC_901" => {
                let engines = p
                    .board
                    .iter()
                    .filter(|b| {
                        b.eid != m.eid
                            && !catalog::is(b, key)
                            && script(b.card).is_some_and(|s| s.end_of_turn.is_some())
                    })
                    .count();
                (3.5 * engines as f64).min(12.0)
            }
            _ => 0.0,
        };
        if copies(key) > usize::from(p.board.iter().any(|b| b.eid == m.eid)) && !m.golden {
            synergy * 0.40
        } else {
            synergy
        }
    }

    pub fn spell_bonus(&self, card: CardId) -> f64 {
        let p = self.p;
        let board = &p.board;
        let n = board.len() as f64;
        let goldens = board.iter().filter(|m| m.golden).count() as f64;
        let shield = board.iter().filter(|m| m.has(Kw::DIVINE_SHIELD)).count() as f64;
        let dragons = board.iter().filter(|m| m.is(Races::DRAGON)).count() as f64;
        let types = board
            .iter()
            .fold(Races::NONE, |all, m| all.union(m.races))
            .iter()
            .count();
        let gain = match catalog::key(card) {
            "BG28_169" => 8.0 * n,
            "BG34_990" => 6.0 * (n + goldens),
            "BG36_246" => 5.0 * (n + dragons + shield),
            "BG28_888" => 6.0 * types.min(board.len()) as f64,
            "BG31_881" => {
                if self.pressure {
                    4.0 * n
                } else {
                    7.0 * n
                }
            }
            "BG31_886" => (4.0 * n).max(24.0 * f64::from(n > 0.0)),
            "BG28_825" if n > 0.0 => 17.0,
            "BG36_624" if n > 0.0 => 12.0,
            "BG28_838" => board
                .iter()
                .map(|m| (40 - m.atk - m.hp).max(0))
                .max()
                .unwrap_or(0) as f64,
            "BG28_845" => {
                4.0 * board
                    .iter()
                    .map(|m| board.iter().filter(|b| b.races.intersects(m.races)).count())
                    .max()
                    .unwrap_or(0) as f64
            }
            _ => 0.0,
        };
        let upside = (gain
            * if self.pressure || self.turn <= 8 {
                0.27
            } else {
                0.19
            })
        .min(12.0);
        upside
            + match catalog::key(card) {
                "BG28_827" if p.free_refreshes < 4 => 2.0,
                "BG28_805" if (5..=8).contains(&self.turn) && !self.pressure && self.life >= 26 => {
                    3.5
                }
                "BG28_884"
                    if !self.pressure
                        && self.turn <= 8
                        && self.tempo >= 12.0 * f64::from(self.turn) =>
                {
                    4.0
                }
                "BG28_GIL_836" if apm::has(board, catalog::BRANN) || p.tier >= 5 => 6.5,
                "BG31_890" | "BG34_330" if p.tier >= 5 && !self.pressure => 6.0,
                "BG28_500" if p.armor < 3 && self.life <= 13 => 7.0,
                "BG28_838" if gain < 6.0 => -5.0,
                _ => 0.0,
            }
    }

    pub fn can_cast(&self, card: CardId) -> bool {
        catalog::key(card) != "BG28_838" || self.p.board.iter().any(|m| m.atk + m.hp < 38)
    }

    pub fn pending(&self, menu: &Pending, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        let p = self.p;
        if let Pending::ChooseOne { card, .. } = menu {
            let choice = match catalog::key(*card) {
                "BG31_881" => usize::from(!self.pressure && p.board.len() >= 4),
                "BG31_886" => usize::from(p.board.len() >= 6),
                "BG31_890" => 0,
                "BG31_880" => 1,
                "BG36_332" => usize::from(!self.pressure && self.turn <= 12),
                _ => return None,
            };
            return legal(Action::Choose(choice)).then_some(Action::Choose(choice));
        }
        let Pending::Target {
            cont: Continuation::HandSpell { hid, .. },
            options,
        } = menu
        else {
            return None;
        };
        let card = catalog::key(p.hand.iter().find(|h| h.hid == *hid)?.item.card());
        let candidates = options
            .iter()
            .enumerate()
            .filter(|(i, _)| legal(Action::Choose(*i)))
            .filter_map(|(i, eid)| p.board.iter().find(|m| m.eid == *eid).map(|m| (i, m)));
        let score = |m: &Minion| -> Option<f64> {
            let deathrattle = script(m.card).is_some_and(|s| s.deathrattle.is_some());
            let power = cadence::minion_strength(m);
            Some(match card {
                "BG28_838" => {
                    f64::from(40 - m.atk - m.hp) + 5.0 * f64::from(m.has(Kw::DIVINE_SHIELD))
                }
                "BG28_503" | "BG28_825" => {
                    power.sqrt() + 8.0 * f64::from(deathrattle || m.has(Kw::REBORN))
                        - 10.0
                            * f64::from(matches!(
                                catalog::key(m.card),
                                catalog::TITUS | catalog::BRANN | catalog::DRAKKARI
                            ))
                }
                "BG28_845" => p
                    .board
                    .iter()
                    .filter(|b| b.races.intersects(m.races))
                    .count() as f64,
                "BG28_897" | "BG36_624" | "BG32_337" | "BG36_883" | "BG31_880" => {
                    power.sqrt()
                        + 8.0 * f64::from(m.has(Kw::DIVINE_SHIELD) || m.has(Kw::CLEAVE))
                        + 2.0 * f64::from(m.golden)
                }
                "BG28_507" => f64::from(m.atk) + 6.0 * f64::from(m.has(Kw::CLEAVE)),
                _ => return None,
            })
        };
        candidates
            .filter_map(|(i, m)| score(m).map(|v| (i, v)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| Action::Choose(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Phase;
    use crate::{RulesConfig, cardid};

    fn setup() -> Game {
        let mut game = Game::new(171, RulesConfig::default());
        game.phase = Phase::Recruit;
        game.turn = 9;
        let p = &mut game.players[0];
        p.pending.clear();
        p.tier = 5;
        p.upgrade_cost = 8;
        p.gold = 10;
        p.health = 30;
        p.armor = 0;
        p.board = (0..5)
            .map(|i| {
                let mut m = Minion::new(300 + i, cardid!("BG25_001"));
                m.atk = 15;
                m.hp = 12;
                m
            })
            .collect();
        game
    }

    #[test]
    fn sixth_tier_requires_a_buffer_or_an_actual_endgame_need() {
        let mut game = setup();
        assert!(Recruit::new(&game, 0).upgrade(&game, 0, Build::General, false));
        game.players[0].health = 10;
        assert!(!Recruit::new(&game, 0).upgrade(&game, 0, Build::General, true));
    }

    #[test]
    fn stat_setter_targets_the_actual_upgrade() {
        let mut game = setup();
        game.players[0].board.truncate(2);
        let p = &mut game.players[0];
        p.board[0].atk = 100;
        p.board[0].hp = 100;
        let target = p.board[1].eid;
        game.with_recruit(0, |ctx| ctx.give_card(0, cardid!("BG28_838")));
        game.apply(0, Action::Play { hand: 0, pos: 0 }).unwrap();
        let pending = game.players[0].pending.last().unwrap();
        let Pending::Target { options, .. } = pending else {
            panic!("target")
        };
        let option = options.iter().position(|&eid| eid == target).unwrap();
        let mask = game.legal_mask(0);
        assert_eq!(
            Recruit::new(&game, 0).pending(pending, &|a| mask[a.encode()]),
            Some(Action::Choose(option))
        );
    }
}
