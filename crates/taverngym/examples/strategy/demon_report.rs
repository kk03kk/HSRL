use serde::Serialize;
use std::io::{self, Write};
use taverngym::cards::db;
use taverngym::state::{Continuation, Pending, Player};
use taverngym::types::EntityId;
use taverngym::{Action, Game, cardid};

use crate::demon;

#[derive(Default)]
pub struct Formation {
    pub rewinder_turn: Option<u32>,
    pub engine_turn: Option<u32>,
    pub working_turn: Option<u32>,
    pub six_turn: Option<u32>,
    pub damage: i32,
    pub safely_rewound: i32,
    pub unrecovered_damage: i32,
    pub cupcakes_gained: u32,
    pub targeted: u32,
    pub off_target: u32,
    pub missed_distractor: u32,
    pub health_refreshes: u32,
    pub percussion_plays: u32,
    pub engine_casts: u32,
    pub spell_shop_atk: i32,
    pub spell_shop_hp: i32,
    pub cupcakes_consumed: u32,
    pub last: Option<Warband>,
}

pub struct Witness {
    damage: i32,
    life: i32,
    cupcakes: u32,
    pieces: u8,
    shop_atk: i32,
    shop_hp: i32,
    donors: u32,
    target: Option<(bool, bool)>,
    refreshed_with_malchezaar: bool,
    played_percussionist: bool,
}

fn cupcake_count(p: &Player) -> u32 {
    p.hand
        .iter()
        .filter(|h| h.item.card() == demon::cupcakes())
        .count() as u32
}

fn donor_count(p: &Player) -> u32 {
    p.shop
        .iter()
        .filter(|s| s.item.as_minion().is_some())
        .count() as u32
}

impl Witness {
    pub fn before_action(game: &Game, seat: usize, action: Action) -> Option<Self> {
        if action == Action::EndTurn {
            return None;
        }
        let p = &game.players[seat];
        let target = if let Action::Choose(index) = action
            && let Some(Pending::Target {
                options,
                cont: Continuation::HandSpell { hid, .. },
            }) = p.pending.last()
            && p.hand
                .iter()
                .any(|h| h.hid == *hid && h.item.card() == demon::cupcakes())
        {
            let distractor = |target: &EntityId| {
                p.board
                    .iter()
                    .any(|m| m.eid == *target && demon::part(m.card) == demon::DISTRACTOR)
            };
            options
                .get(index)
                .map(|target| (distractor(target), options.iter().any(distractor)))
        } else {
            None
        };
        let pieces = demon::pieces(&p.board);
        Some(Self {
            damage: p.stats.damage_taken_outside_combat,
            life: p.health + p.armor,
            cupcakes: if pieces & demon::ESCAPIST != 0 {
                cupcake_count(p)
            } else {
                0
            },
            pieces,
            shop_atk: p.stats.tavern_atk,
            shop_hp: p.stats.tavern_hp,
            donors: target.map_or(0, |_| donor_count(p)),
            target,
            refreshed_with_malchezaar: action == Action::Refresh
                && game.refresh_price(seat) > 0
                && game.malchezaar(seat).is_some(),
            played_percussionist: matches!(action, Action::Play { hand, .. }
                if p.hand.get(hand).is_some_and(|h| db().normal_of(h.item.card()) == cardid!("BG26_525"))),
        })
    }

    pub fn record(self, formation: &mut Formation, p: &Player) {
        let damage = p.stats.damage_taken_outside_combat - self.damage;
        formation.health_refreshes += u32::from(self.refreshed_with_malchezaar && damage > 0);
        formation.percussion_plays += u32::from(self.played_percussionist);
        if damage > 0 {
            formation.damage += damage;
            let life_loss = (self.life - (p.health + p.armor)).clamp(0, damage);
            formation.unrecovered_damage += life_loss;
            if life_loss == 0 && demon::has_rewind(self.pieces | demon::pieces(&p.board)) {
                formation.safely_rewound += damage;
            }
            if self.pieces & demon::ESCAPIST != 0 {
                formation.cupcakes_gained += cupcake_count(p).saturating_sub(self.cupcakes);
            }
        }
        if let Some((targeted, could_target)) = self.target {
            if targeted {
                formation.targeted += 1;
                let attack = (p.stats.tavern_atk - self.shop_atk).max(0);
                let health = (p.stats.tavern_hp - self.shop_hp).max(0);
                formation.spell_shop_atk += attack;
                formation.spell_shop_hp += health;
                formation.cupcakes_consumed += self.donors.saturating_sub(donor_count(p));
                if demon::has_engine(self.pieces) && attack > 0 && health > 0 {
                    formation.engine_casts += 1;
                }
            } else {
                formation.off_target += 1;
                formation.missed_distractor += u32::from(could_target);
            }
        }
    }
}

#[derive(Serialize)]
pub struct Warband {
    turn: u32,
    tier: u8,
    hero: Option<&'static str>,
    pub pieces: u8,
    pub attack: i64,
    pub health: i64,
    tavern_attack: i32,
    tavern_health: i32,
    rewound: i32,
    unrecovered_damage: i32,
    cupcakes_gained: u32,
    pub targeted: u32,
    off_target: u32,
    health_refreshes: u32,
    percussion_plays: u32,
    cupcake_tavern_attack: i32,
    cupcake_tavern_health: i32,
    pub cupcake_donors_consumed: u32,
    pub preferred_cycle_minion: bool,
    minions: Vec<MinionRow>,
}

#[derive(Serialize)]
struct MinionRow {
    id: &'static str,
    name: &'static str,
    golden: bool,
    attack: i32,
    health: i32,
}

impl Formation {
    pub fn record_turn(&mut self, game: &Game, seat: usize, turn: u32, tier: u8) {
        let p = &game.players[seat];
        let board = &p.last_board;
        let pieces = demon::pieces(board);
        if pieces & demon::REWINDER != 0 {
            self.rewinder_turn.get_or_insert(turn);
        }
        if demon::has_engine(pieces) {
            self.engine_turn.get_or_insert(turn);
            if self.engine_casts >= 2 && self.cupcakes_gained > 0 {
                self.working_turn.get_or_insert(turn);
            }
        }
        if pieces & demon::SIX == demon::SIX {
            self.six_turn.get_or_insert(turn);
        }
        self.last = Some(Warband {
            turn,
            tier,
            hero: p.hero.map(|id| db().get(id).name_zh.as_str()),
            pieces,
            attack: board.iter().map(|m| i64::from(m.atk)).sum(),
            health: board.iter().map(|m| i64::from(m.hp)).sum(),
            tavern_attack: p.stats.tavern_atk,
            tavern_health: p.stats.tavern_hp,
            rewound: self.safely_rewound,
            unrecovered_damage: self.unrecovered_damage,
            cupcakes_gained: self.cupcakes_gained,
            targeted: self.targeted,
            off_target: self.off_target,
            health_refreshes: self.health_refreshes,
            percussion_plays: self.percussion_plays,
            cupcake_tavern_attack: self.spell_shop_atk,
            cupcake_tavern_health: self.spell_shop_hp,
            cupcake_donors_consumed: self.cupcakes_consumed,
            preferred_cycle_minion: board
                .iter()
                .any(|m| m.is_card_id(cardid!("BG26_524")) || m.is_card_id(cardid!("BG26_525"))),
            minions: board
                .iter()
                .map(|m| {
                    let card = db().get(m.card);
                    MinionRow {
                        id: &card.id,
                        name: &card.name_zh,
                        golden: m.golden,
                        attack: m.atk,
                        health: m.hp,
                    }
                })
                .collect(),
        });
    }
}

pub fn write_board(out: &mut impl Write, board: &Warband) -> io::Result<()> {
    writeln!(
        out,
        "\n英雄 {}。第 {} 回合，{} 本，战团总攻击 {}、总生命 {}。\n",
        board.hero.unwrap_or("未知"),
        board.turn,
        board.tier,
        board.attack,
        board.health,
    )?;
    writeln!(
        out,
        "观察到 {} 点伤害在无净损失的动作中回溯，未回收伤害 {} 点。自伤结算期间手中净增加了 {} 张糕点。\n",
        board.rewound, board.unrecovered_damage, board.cupcakes_gained,
    )?;
    writeln!(
        out,
        "玛克扎尔以生命刷新 {} 次，使用奇瑰打击乐手 {} 次。\n",
        board.health_refreshes, board.percussion_plays,
    )?;
    writeln!(
        out,
        "{} 张糕点对恶魔干扰者施放、{} 张给其他目标。定向施放动作中永久酒馆加成共增加 +{}/+{}，酒馆随从净减少 {} 个。终局永久酒馆加成 +{}/+{}。\n",
        board.targeted,
        board.off_target,
        board.cupcake_tavern_attack,
        board.cupcake_tavern_health,
        board.cupcake_donors_consumed,
        board.tavern_attack,
        board.tavern_health,
    )?;
    writeln!(
        out,
        "| 站位 | 随从 | 攻击 | 生命 |\n| ---: | --- | ---: | ---: |"
    )?;
    for (i, minion) in board.minions.iter().enumerate() {
        writeln!(
            out,
            "| {} | {}{} | {} | {} |",
            i + 1,
            minion.name,
            if minion.golden { "〈金色〉" } else { "" },
            minion.attack,
            minion.health,
        )?;
    }
    Ok(())
}
