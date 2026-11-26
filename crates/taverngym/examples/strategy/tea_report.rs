use serde::Serialize;
use std::io::{self, Write};
use taverngym::cards::db;
use taverngym::state::{Continuation, Pending, Player};
use taverngym::{Action, Game, cardid};

use crate::tea;

#[derive(Default)]
pub struct Formation {
    pub core_turn: Option<u32>,
    pub working_turn: Option<u32>,
    pub four_turn: Option<u32>,
    pub last: Option<Warband>,
    pub targeted_casts: u32,
    pub gem_casts: u32,
    pub other_attack_gain: i64,
    pub other_health_gain: i64,
}

#[derive(Serialize)]
pub struct Warband {
    turn: u32,
    tier: u8,
    hero: Option<&'static str>,
    pub pieces: u8,
    pub attack: i64,
    pub health: i64,
    pub targeted_casts: u32,
    gem_casts: u32,
    pub other_attack_gain: i64,
    pub other_health_gain: i64,
    spell_attack: i32,
    spell_health: i32,
    minions: Vec<MinionRow>,
}

#[derive(Serialize)]
struct MinionRow {
    id: &'static str,
    name: &'static str,
    golden: bool,
    attack: i32,
    health: i32,
    races: String,
}

pub struct Cast {
    target: u32,
    is_gem: bool,
    others: Vec<(u32, i32, i32)>,
}

impl Cast {
    pub fn before_action(game: &Game, seat: usize, action: Action) -> Option<Self> {
        let player = &game.players[seat];
        let Action::Choose(index) = action else {
            return None;
        };
        let Pending::Target {
            options,
            cont: Continuation::HandSpell { hid, .. },
        } = player.pending.last()?
        else {
            return None;
        };
        if !tea::aimed_at_gate(player, options, index, *hid) {
            return None;
        }
        let target = options[index];
        Some(Self {
            target,
            is_gem: player
                .hand
                .iter()
                .any(|h| h.hid == *hid && h.item.card() == cardid!("BG20_GEM")),
            others: player
                .board
                .iter()
                .filter(|m| m.eid != target)
                .map(|m| (m.eid, m.atk, m.hp))
                .collect(),
        })
    }
}

impl Formation {
    pub fn record_cast(&mut self, cast: Cast, player: &Player) {
        if !player.board.iter().any(|m| m.eid == cast.target) {
            return;
        }
        self.targeted_casts += 1;
        self.gem_casts += u32::from(cast.is_gem);
        // Count stats that appeared on the same other entities during this resolved action.
        // A spell with a second team-wide effect can contribute to this net gain too.
        for (id, attack, health) in cast.others {
            if let Some(minion) = player.board.iter().find(|m| m.eid == id) {
                self.other_attack_gain += (i64::from(minion.atk) - i64::from(attack)).max(0);
                self.other_health_gain += (i64::from(minion.hp) - i64::from(health)).max(0);
            }
        }
    }

    pub fn record_turn(&mut self, game: &Game, seat: usize, turn: u32, tier: u8) {
        let player = &game.players[seat];
        let board = &player.last_board;
        let pieces = tea::pieces(board);
        if tea::has_core(pieces) {
            self.core_turn.get_or_insert(turn);
            if self.targeted_casts >= 5 && self.other_health_gain >= 60 {
                self.working_turn.get_or_insert(turn);
            }
        }
        if pieces & tea::GUIDE_FOUR == tea::GUIDE_FOUR {
            self.four_turn.get_or_insert(turn);
        }
        self.last = Some(Warband {
            turn,
            tier,
            hero: player.hero.map(|id| db().get(id).name_zh.as_str()),
            pieces,
            attack: board.iter().map(|m| i64::from(m.atk)).sum(),
            health: board.iter().map(|m| i64::from(m.hp)).sum(),
            targeted_casts: self.targeted_casts,
            gem_casts: self.gem_casts,
            other_attack_gain: self.other_attack_gain,
            other_health_gain: self.other_health_gain,
            spell_attack: player.stats.spell_atk,
            spell_health: player.stats.spell_hp,
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
                        races: format!("{:?}", m.races),
                    }
                })
                .collect(),
        });
    }
}

pub fn write_board(out: &mut impl Write, board: &Warband) -> io::Result<()> {
    writeln!(
        out,
        "英雄 {}。第 {} 回合，{} 本，战团总攻击 {}、总生命 {}。",
        board.hero.unwrap_or("未知"),
        board.turn,
        board.tier,
        board.attack,
        board.health
    )?;
    writeln!(
        out,
        "\n累计从手牌手动指向守门融合怪施法 {} 次，其中鲜血宝石 {} 次。同一动作期间其他在场随从累计增加 {} 攻／{} 血。酒馆法术额外 +{}/+{}。",
        board.targeted_casts,
        board.gem_casts,
        board.other_attack_gain,
        board.other_health_gain,
        board.spell_attack,
        board.spell_health
    )?;
    writeln!(
        out,
        "\n| 站位 | 随从 | 攻击 | 生命 |\n| ---: | --- | ---: | ---: |"
    )?;
    for (i, m) in board.minions.iter().enumerate() {
        writeln!(
            out,
            "| {} | {}{} | {} | {} |",
            i + 1,
            m.name,
            if m.golden { "〈金色〉" } else { "" },
            m.attack,
            m.health
        )?;
    }
    Ok(())
}
