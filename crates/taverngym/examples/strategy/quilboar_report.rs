use serde::Serialize;
use std::io::{self, Write};
use taverngym::cards::db;
use taverngym::state::{Continuation, Pending, Player};
use taverngym::types::Races;
use taverngym::{Action, Game, cardid};

use crate::quilboar;

#[derive(Default)]
pub struct Formation {
    pub core_turn: Option<u32>,
    pub working_turn: Option<u32>,
    pub five_turn: Option<u32>,
    pub gem_waves: u32,
    pub gem_day_waves: u32,
    pub added_gems: u64,
    pub last: Option<Warband>,
}

struct StoredGemStats {
    eid: u32,
    gems: u16,
}

pub struct Choice {
    gem_day: bool,
    targets: Vec<StoredGemStats>,
}

impl Choice {
    pub fn before_action(game: &Game, seat: usize, action: Action) -> Option<Self> {
        let player = &game.players[seat];
        if !player
            .board
            .iter()
            .any(|m| m.is_card_id(cardid!("BG31_323")))
        {
            return None;
        }
        let card = match action {
            Action::Play { hand, .. } => player.hand.get(hand)?.item.card(),
            Action::Choose(_) => match player.pending.last()? {
                Pending::ChooseOne {
                    card,
                    cont: Continuation::HandSpell { .. } | Continuation::Battlecry { .. },
                } => *card,
                Pending::Target {
                    cont: Continuation::HandSpell { hid, .. },
                    ..
                } => player.hand.iter().find(|h| h.hid == *hid)?.item.card(),
                _ => return None,
            },
            _ => return None,
        };
        if !quilboar::is_choose_one(card) {
            return None;
        }
        Some(Self {
            gem_day: card == cardid!("BG31_893"),
            targets: player
                .board
                .iter()
                .filter(|m| m.is(Races::QUILBOAR))
                .map(|m| StoredGemStats {
                    eid: m.eid,
                    gems: m.gems,
                })
                .collect(),
        })
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
    gem_attack: i32,
    gem_health: i32,
    pub gem_waves: u32,
    gem_day_waves: u32,
    added_gems: u64,
    minions: Vec<MinionRow>,
}

#[derive(Serialize)]
struct MinionRow {
    id: &'static str,
    name: &'static str,
    golden: bool,
    attack: i32,
    health: i32,
    gems: u16,
}

impl Formation {
    pub fn record_choice(&mut self, choice: Choice, player: &Player) {
        let (mut recipients, mut total) = (0, 0);
        for old in choice.targets {
            if let Some(new) = player
                .board
                .iter()
                .find(|m| m.eid == old.eid && m.gems > old.gems)
            {
                recipients += 1;
                total += u64::from(new.gems - old.gems);
            }
        }
        if recipients >= 2 {
            self.gem_waves += 1;
            self.gem_day_waves += u32::from(choice.gem_day);
            self.added_gems += total;
        }
    }

    pub fn record_turn(&mut self, game: &Game, seat: usize, turn: u32, tier: u8) {
        let p = &game.players[seat];
        let board = &p.last_board;
        let pieces = quilboar::pieces(board);
        if quilboar::has_core(pieces) {
            self.core_turn.get_or_insert(turn);
            if self.gem_waves >= 3 {
                self.working_turn.get_or_insert(turn);
            }
        }
        if pieces == quilboar::GUIDE_FIVE {
            self.five_turn.get_or_insert(turn);
        }
        self.last = Some(Warband {
            turn,
            tier,
            hero: p.hero.map(|id| db().get(id).name_zh.as_str()),
            pieces,
            attack: board.iter().map(|m| i64::from(m.atk)).sum(),
            health: board.iter().map(|m| i64::from(m.hp)).sum(),
            gem_attack: 1 + p.stats.gem_atk,
            gem_health: 1 + p.stats.gem_hp,
            gem_waves: self.gem_waves,
            gem_day_waves: self.gem_day_waves,
            added_gems: self.added_gems,
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
                        gems: m.gems,
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
        "\n鲜血宝石基础身材 +{}/+{}。累计 {} 次抉择结算使至少两只已在场的野猪人得到宝石，其中宝石日 {} 次。这些结算中，已有随从净增 {} 颗宝石。\n",
        board.gem_attack, board.gem_health, board.gem_waves, board.gem_day_waves, board.added_gems
    )?;
    writeln!(
        out,
        "| 站位 | 随从 | 攻击 | 生命 | 现存宝石 |\n| ---: | --- | ---: | ---: | ---: |"
    )?;
    for (i, m) in board.minions.iter().enumerate() {
        writeln!(
            out,
            "| {} | {}{} | {} | {} | {} |",
            i + 1,
            m.name,
            if m.golden { "〈金色〉" } else { "" },
            m.attack,
            m.health,
            m.gems
        )?;
    }
    Ok(())
}
