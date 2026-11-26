use serde::Serialize;
use std::io::{self, Write};
use taverngym::cards::db;
use taverngym::state::{Continuation, Item, Pending};
use taverngym::types::Races;
use taverngym::{Action, Game, cardid};

use crate::azerite::{self, CORE, ENGINE, FULL};

#[derive(Default)]
pub struct Formation {
    pub core_turn: Option<u32>,
    pub engine_turn: Option<u32>,
    pub working_turn: Option<u32>,
    pub full_turn: Option<u32>,
    pub big_full_turn: Option<u32>,
    pub last: Option<Warband>,
    pub seen_pieces: u8,
    pub absorption_casts: u32,
    pub kelp_into_surfacer: u32,
}

#[derive(Serialize)]
pub struct Warband {
    pub turn: u32,
    pub tier: u8,
    pub pieces: u8,
    pub lobby_types: Vec<&'static str>,
    pub attack: i64,
    pub health: i64,
    pub tavern_elemental_attack: i32,
    pub tavern_elemental_health: i32,
    pub tavern_spells_cast: i32,
    pub absorption_casts: u32,
    pub kelp_into_surfacer: u32,
    pub minions: Vec<MinionRow>,
}

#[derive(Serialize)]
pub struct MinionRow {
    id: &'static str,
    name: &'static str,
    golden: bool,
    attack: i32,
    health: i32,
}

pub enum Witness {
    Absorption,
    KelpIntoSurfacer,
}

impl Witness {
    pub fn before_action(game: &Game, seat: usize, action: Action) -> Option<Self> {
        let player = &game.players[seat];
        let Action::Choose(index) = action else {
            return None;
        };
        let Pending::Target { options, cont } = player.pending.last()? else {
            return None;
        };
        match cont {
            Continuation::HandSpell { hid, .. } if player.hand.iter().any(|h| {
                h.hid == *hid && matches!(h.item, Item::Spell { card, .. } if card == azerite::absorption())
            }) => Some(Self::Absorption),
            Continuation::Activate { eid }
                if player.board.iter().any(|m| m.eid == *eid && m.is_card_id(cardid!("BG36_701")))
                    && options.get(index).is_some_and(|target| player.board.iter().any(|m| {
                        m.eid == *target && m.is_card_id(cardid!("BG35_881"))
                    })) => Some(Self::KelpIntoSurfacer),
            _ => None,
        }
    }
}

impl Formation {
    pub fn record_action(&mut self, witness: Witness, game: &Game, seat: usize) {
        match witness {
            Witness::Absorption => self.absorption_casts += 1,
            Witness::KelpIntoSurfacer => {
                self.kelp_into_surfacer += 1;
                let player = &game.players[seat];
                if azerite::pieces(&player.board) & ENGINE == ENGINE
                    && player.stats.tavern_elemental_atk > 0
                    && player.stats.tavern_elemental_hp > 0
                {
                    self.working_turn.get_or_insert(game.turn);
                }
            }
        }
    }

    pub fn record_turn(&mut self, game: &Game, seat: usize, turn: u32, tier: u8) {
        let player = &game.players[seat];
        let board = &player.last_board;
        let mask = azerite::pieces(board);
        self.seen_pieces |= mask;
        if mask & CORE == CORE {
            self.core_turn.get_or_insert(turn);
        }
        if mask & ENGINE == ENGINE {
            self.engine_turn.get_or_insert(turn);
        }
        if mask == FULL {
            self.full_turn.get_or_insert(turn);
            if board
                .iter()
                .any(|m| m.is(Races::ELEMENTAL) && m.atk >= 100 && m.hp >= 100)
            {
                self.big_full_turn.get_or_insert(turn);
            }
        }
        self.last = Some(Warband {
            turn,
            tier,
            pieces: mask,
            lobby_types: game
                .lobby_races
                .iter()
                .filter_map(Races::index)
                .map(|i| Races::NAMES[i])
                .collect(),
            attack: board.iter().map(|m| i64::from(m.atk)).sum(),
            health: board.iter().map(|m| i64::from(m.hp)).sum(),
            tavern_elemental_attack: player.stats.tavern_elemental_atk,
            tavern_elemental_health: player.stats.tavern_elemental_hp,
            tavern_spells_cast: player.stats.tavern_spells_cast,
            absorption_casts: self.absorption_casts,
            kelp_into_surfacer: self.kelp_into_surfacer,
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

pub fn summarize<'a>(
    label: &str,
    games: impl Iterator<Item = &'a Formation>,
    out: &mut impl Write,
) -> io::Result<()> {
    let (mut n, mut core, mut engine, mut full, mut big, mut full_final, mut casts, mut kelp) =
        (0, 0, 0, 0, 0, 0, 0, 0);
    for log in games {
        n += 1;
        core += usize::from(log.core_turn.is_some());
        engine += usize::from(log.engine_turn.is_some());
        full += usize::from(log.full_turn.is_some());
        big += usize::from(log.big_full_turn.is_some());
        full_final += usize::from(log.last.as_ref().is_some_and(|b| b.pieces == FULL));
        casts += log.absorption_casts;
        kelp += log.kelp_into_surfacer;
    }
    if n == 0 {
        return Ok(());
    }
    let rate = |count: usize| 100.0 * count as f64 / n as f64;
    writeln!(
        out,
        "{label}: {n} games; Azerite + Brann {core} ({:.1}%); four-piece engine {engine} ({:.1}%); five-piece {full} ({:.1}%); five-piece + a 100/100 Elemental {big} ({:.1}%); five-piece on final board {full_final} ({:.1}%)",
        rate(core),
        rate(engine),
        rate(full),
        rate(big),
        rate(full_final)
    )?;
    if label != "Original" {
        writeln!(
            out,
            "  direct Arcane Absorption casts: {casts}; Kelp Keeper activations targeting Surfacer: {kelp}"
        )?;
    }
    Ok(())
}

pub fn write_board(out: &mut impl Write, board: &Warband) -> io::Result<()> {
    writeln!(
        out,
        "第 {} 回合，{} 本，战团总攻击 {}、总生命 {}。商店元素永久加成 +{}/+{}。",
        board.turn,
        board.tier,
        board.attack,
        board.health,
        board.tavern_elemental_attack,
        board.tavern_elemental_health
    )?;
    writeln!(
        out,
        "\n当时累计施放 {} 张奥术吸收，海藻护卫对魔网浮灵激活 {} 次。\n",
        board.absorption_casts, board.kelp_into_surfacer
    )?;
    writeln!(
        out,
        "| 攻击顺序 | 随从 | 攻击 | 生命 |\n| ---: | --- | ---: | ---: |"
    )?;
    for (i, m) in board.minions.iter().enumerate() {
        let golden = if m.golden { "〈金色〉" } else { "" };
        writeln!(
            out,
            "| {} | {}{} | {} | {} |",
            i + 1,
            m.name,
            golden,
            m.attack,
            m.health
        )?;
    }
    Ok(())
}
