use taverngym::content::is_bounty;
use taverngym::state::{Continuation, Item, Pending, Player};
use taverngym::types::Kw;
use taverngym::{Action, Game};

use super::apm::{self, Route};
use super::apm_report::Warband;
use super::catalog;
use super::undead_apm::{
    self, BELLRINGER, BUTCHERING, GRAVEROBBER, HANDLESS, MAW_CASTER, PHANTOM, WEAVER,
};

const SACRIFICES: [&str; 4] = [BELLRINGER, BUTCHERING, MAW_CASTER, GRAVEROBBER];

#[derive(Default)]
pub struct Proof {
    pub trio: Option<u32>,
    pub same_turn_loop: Option<u32>,
    pub full_with_loop: Option<u32>,
    pub golden_reborn: Option<u32>,
    pub final_full: bool,
    pub bellringer_gains: u32,
    pub phantom_health_added: i32,
    pub peak_spend: i32,
    pub peak_spell_growth: i32,
    pub finance_buys: u32,
    pub trio_gold_spent: i32,
    pub trio_gold_inflow: i32,
    pub sacrifice_growth: [u32; 4],
    pub permanent_phantom_hp: i32,
    pub peak_phantom_gain: i32,
    pub core_copies: u32,
    pub max_phantom_hp: i32,
    pub tallest_phantom: Option<Warband>,
    pub best: Option<Warband>,
    turn: u32,
    spend: i32,
    refund: i32,
    bounties: u32,
    spell_growth: i32,
}

fn weaver_attack_per_cast(p: &Player) -> i32 {
    p.board
        .iter()
        .filter(|m| catalog::is(m, WEAVER))
        .map(|m| if m.golden { 6 } else { 3 })
        .sum()
}

#[derive(Default)]
pub struct Witness {
    economy: Option<(i32, i32, i32, i32)>,
    bounty: bool,
    sacrifice: Option<(usize, i32)>,
    core: Option<(&'static str, u32)>,
    finance: bool,
}

fn core_copies(p: &Player, key: &str) -> u32 {
    p.board
        .iter()
        .chain(p.hand.iter().filter_map(|h| h.item.as_minion()))
        .filter(|m| catalog::is(m, key))
        .map(|m| if m.golden { 3 } else { 1 })
        .sum()
}

fn phantom_hp(board: &[taverngym::state::Minion]) -> i32 {
    board
        .iter()
        .filter(|m| catalog::is(m, PHANTOM))
        .map(|m| m.hp)
        .max()
        .unwrap_or(0)
}

impl Witness {
    pub fn before(game: &Game, seat: usize, action: Action) -> Option<Self> {
        let p = &game.players[seat];
        let economics = apm::economy(&p.board) && apm::has(&p.board, WEAVER);
        let mut core = None;
        let sacrifice = if let Action::Choose(choice) = action
            && let Some(Pending::Target { cont, options }) = p.pending.last()
        {
            let source = match cont {
                Continuation::Activate { eid } | Continuation::Battlecry { eid, .. } => {
                    p.board.iter().find(|m| m.eid == *eid).map(|m| m.card)
                }
                Continuation::HandSpell { hid, .. } => {
                    p.hand.iter().find(|h| h.hid == *hid).map(|h| h.item.card())
                }
                _ => None,
            }
            .map(catalog::key);
            if source == Some(GRAVEROBBER)
                && let Some(m) = options
                    .get(choice)
                    .and_then(|eid| p.board.iter().find(|m| m.eid == *eid))
                && matches!(catalog::key(m.card), WEAVER | PHANTOM)
                && m.has(Kw::REBORN)
            {
                let key = catalog::key(m.card);
                core = Some((key, core_copies(p, key)));
            }
            source
                .and_then(|key| SACRIFICES.iter().position(|&x| x == key))
                .map(|index| (index, phantom_hp(&p.board)))
        } else {
            None
        };
        if (!economics || action == Action::EndTurn) && sacrifice.is_none() {
            return None;
        }
        let bounty = match action {
            Action::Play { hand, .. } => p.hand.get(hand).is_some_and(|h| is_bounty(h.item.card())),
            _ => false,
        };
        let finance = economics
            && match action {
                Action::Buy(i) => p.shop.get(i).is_some_and(|slot| match &slot.item {
                    Item::Minion(m) => undead_apm::finance_minion(p, m) > 0,
                    Item::Spell { card, .. } => undead_apm::finance_spell(p, *card) > 0,
                }),
                _ => false,
            };
        Some(Self {
            economy: economics.then_some((
                p.stats.gold_spent_total,
                p.gold,
                p.stats.tavern_spells_cast,
                p.stats.undead_atk,
            )),
            bounty,
            sacrifice,
            core,
            finance,
        })
    }

    pub fn apply(self, proof: &mut Proof, p: &Player, turn: u32) {
        if let Some((tool, hp)) = self.sacrifice
            && hp > 0
            && let gain @ 1.. = phantom_hp(&p.board) - hp
        {
            proof.sacrifice_growth[tool] += 1;
            proof.permanent_phantom_hp += gain;
            proof.peak_phantom_gain = proof.peak_phantom_gain.max(gain);
            if tool == 0 {
                proof.bellringer_gains += 1;
                proof.phantom_health_added += gain;
            }
        }
        if let Some((key, before)) = self.core {
            proof.core_copies += core_copies(p, key).saturating_sub(before);
        }
        if let Some((spent, gold, spells, attack)) = self.economy {
            if proof.turn != turn {
                proof.turn = turn;
                proof.spend = 0;
                proof.refund = 0;
                proof.bounties = 0;
                proof.spell_growth = 0;
            }
            let spend = (p.stats.gold_spent_total - spent).max(0);
            let casts = (p.stats.tavern_spells_cast - spells).max(0);
            let growth = (p.stats.undead_atk - attack).clamp(0, casts * weaver_attack_per_cast(p));
            proof.spend += spend;
            let inflow = (p.gold - gold + spend).max(0);
            proof.refund += inflow;
            proof.trio_gold_spent += spend;
            proof.trio_gold_inflow += inflow;
            proof.finance_buys += u32::from(self.finance);
            proof.bounties += u32::from(self.bounty && casts > 0);
            proof.spell_growth += growth;
            proof.peak_spend = proof.peak_spend.max(proof.spend);
            proof.peak_spell_growth = proof.peak_spell_growth.max(proof.spell_growth);
        }
    }
}

impl Proof {
    pub fn record_turn(&mut self, p: &Player, turn: u32, tier: u8) {
        let board = &p.last_board;
        let hp = phantom_hp(board);
        if hp > self.max_phantom_hp {
            self.max_phantom_hp = hp;
            if self.permanent_phantom_hp > 0 {
                self.tallest_phantom = Some(Warband::new(p, turn, tier));
            }
        }
        self.final_full = Route::Undead.full(board);
        if !apm::economy(board) || !apm::has(board, WEAVER) {
            return;
        }
        self.trio.get_or_insert(turn);
        let strict = self.turn == turn
            && self.spend >= 15
            && self.refund >= 4
            && self.bounties >= 2
            && self.spell_growth >= 12;
        if strict {
            self.same_turn_loop.get_or_insert(turn);
        }
        if strict && self.final_full {
            self.full_with_loop.get_or_insert(turn);
        }
        if !self.final_full {
            return;
        }
        let premium = board
            .iter()
            .any(|m| catalog::is(m, HANDLESS) && m.golden && m.has(Kw::REBORN));
        if premium {
            self.golden_reborn.get_or_insert(turn);
        }
        let old = self.best.as_ref();
        let old_premium =
            old.is_some_and(|b| b.minions.iter().any(|m| m.card == "BG25_010_G" && m.reborn));
        if old.is_none_or(|b| (premium, p.stats.undead_atk) > (old_premium, b.undead_attack)) {
            self.best = Some(Warband::new(p, turn, tier));
        }
    }
}
