//! Additive, versioned training observations. No hidden pool, RNG, or opponent hands.
//! Owner 0 = player; owner i+1 = base observation card slot i.
use crate::cards::CardId;
use crate::engine::Game;
use crate::state::{Item, Minion, Pending};
use crate::types::PlayerId;

pub const VERSION: u32 = 1;
pub type Token = (i32, u16, Vec<f32>);

fn token(out: &mut Vec<Token>, card: CardId, owner: u16, kind: f32, values: &[f32]) {
    let mut f = vec![0.0; 8];
    f[0] = kind
        + if card.0 & CardId::COPY != 0 {
            16.0
        } else {
            0.0
        };
    f[1..1 + values.len()].copy_from_slice(values);
    // CardId high bits identify runtime copies; embedding indices use the static db index.
    out.push((card.idx() as i32 + 1, owner, f));
}

fn minion(out: &mut Vec<Token>, m: &Minion, owner: u16) {
    token(
        out,
        m.card,
        owner,
        1.0,
        &[
            m.counter as f32,
            m.gift_data as f32,
            m.shield_max as f32,
            m.gems as f32,
            m.temp_atk as f32,
            m.temp_hp as f32,
            m.activations as f32,
        ],
    );
    token(
        out,
        m.card,
        owner,
        8.0,
        &[
            m.gem_atk as f32,
            m.gem_hp as f32,
            m.cycle as f32,
            m.cycle_turn as f32,
            m.first_play_reward_used as u8 as f32,
            m.temp_kw.0 as f32,
            m.bonus_kw.0 as f32,
        ],
    );
    for (i, card) in m.attached.iter().enumerate() {
        token(
            out,
            *card,
            owner,
            2.0,
            &[m.attachment_data.get(i).copied().unwrap_or(0) as f32],
        );
    }
    for card in &m.gained_dr {
        token(out, *card, owner, 3.0, &[]);
    }
}

pub fn observe(g: &Game, p: PlayerId) -> Vec<Token> {
    use crate::actions::{N_BOARD, N_CHOICE, N_HAND, N_POWERS, N_SHOP};
    use crate::obs::{N_QUESTS, N_SECRETS};
    let pl = &g.players[p];
    let mut out = vec![];
    // Numeric player budget has no card source (embedding 0, owner 0).
    let cap = g
        .cfg
        .turn_action_caps
        .get(g.turn.saturating_sub(1) as usize)
        .or_else(|| g.cfg.turn_action_caps.last())
        .copied()
        .unwrap_or(0);
    out.push((
        0,
        0,
        vec![
            0.0,
            pl.actions_this_turn as f32,
            cap as f32,
            pl.freezes_this_turn as f32,
            g.turn as f32,
            pl.tier7 as u8 as f32,
            pl.stats.battlecries_triggered as f32,
            0.0,
        ],
    ));
    if g.cfg.max_moves_per_turn > 0 || g.cfg.block_move_cycles {
        out.push((
            0,
            0,
            vec![
                13.0,
                pl.moves_this_turn as f32,
                g.cfg.max_moves_per_turn as f32,
                pl.move_orders.len() as f32,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ));
    }
    let board = 1 + N_POWERS + 2 + 3 * N_QUESTS + N_SECRETS;
    for (i, m) in pl.board.iter().enumerate() {
        minion(&mut out, m, (board + i + 1) as u16);
    }
    for (i, h) in pl.hand.iter().enumerate() {
        if let Item::Minion(m) = &h.item {
            minion(&mut out, m, (board + N_BOARD + i + 1) as u16);
        }
    }
    for (i, s) in pl.shop.iter().enumerate() {
        if let Item::Minion(m) = &s.item {
            minion(&mut out, m, (board + N_BOARD + N_HAND + i + 1) as u16);
        }
    }
    let choice_owner = board + N_BOARD + N_HAND + N_SHOP + 1;
    let mut choice_item = |i: usize, item: &Item| {
        if let Item::Minion(m) = item {
            minion(&mut out, m, (choice_owner + i) as u16);
        }
    };
    match pl.pending.last() {
        Some(Pending::Discover { options, .. }) => {
            for (i, item) in options.iter().enumerate().take(N_CHOICE) {
                choice_item(i, item);
            }
        }
        Some(Pending::Timewarp { options, .. }) => {
            for (i, (item, _)) in options.iter().enumerate().take(N_CHOICE) {
                choice_item(i, item);
            }
        }
        Some(Pending::Target { options, .. }) => {
            for (i, eid) in options.iter().enumerate().take(N_CHOICE) {
                if let Some(m) = pl.board.iter().find(|m| m.eid == *eid) {
                    choice_item(i, &Item::Minion(m.clone()));
                } else if let Some(s) = pl.shop.iter().find(|s| s.sid == *eid) {
                    choice_item(i, &s.item);
                } else if let Some(h) = pl.hand.iter().find(|h| h.hid == *eid) {
                    choice_item(i, &h.item);
                }
            }
        }
        _ => {}
    }
    for e in &pl.effects {
        token(&mut out, e.card, 0, 4.0, &[e.stacks as f32]);
    }
    for (card, count) in &pl.counters {
        token(&mut out, *card, 0, 5.0, &[*count as f32]);
    }
    for (card, level) in &pl.levels {
        token(&mut out, *card, 0, 6.0, &[*level as f32]);
    }
    for (i, power) in pl.powers.iter().enumerate() {
        token(
            &mut out,
            power.card,
            (i + 2) as u16,
            7.0,
            &[power.counter as f32, power.uses_game as f32],
        );
    }
    let s = &pl.stats;
    let counters = [
        s.spells_cast,
        s.tavern_spells_cast,
        s.battlecries_triggered,
        s.deathrattles_triggered,
        s.elementals_played,
        s.cards_played,
        s.cards_discarded,
        s.golden_played,
        s.minions_bought,
        s.cards_bought,
        s.minions_sold,
        s.gold_spent_total,
        s.gold_spent_turn,
        s.refreshes,
        s.undead_atk,
        s.undead_hp,
        s.beetle_atk,
        s.beetle_hp,
        s.eternal_knight_deaths,
        s.knight_bonus,
        s.elementals_played_turn,
        s.battlecries_turn,
        s.volumizer_atk,
        s.volumizer_hp,
        s.magnetized,
        s.elem_give_atk,
        s.elem_give_hp,
        s.tavern_elemental_atk,
        s.tavern_elemental_hp,
        s.baller_bonus,
        s.lobster_bonus,
        s.spells_cast_turn,
        s.tavern_spells_cast_turn,
        s.cards_played_turn,
        s.minions_played_turn,
        s.spell_discount_next,
        s.combats_won,
        s.combats_lost,
        s.damage_taken_outside_combat,
        s.last_damage,
        s.streak,
        s.ballers_sold,
    ];
    for (i, values) in counters.chunks(7).enumerate() {
        let mut f = vec![0.0; 8];
        f[0] = (32 + i) as f32;
        for (j, n) in values.iter().enumerate() {
            f[j + 1] = *n as f32;
        }
        out.push((0, 0, f));
    }
    for (races, atk, hp) in &s.tavern_race {
        out.push((
            0,
            0,
            vec![
                40.0,
                *races as f32,
                *atk as f32,
                *hp as f32,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ));
    }
    if let Some(card) = s.last_spell_cast {
        token(&mut out, card, 0, 41.0, &[s.last_spell_cast_data as f32]);
    }
    for schedule in &pl.scheduled {
        let owner = pl
            .board
            .iter()
            .position(|m| m.eid == schedule.target)
            .map(|i| board + i + 1)
            .unwrap_or(0) as u16;
        token(
            &mut out,
            schedule.script,
            owner,
            42.0,
            &[
                schedule.turns_left as f32,
                schedule.a as f32,
                schedule.b as f32,
                schedule.if_won as u8 as f32,
            ],
        );
    }
    for item in &pl.hand_queue {
        token(&mut out, item.card(), 0, 43.0, &[]);
        if let Item::Minion(m) = item {
            minion(&mut out, m, 0);
        }
    }
    // Only scouted snapshots, using the same visibility rule as obs::observe.
    let opponent_board = board + N_BOARD + N_HAND + N_SHOP + N_CHOICE + 8 + 7;
    for (j, q) in (0..g.players.len()).filter(|q| *q != p).enumerate() {
        for (i, m) in g.known_board(p, q).0.iter().enumerate().take(N_BOARD) {
            minion(&mut out, m, (opponent_board + j * N_BOARD + i + 1) as u16);
        }
    }
    out
}
