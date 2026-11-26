//! Set the warband order when a minion enters play.
use crate::script::script;
use crate::state::Minion;
use crate::types::{Kw, Races};
use crate::{Action, Game};

use super::{catalog, undead_apm};

fn attack_priority(minion: &Minion) -> i32 {
    if minion.atk <= 0 {
        return -15;
    }
    let hooks = script(minion.card);
    (f64::from(minion.atk).sqrt().min(14.0) as i32)
        + 12 * i32::from(hooks.is_some_and(|s| s.rally.is_some()))
        + 10 * i32::from(minion.has(Kw::CLEAVE))
        + 3 * i32::from(minion.has(Kw::DIVINE_SHIELD))
        - 11 * i32::from(hooks.is_some_and(|s| s.avenge.is_some()))
}

fn order_cost(left: &Minion, right: &Minion) -> i32 {
    if left.is(Races::UNDEAD) && right.is(Races::UNDEAD) {
        let left_phantom = catalog::is(left, undead_apm::PHANTOM);
        let right_phantom = catalog::is(right, undead_apm::PHANTOM);
        let strength = |m: &Minion| (m.max_hp, m.golden, m.atk);
        if left_phantom && (!right_phantom || strength(left) > strength(right)) {
            return 1_000;
        }
    }
    (attack_priority(right) - attack_priority(left)).max(0)
}

fn nearby_value(a: &Minion, b: &Minion) -> i32 {
    let useful = |engine: &Minion, neighbor: &Minion| {
        if catalog::is(engine, catalog::EYE)
            && script(neighbor.card).is_some_and(|s| s.battlecry.is_some())
        {
            18
        } else if catalog::is(engine, catalog::POET)
            && neighbor.is(Races::DRAGON)
            && !catalog::is(neighbor, catalog::POET)
        {
            14
        } else {
            0
        }
    };
    useful(a, b) + useful(b, a)
}

pub(super) fn position(game: &Game, seat: usize, action: Action) -> Action {
    let Action::Play { hand, pos: prior } = action else {
        return action;
    };
    let player = &game.players[seat];
    let Some(minion) = player.hand[hand].item.as_minion() else {
        return action;
    };
    let board = &player.board;
    let mask = game.legal_mask(seat);
    (0..=board.len())
        .filter_map(|pos| {
            let candidate = Action::Play { hand, pos };
            if !mask[candidate.encode()] {
                return None;
            }
            let (left, right) = board.split_at(pos);
            let mut cost = left.iter().map(|m| order_cost(m, minion)).sum::<i32>()
                + right.iter().map(|m| order_cost(minion, m)).sum::<i32>();
            if let Some(last) = left.last() {
                cost -= nearby_value(last, minion);
            }
            if let Some(first) = right.first() {
                cost -= nearby_value(minion, first);
            }
            if let (Some(last), Some(first)) = (left.last(), right.first()) {
                cost += nearby_value(last, first);
            }
            Some((candidate, cost, pos.abs_diff(prior)))
        })
        .min_by_key(|&(_, cost, distance)| (cost, distance))
        .map_or(action, |(candidate, _, _)| candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::db;
    use crate::engine::Phase;
    use crate::{RulesConfig, cardid};

    fn setup(cards: &[&str], new_card: &str) -> Game {
        let mut game = Game::new(77, RulesConfig::hsrl());
        game.phase = Phase::Recruit;
        let p = &mut game.players[0];
        p.pending.clear();
        p.hand.clear();
        p.ended_turn = false;
        p.skipping = false;
        p.board = cards
            .iter()
            .enumerate()
            .map(|(i, id)| Minion::new(300 + i as u32, db().id(id)))
            .collect();
        game.with_recruit(0, |ctx| ctx.give_card(0, db().id(new_card)));
        game
    }

    #[test]
    fn a_new_undead_is_placed_before_the_phantom_and_keeps_it_the_growth_target() {
        let mut game = setup(&[undead_apm::PHANTOM], undead_apm::HANDLESS);
        let action = position(&game, 0, Action::Play { hand: 0, pos: 1 });
        assert_eq!(action, Action::Play { hand: 0, pos: 0 });
        game.apply(0, action).unwrap();
        assert!(catalog::is(&game.players[0].board[1], undead_apm::PHANTOM));
    }

    #[test]
    fn a_new_rally_minion_is_placed_ahead_of_an_ordinary_attacker() {
        let mut game = setup(&["BGS_039"], "BG33_924");
        let action = position(&game, 0, Action::Play { hand: 0, pos: 1 });
        assert_eq!(action, Action::Play { hand: 0, pos: 0 });
        game.apply(0, action).unwrap();
        assert!(game.players[0].board[0].is_card_id(cardid!("BG33_924")));
    }
}
