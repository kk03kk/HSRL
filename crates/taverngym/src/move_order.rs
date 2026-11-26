//! Player150 reorder budget and entity-based undo guard. Mirror the live receipt history.

use serde::{Deserialize, Serialize};

use crate::state::Minion;
use crate::types::EntityId;

const MOVE_CAP: u8 = 3;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct RecruitMoveHistory {
    spent: u8,
    orders: Vec<Vec<EntityId>>,
}

impl RecruitMoveHistory {
    pub fn is_empty(&self) -> bool {
        self.spent == 0
    }

    pub fn reset(&mut self) {
        self.spent = 0;
        self.orders.clear();
    }

    pub fn context(&self, board: &[Minion]) -> MoveContext<'_> {
        let current: Vec<_> = board.iter().map(|minion| minion.eid).collect();
        let history = if self.orders.last() == Some(&current) {
            &self.orders[..]
        } else {
            &[]
        };
        MoveContext {
            current,
            history,
            spent: self.spent,
        }
    }

    pub fn record(&mut self, before: Vec<EntityId>, after: Vec<EntityId>) {
        if self.orders.last() != Some(&before) {
            self.orders.clear();
            self.orders.push(before);
        }
        self.orders.push(after);
        self.spent += 1;
    }
}

pub(crate) struct MoveContext<'a> {
    current: Vec<EntityId>,
    history: &'a [Vec<EntityId>],
    spent: u8,
}

impl MoveContext<'_> {
    pub fn original(&self) -> &[EntityId] {
        &self.current
    }

    pub fn moved(&self, from: usize, to: usize) -> Vec<EntityId> {
        let mut proposed = self.current.clone();
        let moved = proposed.remove(from);
        proposed.insert(to, moved);
        proposed
    }

    pub fn allows(&self, from: usize, to: usize) -> bool {
        if self.spent >= MOVE_CAP {
            return false;
        }
        if self.history.is_empty() {
            return true;
        }
        if self.history.contains(&self.moved(from, to)) {
            return false;
        }
        if from.abs_diff(to) != 1 {
            return true;
        }
        let (left, right) = (self.current[from.min(to)], self.current[from.max(to)]);
        // An adjacent move only crosses this pair. Refuse it when any earlier
        // order put the same entities in the opposite relative order.
        !self.history.iter().any(|order| {
            order
                .iter()
                .find(|&&entity| entity == left || entity == right)
                == Some(&right)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::Action;
    use crate::engine::Phase;
    use crate::{Game, RulesConfig};

    fn ready_game(guard: bool) -> Game {
        let mut game = Game::new(
            54,
            RulesConfig {
                player_view_150: true,
                player150_order_guard: guard,
                ..RulesConfig::default()
            },
        );
        game.phase = Phase::Recruit;
        let player = &mut game.players[0];
        player.pending.clear();
        player.ended_turn = false;
        player.board = (100..104)
            .map(|eid| Minion::new(eid, crate::cardid!("BG25_010")))
            .collect();
        game
    }

    fn reorder(from: usize, to: usize) -> Action {
        Action::Move { from, to }
    }

    #[test]
    fn player150_move_guard_rejects_both_representations_of_a_repeated_pair() {
        let mut game = ready_game(true);
        assert!(game.apply(0, reorder(0, 1)).is_ok());
        for action in [reorder(0, 1), reorder(1, 0)] {
            assert!(!game.legal_mask(0)[action.encode()]);
            assert!(game.apply(0, action).is_err());
        }
        assert_eq!(game.players[0].actions_this_turn, 1);
        assert!(game.apply(0, reorder(2, 3)).is_ok());
        assert!(!game.legal_mask(0)[reorder(0, 1).encode()]);
        assert!(game.legal_mask(0)[reorder(1, 2).encode()]);
        assert!(game.apply(0, reorder(1, 2)).is_ok());
        assert!(game.legal_mask(0)[Action::EndTurn.encode()]);
        assert!(!game.legal_mask(0)[reorder(0, 1).encode()]);
    }

    #[test]
    fn player150_move_guard_resets_order_after_roster_changes_and_budget_after_new_turn() {
        let mut game = ready_game(true);
        assert!(game.apply(0, reorder(0, 1)).is_ok());
        let player = &mut game.players[0];
        player
            .board
            .insert(0, Minion::new(104, crate::cardid!("BG25_010")));
        assert!(game.apply(0, reorder(1, 2)).is_ok());
        assert!(game.apply(0, reorder(3, 4)).is_ok());
        assert!(game.apply(0, reorder(0, 1)).is_err());
        game.players[0].recruit_moves.reset();
        assert!(game.apply(0, reorder(0, 1)).is_ok());
        let mut legacy = ready_game(false);
        assert!(legacy.apply(0, reorder(0, 1)).is_ok());
        assert!(legacy.apply(0, reorder(0, 1)).is_ok());
    }
}
