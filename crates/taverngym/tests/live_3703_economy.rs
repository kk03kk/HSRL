mod common;
use common::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Evidence {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    action: String,
    turn: u32,
    gold_before: i32,
    gold_after: i32,
    tier_before: u8,
    tier_after: u8,
}

#[test]
fn observed_ordinary_recruit_transactions_match_power_log() {
    let evidence: Evidence =
        serde_json::from_str(include_str!("fixtures/live_3703_economy.json")).unwrap();
    for case in evidence.cases {
        let mut g = game(6200549);
        g.cfg = RulesConfig::hsrl();
        if case.action == "upgrade" {
            end_turn_all(&mut g);
            assert_eq!(g.turn, case.turn);
        }
        g.turn = case.turn;
        g.players[0].tier = case.tier_before;
        g.players[0].gold = case.gold_before;
        let action = match case.action.as_str() {
            "buy" => Action::Buy(0),
            "sell" => {
                set_board(&mut g, 0, &["BGS_039"]);
                Action::Sell(0)
            }
            "upgrade" => Action::Upgrade,
            "freeze" => Action::Freeze,
            "refresh" => Action::Refresh,
            _ => panic!("unknown transaction: {}", case.action),
        };
        let hand = g.players[0].hand.len();
        let board = g.players[0].board.len();
        let card = g.players[0].shop.first().map(|slot| slot.item.card());
        assert!(g.legal_mask(0)[action.encode()], "{}", case.action);
        g.apply(0, action).unwrap();
        let p = &g.players[0];
        assert_eq!(p.gold, case.gold_after, "{}", case.action);
        assert_eq!(p.tier, case.tier_after, "{}", case.action);
        match action {
            Action::Buy(_) => {
                assert_eq!(p.hand.len(), hand + 1);
                assert_eq!(p.hand.last().map(|slot| slot.item.card()), card);
            }
            Action::Sell(_) => assert_eq!(p.board.len(), board - 1),
            Action::Freeze => assert!(p.is_frozen()),
            _ => {}
        }
    }
}
