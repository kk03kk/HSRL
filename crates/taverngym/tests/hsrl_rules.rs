mod common;
use common::*;
use taverngym::actions::{A_CANCEL, A_END, A_MOVE};
use taverngym::obs;
use taverngym::state::Continuation;

#[test]
fn research_profile_rejects_move_and_cancel_without_mutating_the_state() {
    let mut g = game(6200549);
    g.cfg = RulesConfig::hsrl();
    set_board(&mut g, 0, &["BGS_039", "BGS_039"]);
    assert!(!g.legal_mask(0)[A_MOVE..].iter().any(|&legal| legal));
    assert!(g.apply(0, Action::Move { from: 0, to: 1 }).is_err());
    let eid = g.players[0].board[0].eid;
    g.players[0].pending.push(Pending::Target {
        options: vec![eid],
        cont: Continuation::HeroPower { opt: 0, slot: 0 },
    });
    assert!(!g.legal_mask(0)[A_CANCEL]);
    assert!(g.legal_mask(0)[Action::Choose(0).encode()]);
    let before = serde_json::to_vec(&g).unwrap();
    assert!(g.apply(0, Action::Cancel).is_err());
    assert_eq!(serde_json::to_vec(&g).unwrap(), before);
}

#[test]
fn research_budget_allows_menus_then_only_end_turn() {
    let mut g = game(1);
    g.cfg = RulesConfig::hsrl();
    g.players[0].uncapped = true;
    g.players[0].actions_this_turn = 99;
    let budget = |g: &Game| obs::observe(g, 0).floats[obs::OBS_FLOATS - 2..].to_vec();
    assert_eq!(budget(&g), [0.99, 0.01]);
    g.apply(0, Action::Freeze).unwrap();
    assert_eq!(budget(&g), [1.0, 0.0]);
    assert!(g.apply(0, Action::Freeze).is_err());
    assert_eq!(g.players[0].actions_this_turn, 100);

    let item = Item::Minion(minion(&mut g, 0, "BGS_039"));
    g.players[0].pending.push(Pending::Discover {
        source: cid("BG36_MagicItem_610"),
        options: vec![item],
    });
    assert!(g.apply(0, Action::EndTurn).is_err());
    assert_eq!(g.legal_actions(0), [Action::Choose(0).encode()]);
    g.apply(0, Action::Choose(0)).unwrap();
    assert_eq!(budget(&g)[1], 0.0);
    assert_eq!(g.legal_actions(0), [A_END]);
    g.apply(0, Action::EndTurn).unwrap();
    assert!(!g.needs_action(0));
}

#[test]
fn historical_6200549_unaffordable_replacement_unblocks_after_discover() {
    let mut g = game(6200549);
    g.cfg = RulesConfig::hsrl();
    g.turn = 9;
    g.players[0].gold = 3;
    let eclipse = cid("BG36_MagicItem_412t2");
    let cards = [
        "BG32_MagicItem_367",
        "BG36_MagicItem_610",
        "BG36_MagicItem_602",
        "BG36_MagicItem_212",
    ];
    let options: Vec<_> = cards.into_iter().map(cid).collect();
    assert!(options.iter().all(|&card| db().get(card).cost > 3));
    g.players[0].trinkets.push(eclipse);
    let item = Item::Minion(minion(&mut g, 0, "BGS_039"));
    g.players[0].pending = vec![
        Pending::Trinket {
            options,
            free: false,
            replace: Some(eclipse),
        },
        Pending::Discover {
            source: cid("BG36_MagicItem_610"),
            options: vec![item],
        },
    ];
    assert!(!g.legal_mask(0)[A_END]);
    g.apply(0, Action::Choose(0)).unwrap();
    assert!(g.players[0].pending.is_empty());
    assert!(g.legal_mask(0)[A_END]);
    assert_eq!(g.players[0].gold, 3);
    assert!(g.players[0].trinkets.contains(&eclipse));
}
