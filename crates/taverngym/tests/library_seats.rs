use std::collections::BTreeMap;
use std::sync::Arc;
use taverngym::cards::db;
use taverngym::engine::Phase;
use taverngym::library::{Bank, Trace};
use taverngym::state::Minion;
use taverngym::{Action, Game, RulesConfig};

fn cfg() -> RulesConfig {
    RulesConfig {
        max_moves_per_turn: 8,
        block_move_cycles: true,
        ..Default::default()
    }
}
fn selected(seed: u64) -> Game {
    let mut g = Game::new(seed, cfg());
    for p in 0..8 {
        let a = taverngym::bot::choose_action(&mut g, p);
        g.apply(p, a).unwrap();
    }
    assert_eq!(g.phase, Phase::Recruit);
    g
}
fn bank(g: &Game) -> Bank {
    let mut p = g.players[1].clone();
    p.pending.clear();
    p.shop.clear();
    p.hand.clear();
    p.board = vec![Minion::new(9000, db().id("BG20_100"))];
    p.board[0].atk = 17;
    p.board[0].hp = 23;
    p.board[0].max_hp = 23;
    Bank {
        format: "taverngym.library.v1".into(),
        split: "train".into(),
        rules: g.cfg.clone(),
        card_table: g.card_table,
        traces: vec![Arc::new(Trace {
            id: "source-seat1".into(),
            source_group: "source".into(),
            weight: 1,
            lobby: g.lobby_races.0,
            deity: g.deity,
            paradox: g.paradox,
            frames: BTreeMap::from([(1, p)]),
        })],
    }
}

#[test]
fn move_cycles_are_masked_and_refused_without_consuming_budget() {
    let mut g = selected(104);
    g.players[0].board.clear();
    g.players[0].pending.clear();
    for _ in 0..3 {
        let eid = g.alloc_eid();
        g.players[0]
            .board
            .push(Minion::new(eid, db().id("BG20_100")));
    }
    g.apply(0, Action::Move { from: 0, to: 1 }).unwrap();
    let before = g.players[0].actions_this_turn;
    assert!(!g.legal_mask(0)[Action::Move { from: 1, to: 0 }.encode()]);
    assert!(g.apply(0, Action::Move { from: 1, to: 0 }).is_err());
    assert_eq!(g.players[0].actions_this_turn, before);
    g.players[0].moves_this_turn = 8;
    assert!(
        g.legal_mask(0)[taverngym::actions::A_MOVE..]
            .iter()
            .all(|x| !x)
    );
}

#[test]
fn library_transplant_preserves_life_history_pool_and_unique_entities() {
    let source = selected(105);
    let b = bank(&source);
    b.validate().unwrap();
    let mut g = Game::new(105, cfg());
    g.bind_library(&b, 0).unwrap();
    for p in 0..8 {
        let a = taverngym::bot::choose_action(&mut g, p);
        g.apply(p, a).unwrap();
    }
    assert!(g.known_board(0, 1).0.is_empty());
    g.players[1].health = 11;
    g.players[1].armor = 2;
    let rng = g.rng.clone();
    let pool = taverngym::invariants::pool_balance(&g);
    assert_eq!(pool.0, pool.1);
    g.apply_library_round();
    assert_eq!(g.rng, rng);
    assert_eq!((g.players[1].health, g.players[1].armor), (11, 2));
    assert_eq!(g.players[1].board[0].atk, 17);
    assert_eq!(g.players[1].board[0].pool_copies, 0);
    assert_ne!(g.players[1].board[0].eid, g.players[2].board[0].eid);
    assert!(taverngym::invariants::check(&g).is_empty());
    assert!(taverngym::invariants::pool_species_violations(&g).is_empty());
    let bytes = serde_json::to_string(&g).unwrap();
    let restored: Game = serde_json::from_str(&bytes).unwrap();
    restored.check_loaded().unwrap();
    assert_eq!(serde_json::to_string(&restored).unwrap(), bytes);
    assert!(g.known_board(0, 1).0.is_empty());
}

#[test]
fn library_rejects_context_rule_and_incomplete_trace_mismatch() {
    let src = selected(106);
    let mut b = bank(&src);
    b.rules.buy_cost += 1;
    assert!(Game::new(106, cfg()).bind_library(&b, 0).is_err());
    b.rules = cfg();
    Arc::make_mut(&mut b.traces[0]).lobby = 0;
    assert!(Game::new(106, cfg()).bind_library(&b, 0).is_err());
    let p = b.traces[0].frames[&1].clone();
    Arc::make_mut(&mut b.traces[0]).frames.insert(3, p);
    assert!(b.validate().is_err());
}

#[test]
fn captured_end_of_turn_does_not_change_actions_or_rng() {
    let mut a = selected(107);
    let mut b = a.clone();
    b.cfg.capture_library = true;
    for _ in 0..3000 {
        if a.turn >= 3 {
            break;
        }
        for p in 0..8 {
            if a.needs_action(p) {
                let action = taverngym::bot::choose_action(&mut a, p);
                a.apply(p, action).unwrap();
                b.apply(p, action).unwrap();
            }
        }
    }
    assert!(b.captured_library.is_some());
    assert_eq!(a.rng, b.rng);
    assert_eq!(a.turn, b.turn);
    assert_eq!(
        serde_json::to_value(&a.players).unwrap(),
        serde_json::to_value(&b.players).unwrap()
    );
}
