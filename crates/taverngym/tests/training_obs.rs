use taverngym::cards::db;
use taverngym::state::Minion;
use taverngym::types::Races;
use taverngym::{Game, RulesConfig};
use taverngym::{obs, training_obs};

#[test]
fn counters_and_attachment_identity_resolve_base_observation_aliasing() {
    let mut a = Game::new(7, RulesConfig::default());
    let card = a.universe[0];
    let mut m = Minion::new(10000, card);
    m.attached.push(db().id("BG21_015"));
    a.players[0].board.push(m);
    let mut b = a.clone();
    b.players[0].board[0].counter = 34;
    b.players[0].board[0].attached[0] = db().id("BG36_372");
    assert_eq!(obs::observe(&a, 0).floats, obs::observe(&b, 0).floats);
    assert_eq!(obs::observe(&a, 0).cards, obs::observe(&b, 0).cards);
    assert_ne!(training_obs::observe(&a, 0), training_obs::observe(&b, 0));
}

#[test]
fn runtime_copy_counters_have_valid_embeddings_and_distinct_provenance() {
    let mut g = Game::new(9, RulesConfig::default());
    let card = g.universe[0];
    g.players[0].counters = vec![(card, 2), (card.copy(), 3)];
    let tokens = training_obs::observe(&g, 0);
    assert!(tokens.iter().all(|t| t.0 >= 0 && t.0 <= db().len() as i32));
    assert!(tokens.iter().any(|t| t.2[0] == 5.0));
    assert!(tokens.iter().any(|t| t.2[0] == 21.0));
}

#[test]
fn live_hidden_opponent_boards_do_not_enter_training_tokens() {
    let a = Game::new(10, RulesConfig::default());
    let mut b = a.clone();
    let mut m = Minion::new(20000, b.universe[0]);
    m.counter = 12345;
    b.players[1].board.push(m);
    assert_eq!(training_obs::observe(&a, 0), training_obs::observe(&b, 0));
}

#[test]
fn current_default_lobbies_never_restore_naga_or_azshara() {
    for seed in 0..64 {
        let g = Game::new(seed, RulesConfig::default());
        assert!(!g.lobby_races.intersects(Races::NAGA));
        assert!(g.lobby_heroes().all(|h| h.id != "BG22_HERO_007"));
    }
}
