//! Explicit regression scenarios for the BG changes in the two official October patches.
mod common;
use common::*;
use taverngym::script::{Shown, script};
use taverngym::state::DeityState;

#[test]
fn patch_3701_ship_discover_triggers_hooktusk_without_a_hand_card() {
    for (ship, hook, delta) in [
        ("BG34_Giant_074t", "BG36_344", 1),
        ("BG34_Giant_074t_G", "BG36_344_G", 4),
    ] {
        let mut g = (1..200)
            .map(game)
            .find(|g| g.lobby_races.contains(Races::PIRATE))
            .unwrap();
        set_board(&mut g, 0, &[ship, hook, "BG26_810"]);
        let before = g.players[0].board[2].atk;
        // The Discover still counts even though there is no space in hand.
        for _ in 0..10 {
            give(&mut g, 0, "BG25_008");
        }
        g.with_recruit(0, |c| {
            c.run_board_hook(0, |s| s.start_of_turn, 1, "start of turn")
        });
        while !g.players[0].pending.is_empty() {
            g.apply(0, Action::Choose(0)).unwrap();
        }
        assert_eq!(g.players[0].hand.len(), 10);
        assert_ne!(g.players[0].board[0].counter, 0);
        assert_eq!(g.players[0].board[2].atk, before + delta);
    }
}

#[test]
fn patch_370_mask_golden_deity_summons_with_all_stored_stats() {
    let mut g = game(370);
    g.players[0].deity = Some(DeityState {
        card: cid("BGFYM_000"),
        atk: 41,
        hp: 53,
        golden: false,
        reborn: false,
    });
    // Use the real trinket attachment rather than manually marking it Golden.
    g.players[0].gold = 10;
    g.players[0].pending.push(Pending::Trinket {
        options: vec![cid("BG36_MagicItem_602")],
        free: false,
        replace: None,
    });
    g.apply(0, Action::Choose(0)).unwrap();
    let expected = g.players[0].deity.clone().unwrap();
    assert!(expected.golden);
    set_board(&mut g, 0, &["BG27_002"]);
    let mut c = combat_ctx(&mut g);
    c.cs[0].aberration_deaths = c.g.cfg.deity_threshold - 1;
    let eid = c.boards[0][0].eid;
    c.destroy(eid);
    c.resolve_deaths();
    let m = c.boards[0]
        .iter()
        .find(|m| db().normal_of(m.card) == cid("BGFYM_000"))
        .unwrap();
    assert!(m.golden);
    // No other friendly body remains for C'Thun to distribute stats to.
    assert_eq!(m.atk, expected.atk);
    assert_eq!(m.max_hp, expected.hp);
}

#[test]
fn patch_370_all_hoarding_hyenas_finish_without_truncation() {
    for seed in 0..24 {
        for card in ["BG36_210", "BG36_210_G"] {
            let mut g = game(seed);
            set_board(&mut g, 0, &[card; 7]);
            set_board(&mut g, 1, &[card; 7]);
            let out = fight(&mut g);
            assert!(!out.truncated, "seed {seed}, {card}");
        }
    }
}

#[test]
fn patch_370_drekthar_and_vanndar_summon_once_even_after_space_reopens() {
    for power in ["BG22_HERO_002p", "BG22_HERO_003p"] {
        let mut g = game(371);
        g.turn = 7;
        g.players[0].set_hero_power(0, cid(power));
        set_board(&mut g, 0, &["BG25_008"; 7]);
        let mut c = combat_ctx(&mut g);
        script(cid(power)).unwrap().start_of_combat.unwrap()(
            &mut c,
            taverngym::script::Me {
                side: 0,
                eid: 0,
                card: cid(power),
                golden: false,
                opt: 0,
                pos: 0,
                data: 0,
            },
        );
        assert_eq!(c.boards[0].len(), 7);
        let eid = c.boards[0][0].eid;
        c.destroy(eid);
        c.resolve_deaths();
        assert_eq!(c.boards[0].len(), 7);
        let eid = c.boards[0][0].eid;
        c.destroy(eid);
        c.resolve_deaths();
        c.flush_waiting();
        assert_eq!(c.boards[0].len(), 6);
    }
}

#[test]
fn patch_3701_recruit_reborn_stays_consumed_after_combat() {
    // Both an added keyword and a card with innate Reborn must remain consumed.
    for card in ["BG25_008", "BG25_001"] {
        let mut g = game(372);
        set_board(&mut g, 0, &[card]);
        set_board(&mut g, 1, &["BG25_008"]);
        g.players[0].board[0].give(Kw::REBORN);
        g.with_recruit(0, |c| {
            let e = c.boards[0][0].eid;
            c.destroy(e);
            c.resolve_deaths();
        });
        assert_eq!(g.players[0].board.len(), 1);
        assert!(!g.players[0].board[0].has(Kw::REBORN));
        fight(&mut g);
        assert!(!g.players[0].board[0].has(Kw::REBORN));
    }
}

#[test]
fn patch_3701_courier_shows_five_gold_countdown_and_resets() {
    for card in ["BG26_810", "BG26_810_G"] {
        let mut g = game(373);
        set_board(&mut g, 0, &[card]);
        g.players[0].gold = 10;
        for (spent, left) in [(0, 5), (3, 2), (2, 5)] {
            g.with_recruit(0, |c| {
                if spent > 0 {
                    c.spend_gold(0, spent);
                }
                let m = &c.boards[0][0];
                let sh = Shown {
                    side: 0,
                    card: m.card,
                    golden: m.golden,
                    minion: Some(m),
                    data: 0,
                };
                let shown = script(m.card).unwrap().shown.unwrap()(c, &sh);
                assert_eq!(shown[3], left);
            });
        }
    }
}
