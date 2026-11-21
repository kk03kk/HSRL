//! 37.0.3 live balance and offer regressions.
mod common;
use common::*;

/// Kill player 0's minion at `idx` in a combat against player 1's warband.
fn kill(g: &mut Game, idx: usize) {
    let mut c = combat_ctx(g);
    let e = c.boards[0][idx].eid;
    c.destroy(e);
    c.resolve_deaths();
}

#[test]
fn duplicate_sous_chef_stickers_grant_only_one_extra_hero_power_use() {
    let mut g = game(370305);
    g.players[0].gold = 10;
    g.players[0].set_hero_power(0, cid("TB_BaconShop_HP_010"));
    set_board(&mut g, 0, &["BGS_039"]);
    let sticker = cid("BG35_MagicItem_801");
    for copies in 0..=2 {
        g.players[0].trinkets = vec![sticker; copies];
        let extra = u32::from(copies > 0);
        g.players[0].powers[0].uses_turn = extra;
        assert!(g.can_hero_power(0, 0));
        g.players[0].powers[0].uses_turn = extra + 1;
        assert!(!g.can_hero_power(0, 0));
    }
}

#[test]
fn urzul_sticker_consume_does_not_trigger_consuming_claw() {
    let mut g = game(370306);
    set_board(&mut g, 0, &["BGS_004"]);
    g.players[0].trinkets = vec![cid("BG35_MagicItem_154"), cid("BG36_MagicItem_801")];
    let make_shop = |g: &mut Game| {
        let snack = minion(g, 0, "BGS_034");
        let (atk, hp) = (snack.atk, snack.hp);
        g.players[0].shop = vec![hsbg_engine::state::ShopSlot::new(snack.eid, Item::Minion(snack))];
        (atk, hp)
    };
    let (a0, h0) = stats(&g, 0)[0];
    let (a1, h1) = make_shop(&mut g);
    give(&mut g, 0, "BGS_004");
    play_last(&mut g, 0);
    assert!(g.players[0].shop.is_empty());
    assert_eq!(stats(&g, 0)[0], (a0 + a1, h0 + h1));
    assert!(!g.players[0].board[0].has(Kw::DIVINE_SHIELD));

    let (a2, h2) = make_shop(&mut g);
    let eater = g.players[0].board[0].eid;
    g.with_recruit(0, |ctx| ctx.consume_random(0, eater, 1)).unwrap();
    assert_eq!(stats(&g, 0)[0], (a0 + a1 + a2 + 5, h0 + h1 + h2 + 5));
    assert!(g.players[0].board[0].has(Kw::DIVINE_SHIELD));
}

#[test]
fn champion_and_shark_use_hotfix_rules() {
    for (card, scale) in [("BG23_017", 1), ("BG23_017_G", 2)] {
        let mut g = game(392034);
        set_board(&mut g, 0, &[]);
        give(&mut g, 0, card);
        play_last(&mut g, 0);
        assert_eq!(
            (g.players[0].stats.gem_atk, g.players[0].stats.gem_hp),
            (scale, scale)
        );
        kill(&mut g, 0);
        assert_eq!(
            (g.players[0].stats.gem_atk, g.players[0].stats.gem_hp),
            (2 * scale, 2 * scale)
        );
    }
    for card in ["BG36_206", "BG36_206_G"] {
        assert_eq!(db().get(cid(card)).tier, 5);
    }
}

#[test]
fn market_prices_offers_and_hero_requirements() {
    let mut g = game(39_204);
    g.players[0].trinket_pivots.clear();
    for (id, price) in [
        ("BG30_MagicItem_962", 5),
        ("BG32_MagicItem_808", 3),
        ("BG32_MagicItem_892", 4),
        ("BG36_MagicItem_409", 1),
        ("BG35_MagicItem_306", 3),
        ("BG36_MagicItem_800", 1),
        ("BG30_MagicItem_411", 0),
        ("BG36_MagicItem_307", 1),
        ("BG32_MagicItem_278", 0),
        ("BG36_MagicItem_390", 0),
        ("BG35_MagicItem_150", 0),
        ("BG30_MagicItem_821", 0),
        ("BG36_MagicItem_402", 2),
        ("BG30_MagicItem_442", 7),
        ("BG36_MagicItem_450", 5),
        ("BG35_MagicItem_732", 7),
        ("BG36_MagicItem_308", 6),
        ("BG36_MagicItem_414", 5),
        ("BG36_MagicItem_423t", 0),
        ("BG30_MagicItem_801", 1),
        ("BG32_MagicItem_284", 0),
        ("BG35_MagicItem_740", 0),
        ("BG32_MagicItem_361t", 1),
        ("BG30_MagicItem_914t", 0),
    ] {
        assert_eq!(g.trinket_price(0, cid(id)), price);
    }
    assert!(!g.trinket_universe.contains(&cid("BG36_MagicItem_812")));
    assert!(g.lobby_heroes().any(|h| h.id == "BG36_HERO_002"));
    g.lobby_races = Races::QUILBOAR;
    assert!(!g.lobby_heroes().any(|h| h.id == "BG36_HERO_002"));
    g.trinket_universe = vec![cid("BG36_MagicItem_307")];
    for wants_wand in [false, true] {
        if wants_wand {
            g.lobby_races = Races::BEAST;
        }
        g.players[0].trinkets.clear();
        g.with_recruit(0, |c| {
            hsbg_engine::content::grant_trinket(c, 0, cid("BG35_MagicItem_816"))
        });
        assert_eq!(
            g.players[0].trinkets.contains(&cid("BG36_MagicItem_307")),
            wants_wand
        );
    }
}

#[test]
fn late_dark_gift_restriction_uses_the_board_type_in_every_slot() {
    use hsbg_engine::recruit::late_gift_matches as eligible;
    for (id, race) in [("BG35_342", Races::MECH), ("BG36_102", Races::ABERRATION)] {
        let card = db().get(cid(id));
        assert!(eligible(card, None));
        assert!(eligible(card, Some(race)));
        assert!(!eligible(card, Some(Races::BEAST)));
        assert!(!eligible(card, Some(Races::NONE)));
    }
    for id in ["BG36_356", "BG35_123"] {
        assert!(eligible(db().get(cid(id)), None));
        assert!(!eligible(db().get(cid(id)), Some(Races::MECH)));
    }
}

#[test]
fn an_undead_host_exposes_the_magnetic_drop_but_not_a_conflicting_plain_placement() {
    use hsbg_engine::actions::{A_MAGNETIZE, A_PLAY};
    let mut g = game(370303);
    set_board(&mut g, 0, &["BG25_010"]);
    g.players[0].hand.clear();
    give(&mut g, 0, "BG_DEEP_015");
    let mask = g.legal_mask(0);
    assert!(!mask[A_PLAY]);
    assert!(mask[A_PLAY + 1]);
    assert!(mask[A_MAGNETIZE]);
    assert!(g.play(0, 0, 0).is_err());
    assert_eq!(g.players[0].hand.len(), 1);
    g.magnetize(0, 0, 0).unwrap();
    assert!(g.players[0].hand.is_empty());
    assert_eq!(g.players[0].board.len(), 1);
    assert_eq!(g.players[0].stats.magnetized, 1);
}
