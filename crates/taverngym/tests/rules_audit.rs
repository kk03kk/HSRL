//! 高影响边界回归。伤害上限依据官方31.2公告；不把测试当实机完全一致证明。
use taverngym::actions::Action;
use taverngym::cards::{cid, db};
use taverngym::state::Item;
use taverngym::types::Races;
use taverngym::{Game, RulesConfig};

fn recruit(seed: u64) -> Game {
    let mut g = Game::new(seed, RulesConfig::default());
    for p in 0..8 {
        g.apply(p, Action::Choose(0)).unwrap();
    }
    for p in 0..8 {
        g.players[p].hero = Some(cid("TB_BaconShop_HERO_34"));
        g.players[p].powers.clear();
        g.players[p].pending.clear();
        g.players[p].skipping = false;
    }
    g
}

#[test]
fn published_3701_gift_pair_exclusions_cover_normal_and_golden_only() {
    // Blizzard 37.0.1 hotfix, rolling out October 19, 2026:
    // https://us.forums.blizzard.com/en/hearthstone/t/3701-hotfix-patch/166622
    let mut g = recruit(3701);
    let immune = cid("BG36_MidGameEffect_000t60");
    assert!(g.with_recruit(0, |c| taverngym::content::gift_fits(
        c,
        0,
        immune,
        db().get(cid("BG36_849")),
        1
    )));
    g.cfg.dark_gift_pair_bans = vec![
        ["BG36_849".into(), "BG36_MidGameEffect_000t60".into()],
        ["BG23_318".into(), "BG36_MidGameEffect_000t16".into()],
        ["BG36_360t5".into(), "BG36_MidGameEffect_000t15".into()],
    ];
    for (minion, gift) in [
        ("BG36_849", "BG36_MidGameEffect_000t60"),
        ("BG36_849_G", "BG36_MidGameEffect_000t60"),
        ("BG23_318", "BG36_MidGameEffect_000t16"),
        ("BG23_318_G", "BG36_MidGameEffect_000t16"),
        ("BG36_360t5", "BG36_MidGameEffect_000t15"),
        ("BG36_360_Gt5", "BG36_MidGameEffect_000t15"),
    ] {
        assert!(
            !g.with_recruit(0, |c| taverngym::content::gift_fits(
                c,
                0,
                cid(gift),
                db().get(cid(minion)),
                1
            )),
            "{minion} + {gift}"
        );
    }
    // Ordinary dragons retain the gift; no global nerf to immunity or Divine Shield.
    assert!(g.with_recruit(0, |c| taverngym::content::gift_fits(
        c,
        0,
        immune,
        db().get(cid("BGS_034")),
        1
    )));
}

#[test]
fn damage_cap_boundaries_and_top_four() {
    let cfg = RulesConfig::default();
    for (turn, cap) in [(1, 5), (3, 5), (4, 10), (7, 10), (8, 15), (20, 15)] {
        for alive in 5..=8 {
            assert_eq!(cfg.damage_cap(turn, alive), Some(cap));
        }
        for alive in 1..=4 {
            assert_eq!(cfg.damage_cap(turn, alive), None);
        }
    }
}

#[test]
fn ordinary_buy_play_sell_and_failed_refresh_are_atomic() {
    let mut g = recruit(73001);
    g.players[0].gold = 3;
    let slot = g.players[0]
        .shop
        .iter()
        .position(|s| matches!(s.item, Item::Minion(_)))
        .unwrap();
    g.apply(0, Action::Buy(slot)).unwrap();
    assert_eq!(g.players[0].gold, 0);
    let before = serde_json::to_value(&g).unwrap();
    assert!(g.apply(0, Action::Refresh).is_err());
    assert_eq!(serde_json::to_value(&g).unwrap(), before);
    g.apply(0, Action::Play { hand: 0, pos: 0 }).unwrap();
    g.apply(0, Action::Sell(0)).unwrap();
    assert_eq!(g.players[0].gold, 1);
}

#[test]
fn default_lobbies_never_select_naga() {
    for seed in 73000..73064 {
        let g = Game::new(seed, RulesConfig::default());
        assert!(!g.lobby_races.intersects(Races::NAGA));
        assert!(g.universe.iter().all(|c| db().get(*c).races != Races::NAGA));
    }
}
