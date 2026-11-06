//! 当前服务器禁用项、候选池排除和待选安全边界。
use hsbg_engine::{Game, RulesConfig, Action};
use hsbg_engine::cards::{cid, db, CardKind};
use hsbg_engine::state::{Continuation, Pending};

fn recruit() -> Game {
    let mut g = Game::new(1078314, RulesConfig {
        hero_pool: vec!["TB_BaconShop_HERO_34".into()],
        ..RulesConfig::default()
    });
    for p in 0..8 { g.apply(p, Action::Choose(0)).unwrap(); }
    g
}

#[test]
fn november_4_server_bans_are_default_but_historical_override_is_explicit() {
    let g = Game::new(3, RulesConfig::default());
    for id in ["BG36_MagicItem_412", "BG36_MagicItem_412t2"] {
        assert!(!g.trinket_universe.contains(&cid(id)), "禁用饰品仍在候选池: {id}");
    }
    let historical = Game::new(3, RulesConfig { excluded_cards: vec![], ..RulesConfig::default() });
    assert!(historical.trinket_universe.contains(&cid("BG36_MagicItem_412")));
    assert!(historical.trinket_universe.contains(&cid("BG36_MagicItem_412t2")));
}

#[test]
fn exclusions_apply_to_trinket_and_dark_gift_universes() {
    let gift = db().iter().find(|c| c.kind == CardKind::DarkGift && c.window.is_some()
        && hsbg_engine::script::script(c.cid).is_some()).unwrap().id.clone();
    let g = Game::new(3, RulesConfig {
        excluded_cards: vec!["BG30_MagicItem_998".into(), gift.clone()],
        ..RulesConfig::default()
    });
    assert!(!g.trinket_universe.contains(&cid("BG30_MagicItem_998")));
    assert!(!g.gift_universe.contains(&cid(&gift)));
}

#[test]
fn bot_cancels_an_empty_cancellable_target_instead_of_choosing_index_zero() {
    let mut g = recruit();
    g.players[0].pending.push(Pending::Target { options: vec![],
        cont: Continuation::HeroPower { opt: 0, slot: 0 } });
    let action = hsbg_engine::bot::choose_action(&mut g, 0);
    assert!(g.legal_mask(0)[action.encode()], "Bot 推荐非法动作: {action:?}");
    assert_eq!(action.encode(), Action::Cancel.encode());
    g.apply(0, action).unwrap();
    assert!(g.players[0].pending.is_empty());
}

#[test]
fn silent_turn_end_does_not_delete_a_pending_choice() {
    let mut g = recruit();
    g.players[0].pending.push(Pending::Target { options: vec![],
        cont: Continuation::HeroPower { opt: 0, slot: 0 } });
    let before = serde_json::to_string(&g).unwrap();
    g.end_turn_silent(0);
    assert_eq!(serde_json::to_string(&g).unwrap(), before,
               "静默结束不能绕过或清空待选");
}

#[test]
fn discarding_one_linked_card_leaves_a_playable_unlinked_partner() {
    let mut g = recruit();
    g.with_recruit(0, |c| hsbg_engine::recruit::give_linked_pair(c, 0,
        cid("BG28_810"), cid("BGS_039")));
    let hid = g.players[0].hand[1].hid;
    g.with_recruit(0, |c| hsbg_engine::recruit::discard_card(c, 0, hid));
    assert_eq!(g.players[0].hand.len(), 1);
    assert_eq!(g.players[0].hand[0].link, 0);
    g.apply(0, Action::Play { hand: 0, pos: 0 }).unwrap();
    assert_eq!(g.players[0].stats.cards_discarded, 1);
}

#[test]
fn tripling_a_linked_card_unlinks_its_surviving_partner() {
    let mut g = recruit();
    g.with_recruit(0, |c| {
        hsbg_engine::recruit::give_linked_pair(c, 0, cid("BGS_115"), cid("BG28_810"));
        c.give_card(0, cid("BGS_115"));
        c.give_card(0, cid("BGS_115"));
        c.check_triples(0);
    });
    let partner = g.players[0].hand.iter().find(|h| h.item.card() == cid("BG28_810")).unwrap();
    assert_eq!(partner.link, 0);
}

#[test]
fn creating_a_pair_does_not_link_to_a_card_already_consumed_by_an_immediate_triple() {
    let mut g = recruit();
    g.with_recruit(0, |c| {
        c.give_card(0, cid("BGS_115"));
        c.give_card(0, cid("BGS_115"));
        hsbg_engine::recruit::give_linked_pair(c, 0, cid("BG28_810"), cid("BGS_115"));
    });
    let partner = g.players[0].hand.iter().find(|h| h.item.card() == cid("BG28_810")).unwrap();
    assert_eq!(partner.link, 0);
}

#[test]
fn a_full_hand_timewarp_purchase_preserves_existing_protected_queue_links() {
    let mut g = recruit();
    g.with_recruit(0, |c| {
        for _ in 0..9 { c.give_card(0, cid("BG28_810")); }
        c.src=Some(cid("BG36_MagicItem_852"));
        hsbg_engine::recruit::give_linked_pair(c, 0, cid("BG28_810"), cid("BG28_810"));
        c.src=None;
    });
    let mark=g.players[0].hand_queue_marks[0].clone();
    g.with_recruit(0, |c| {
        let item=hsbg_engine::state::Item::Minion(c.new_minion(0,cid("BGS_115")));
        hsbg_engine::content::timewarp::bought(c,0,item);
    });
    assert_eq!(g.players[0].hand_queue.len(),2);
    assert_eq!(g.players[0].hand_queue_marks[0],mark);
    assert_eq!(g.players[0].hand[9].link,mark.hid);
    assert!(hsbg_engine::invariants::check(&g).is_empty());
}
