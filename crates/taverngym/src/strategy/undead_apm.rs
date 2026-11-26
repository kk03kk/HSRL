use crate::actions::Action;
use crate::cards::{CardId, db};
use crate::content::{TENTACLE, WATER_DROPLET, is_tavern_spell};
use crate::state::{Item, Minion, Player};
use crate::types::{Kw, Races};
use crate::{Game, cardid};

use super::{
    apm::{self, PRIVATEER, ROGERS, Route},
    cadence, catalog,
    guide_bot::Build,
};

pub const WEAVER: &str = "BG34_692";
pub const HANDLESS: &str = "BG25_010";
pub const PHANTOM: &str = "BG36_515";
pub const BELLRINGER: &str = "BG36_511";
pub const TOMB_TURNING: &str = "BG34_888";
pub const GRAVEROBBER: &str = "BG28_303";
pub const MAW_CASTER: &str = "BG32_340";
pub const BUTCHERING: &str = "BG28_604";
pub const PROSTHETIC_HAND: &str = "BG_DEEP_015";
const BRANN: &str = "BG_LOE_077";
const BUSKER: &str = "BG26_135";
const BLUE_SHELL: &str = "BG24_018";
const PHANTOM_ATTACK_FLOOR: i32 = 48;
const PLAN: Build = Build::Apm(Route::Undead);

fn owned(p: &Player, card: &str) -> bool {
    apm::has(&p.board, card)
        || p.hand
            .iter()
            .any(|h| h.locked == 0 && h.item.as_minion().is_some_and(|m| catalog::is(m, card)))
}

fn offer(game: &Game, seat: usize, key: &str) -> Option<(usize, i32)> {
    game.players[seat]
        .shop
        .iter()
        .enumerate()
        .filter(|(_, s)| s.item.as_minion().is_some_and(|m| catalog::is(m, key)))
        .filter_map(|(i, _)| {
            let (cost, health, _) = game.slot_price(seat, i)?;
            (!health).then_some((i, cost))
        })
        .min_by_key(|&(_, cost)| cost)
}

fn eligible(game: &Game, seat: usize) -> bool {
    game.turn >= 8
        && game.players[seat].tier >= 4
        && game.lobby_races.contains(Route::Undead.required())
}

fn undead_bodies(p: &Player) -> usize {
    p.board.iter().filter(|m| m.is(Races::UNDEAD)).count()
}

pub fn select(game: &Game, seat: usize, usual: Build) -> Build {
    if usual == PLAN || !eligible(game, seat) {
        return usual;
    }
    let p = &game.players[seat];
    let life = p.health + p.armor;
    if apm::economy(&p.board)
        && apm::has(&p.board, WEAVER)
        && p.stats.undead_atk >= 12
        && (undead_bodies(p) >= 2 || owned(p, HANDLESS) || owned(p, PHANTOM))
    {
        return PLAN;
    }
    let rogers = owned(p, ROGERS);
    let weaver = owned(p, WEAVER);
    let privateer = owned(p, PRIVATEER);
    let missing_price = |key, has| {
        if has {
            Some(0)
        } else {
            offer(game, seat, key).map(|(_, cost)| cost)
        }
    };
    let funded = missing_price(ROGERS, rogers)
        .zip(missing_price(WEAVER, weaver))
        .is_some_and(|(economy, payoff)| economy + payoff <= p.gold);
    let established_other_apm = matches!(usual, Build::Apm(route) if route.payoff(&p.board))
        || usual == Build::Catalog(catalog::Route::PirateGold);
    let pressure = life <= 16 || life <= 22 && p.stats.last_damage <= -8;
    if funded
        && (rogers || weaver && privateer)
        && (privateer || undead_bodies(p) >= 3)
        && (!pressure || weaver && undead_bodies(p) >= 3)
        && (!established_other_apm || weaver && undead_bodies(p) >= 3)
    {
        PLAN
    } else {
        usual
    }
}

pub fn buy_core(game: &Game, seat: usize, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
    let p = &game.players[seat];
    if p.hand.len() + 1 >= game.cfg.hand_size
        || game.action_cap_of(seat).saturating_sub(p.actions_this_turn) < 4
    {
        return None;
    }
    let life = p.health + p.armor;
    let invested = owned(p, ROGERS) || owned(p, PRIVATEER) || owned(p, WEAVER);
    let finish = apm::economy(&p.board)
        && apm::has(&p.board, WEAVER)
        && (p.stats.undead_atk >= 12 || life >= 22);
    // The Weaver should see the banked spells. Rogers should see their purchase cost.
    [WEAVER, ROGERS, PRIVATEER, HANDLESS, PHANTOM]
        .into_iter()
        .find_map(|key| {
            if matches!(key, HANDLESS | PHANTOM) && !finish {
                return None;
            }
            if owned(p, key) || !(invested || life >= 23 && p.board.len() >= 5) {
                return None;
            }
            if life <= 16 && key != WEAVER && !owned(p, WEAVER) {
                return None;
            }
            let (index, _) = offer(game, seat, key)?;
            let buy = Action::Buy(index);
            legal(buy).then_some(buy)
        })
}

pub fn discover_bonus(game: &Game, seat: usize, build: Build, item: &Item) -> f64 {
    if !eligible(game, seat) {
        return 0.0;
    }
    let p = &game.players[seat];
    let premium_handless_gift = item.as_minion().is_some_and(|m| {
        catalog::is(m, HANDLESS)
            && m.has(Kw::REBORN)
            && m.attached.contains(&cardid!("BG36_MidGameEffect_000t12"))
    });
    let rogers = owned(p, ROGERS);
    let weaver = owned(p, WEAVER);
    let privateer = owned(p, PRIVATEER);
    let Some(m) = item.as_minion() else {
        return 0.0;
    };
    let life = p.health + p.armor;
    let undead = undead_bodies(p);
    match catalog::key(m.card) {
        ROGERS if !rogers && (weaver || privateer || build == PLAN && life >= 19) => 37.0,
        WEAVER if !weaver && (rogers || undead >= 3) => 34.0,
        PRIVATEER if !privateer && (rogers || weaver && build == PLAN) => 27.0,
        PHANTOM if build == PLAN && weaver && !owned(p, PHANTOM) => 18.0,
        HANDLESS if build == PLAN && weaver && premium_handless_gift => 27.0,
        HANDLESS if build == PLAN && weaver && !owned(p, HANDLESS) => 16.0,
        _ => 0.0,
    }
}

pub fn wants_golden_handless(p: &Player) -> bool {
    p.board
        .iter()
        .any(|m| catalog::is(m, HANDLESS) && !m.golden)
        && !p.board.iter().any(|m| catalog::is(m, HANDLESS) && m.golden)
}

pub fn growth_spell_bonus(p: &Player, card: CardId) -> f64 {
    if catalog::key(card) == TOMB_TURNING && needs_undead_discover(p) {
        return 22.0;
    }
    if !apm::has(&p.board, WEAVER) || !is_tavern_spell(card) {
        return 0.0;
    }
    match catalog::key(card) {
        "BG28_849" if finishers_collected(p) => 13.0,
        "BG28_849" => -12.0, // Leave minions in the search pool until the finishers are found.
        "BG28_810" | "BG36_303" | "BG28_827" => 9.0,
        _ if db().get(card).cost <= 2 => 5.0,
        _ => 0.0,
    }
}

pub fn finishers_collected(p: &Player) -> bool {
    owned(p, HANDLESS) && owned(p, PHANTOM)
}

pub fn needs_undead_discover(p: &Player) -> bool {
    p.tier >= 6
        && (owned(p, ROGERS) || owned(p, WEAVER))
        && (!owned(p, WEAVER) || !finishers_collected(p))
}

pub fn can_play_minion(m: &Minion, turn: u32) -> bool {
    m.doomed_turn == 0
        || m.doomed_turn != turn
        || !matches!(catalog::key(m.card), WEAVER | PHANTOM | HANDLESS)
}

pub fn bellringer_target(p: &Player, m: &Minion) -> bool {
    // Keep the engine and built bodies. A cheap Undead comes back with the global Attack aura.
    let key = catalog::key(m.card);
    m.is(Races::UNDEAD)
        && !m.golden
        && !matches!(key, WEAVER | PHANTOM | HANDLESS | BELLRINGER)
        && m.hp <= 8
        && m.atk <= db().get(m.card).atk + p.stats.undead_atk + 8
}

pub fn prepare_bellringer(p: &Player, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
    if p.stats.undead_atk < 12 || !p.board.iter().any(|m| bellringer_target(p, m)) {
        return None;
    }
    let (index, _) = p
        .board
        .iter()
        .enumerate()
        .find(|(i, m)| catalog::is(m, BELLRINGER) && legal(Action::Activate(*i)))?;
    let (phantom, _) = p
        .board
        .iter()
        .enumerate()
        .find(|(_, m)| catalog::is(m, PHANTOM))?;
    let right = p.board.iter().rposition(|m| m.is(Races::UNDEAD))?;
    let position = Action::Move {
        from: phantom,
        to: right,
    };
    if phantom != right && legal(position) {
        Some(position)
    } else if phantom == right {
        Some(Action::Activate(index))
    } else {
        None
    }
}

pub fn keep_bonus(p: &Player, m: &Minion) -> f64 {
    let active = apm::has(&p.board, WEAVER);
    if !active {
        return 0.0;
    }
    if catalog::is(m, WEAVER) && p.board.iter().filter(|b| catalog::is(b, WEAVER)).count() < 3 {
        return 9.0;
    }
    if catalog::is(m, BELLRINGER) && apm::has(&p.board, PHANTOM) {
        return 8.0;
    }
    if m.is(Races::UNDEAD) {
        let keywords = if m.has(Kw::REBORN) { 8.0 } else { 0.0 }
            + if m.has(Kw::DIVINE_SHIELD) { 9.0 } else { 0.0 };
        let resummons = if m.is_card_id(cardid!("BG25_009")) {
            9.0
        } else {
            0.0
        };
        return keywords + resummons + cadence::minion_strength(m).min(25.0) * 0.08;
    }
    0.0
}

pub fn finance_minion(p: &Player, m: &Minion) -> i32 {
    let repeats = p
        .board
        .iter()
        .filter(|m| catalog::is(m, BRANN))
        .map(|m| if m.golden { 3 } else { 2 })
        .max()
        .unwrap_or(1);
    let times = if m.golden { 2 } else { 1 };
    match catalog::key(m.card) {
        "BG36_098" | "BGS_115" => 15 + 5 * times,
        "BG24_715" => 17 + 3 * repeats + 3 * (times - 1),
        BLUE_SHELL if p.stats.lost_last => 36,
        BUSKER => 10 + 5 * repeats * times,
        "BG36_308" | "BG22_202" => 16 + 3 * times,
        "BG36_100" => 16 + 4 * times,
        "BG28_550" | "BG33_821" => 14 + 11 * repeats * times,
        "BG36_343" => 16 + 9 * repeats * times,
        "BG34_170" | "BGS_123" | "BG36_520" => 12 + 5 * repeats * times,
        _ => 0,
    }
}

pub fn finance_spell(p: &Player, card: CardId) -> i32 {
    match catalog::key(card) {
        "BG28_810" | "BG36_303" => 29,
        "BG28_827" => 23,
        "BG28_800" | "BG28_884" if p.health + p.armor >= 16 => 17,
        "BG28_504" => 16,
        "BG28_GIL_836" => 25,
        "BG31_890" | "BG34_330" if p.tier >= 5 => 25,
        "BG31_892" | "BG28_882" => 21,
        _ => 0,
    }
}

pub fn is_sacrifice(card: &str) -> bool {
    matches!(card, BELLRINGER | GRAVEROBBER | MAW_CASTER | BUTCHERING)
}

fn summon_count(m: &Minion) -> usize {
    let twice = if m.golden { 2 } else { 1 };
    match catalog::key(m.card) {
        HANDLESS | "BG29_611" | "BG31_803" | "BG25_009" => twice,
        "BG28_300" => 2 * twice,
        "BG30_125" => 3 * twice,
        _ => 0,
    }
}

fn perfect_reborn(m: &Minion) -> bool {
    m.has_attached("BG36_MidGameEffect_000t12")
}

fn has_prosthetic_hand(m: &Minion) -> bool {
    m.attached
        .iter()
        .any(|&card| catalog::key(card) == PROSTHETIC_HAND)
}

fn holding(p: &Player, card: &str) -> bool {
    p.hand
        .iter()
        .any(|h| h.locked == 0 && catalog::key(h.item.card()) == card)
}

pub fn phantom_growth_ready(p: &Player) -> bool {
    p.stats.undead_atk >= PHANTOM_ATTACK_FLOOR && apm::has(&p.board, PHANTOM)
}

fn reborn_fodder(m: &Minion) -> bool {
    m.is(Races::UNDEAD)
        && m.has(Kw::REBORN)
        && !m.golden
        && !matches!(
            catalog::key(m.card),
            WEAVER | PHANTOM | HANDLESS | BELLRINGER
        )
        && summon_count(m) == 0
        && m.max_hp <= m.def().hp + m.pbonus_hp + 12
}

pub fn reborn_fodder_value(p: &Player, m: &Minion) -> i32 {
    if !phantom_growth_ready(p) || !reborn_fodder(m) {
        return 0;
    }
    let on_board = p.board.iter().any(|b| reborn_fodder(b) && b.eid != m.eid);
    let in_hand = p.hand.iter().any(|h| {
        h.locked == 0
            && h.item
                .as_minion()
                .is_some_and(|b| reborn_fodder(b) && b.eid != m.eid)
    });
    if on_board || in_hand { 0 } else { 34 }
}

pub fn reborn_shop_ready(p: &Player) -> bool {
    !phantom_growth_ready(p)
        || apm::has(&p.board, BELLRINGER)
            && p.board
                .iter()
                .any(|m| reborn_target_score(p, BELLRINGER, m, 1).is_some())
        || p.board.iter().any(reborn_fodder)
}

pub fn extra_phantom_value(p: &Player, m: &Minion) -> i32 {
    if !catalog::is(m, PHANTOM)
        || p.stats.undead_atk < 100
        || !reborn_shop_ready(p)
        || !apm::has(&p.board, PHANTOM)
    {
        return 0;
    }
    let copies = p
        .board
        .iter()
        .chain(
            p.hand
                .iter()
                .filter_map(|h| (h.locked == 0).then(|| h.item.as_minion()).flatten()),
        )
        .filter(|b| catalog::is(b, PHANTOM) && b.eid != m.eid)
        .count();
    if copies == 1 { 42 } else { 0 }
}

pub fn reborn_target_score(p: &Player, card: &str, m: &Minion, slots: usize) -> Option<i32> {
    if !phantom_growth_ready(p) {
        return sacrifice_score(p, card, m, slots);
    }
    // Deathrattles summon first. A Reborn keyword without a surviving body gives no Phantom buff.
    if !m.is(Races::UNDEAD)
        || slots <= summon_count(m)
        || (card != BELLRINGER && !m.has(Kw::REBORN))
    {
        return None;
    }
    let key = catalog::key(m.card);
    let spare_weaver = key == WEAVER
        && !m.golden
        && p.board.iter().any(|b| {
            catalog::is(b, WEAVER) && b.eid != m.eid && (b.golden, b.max_hp) >= (m.golden, m.max_hp)
        });
    let copying = card == GRAVEROBBER && copy_core_target(p, m);
    let local_health = (m.max_hp - m.def().hp - m.pbonus_hp).max(0);
    let local_attack = (m.atk - m.def().atk - m.pbonus_atk).max(0);
    if !copying
        && (!spare_weaver && matches!(key, WEAVER | PHANTOM | HANDLESS | BELLRINGER)
            || m.golden
            || local_health > 12
            || local_attack > 12 && !has_prosthetic_hand(m))
    {
        return None;
    }
    let attack = if perfect_reborn(m) {
        m.atk
    } else {
        m.def().atk + m.pbonus_atk
    };
    // Keep existing Reborn charges for spells/battlecries whenever an equivalent Bellringer target exists.
    let renew = if card == BELLRINGER && !m.has(Kw::REBORN) {
        35
    } else {
        0
    };
    let copy = if copying {
        if key == WEAVER { 350 } else { 250 }
    } else {
        0
    };
    Some(
        attack + renew + copy - local_health - m.max_hp.min(20) - if spare_weaver { 35 } else { 0 },
    )
}

pub fn reborn_purchase(p: &Player, card: &str, board_size: usize) -> i32 {
    if !phantom_growth_ready(p) || !is_sacrifice(card) {
        return sacrifice_purchase(p, card, board_size);
    }
    if holding(p, card) || card == BELLRINGER && apm::has(&p.board, card) {
        return 0;
    }
    let slots = board_size.saturating_sub(p.board.len() + usize::from(card != BUTCHERING)) + 1;
    let has_target = p
        .board
        .iter()
        .any(|m| reborn_target_score(p, card, m, slots).is_some());
    let has_fodder = p
        .hand
        .iter()
        .any(|h| h.locked == 0 && h.item.as_minion().is_some_and(reborn_fodder))
        || p.shop
            .iter()
            .any(|s| s.item.as_minion().is_some_and(reborn_fodder));
    if has_target || has_fodder {
        if card == BELLRINGER {
            55
        } else if has_target {
            52
        } else {
            25
        }
    } else {
        0
    }
}

pub fn reborn_magnetic_priority(p: &Player, m: &Minion) -> i32 {
    if !phantom_growth_ready(p)
        || m.has(Kw::REBORN)
        || ![BUTCHERING, GRAVEROBBER, MAW_CASTER]
            .into_iter()
            .any(|key| holding(p, key))
        || p.board.iter().any(reborn_fodder)
    {
        return 0;
    }
    if reborn_target_score(p, BELLRINGER, m, 1).is_some() {
        140
    } else {
        0
    }
}

pub fn prepare_reborn_sacrifice(
    game: &Game,
    seat: usize,
    legal: &dyn Fn(Action) -> bool,
    ending_recruit: bool,
) -> Option<Action> {
    let p = &game.players[seat];
    let remaining = game.action_cap_of(seat).saturating_sub(p.actions_this_turn);
    if !phantom_growth_ready(p) || remaining == 0 {
        return None;
    }
    // Let spend-and-cast loops grow the aura first, while reserving room for moves and plays.
    if !ending_recruit
        && remaining > 9
        && p.hand.iter().any(|h| {
            h.locked == 0
                && is_tavern_spell(h.item.card())
                && !matches!(catalog::key(h.item.card()), BUTCHERING | "BG28_849")
        })
    {
        return None;
    }
    let phantom = p
        .board
        .iter()
        .enumerate()
        .filter(|(_, m)| catalog::is(m, PHANTOM))
        .max_by_key(|(_, m)| (m.max_hp, m.golden, m.atk))
        .map(|(i, _)| i)?;
    let right = p.board.iter().rposition(|m| m.is(Races::UNDEAD))?;
    let free = game.cfg.board_size.saturating_sub(p.board.len());
    let best = |card, slots| {
        p.board
            .iter()
            .filter_map(|m| reborn_target_score(p, card, m, slots))
            .max()
    };
    let hand_tool = p
        .hand
        .iter()
        .enumerate()
        .filter_map(|(hand, h)| {
            let card = catalog::key(h.item.card());
            let minion = card != BUTCHERING;
            let play = Action::Play { hand, pos: 0 };
            if h.locked != 0
                || !matches!(card, BUTCHERING | MAW_CASTER | GRAVEROBBER)
                || minion && free == 0
                || !legal(play)
            {
                return None;
            }
            best(card, free + usize::from(!minion)).map(|score| (play, score))
        })
        .max_by_key(|(_, score)| *score)
        .map(|(play, _)| play);
    let bell = p.board.iter().enumerate().find_map(|(i, m)| {
        let action = Action::Activate(i);
        ((ending_recruit || remaining <= 9 || p.gold <= 1)
            && catalog::is(m, BELLRINGER)
            && legal(action)
            && best(BELLRINGER, free + 1).is_some())
        .then_some(action)
    });
    let action = hand_tool.or(bell);
    if action.is_none() && free == 0 && remaining >= 2 {
        let waiting_targeted = [MAW_CASTER, GRAVEROBBER]
            .into_iter()
            .any(|card| holding(p, card) && best(card, 1).is_some());
        let waiting_bell = holding(p, BELLRINGER) && best(BELLRINGER, 1).is_some();
        let waiting_fodder = p.hand.iter().any(|h| {
            h.locked == 0
                && h.item
                    .as_minion()
                    .is_some_and(|m| reborn_fodder_value(p, m) > 0)
        }) && ([BUTCHERING, MAW_CASTER, GRAVEROBBER, BELLRINGER]
            .into_iter()
            .any(|c| holding(p, c))
            || apm::has(&p.board, BELLRINGER));
        if waiting_targeted || waiting_bell || waiting_fodder {
            return p
                .board
                .iter()
                .enumerate()
                .filter_map(|(i, m)| {
                    let action = Action::Sell(i);
                    let key = catalog::key(m.card);
                    (legal(action)
                        && !matches!(
                            key,
                            ROGERS | PRIVATEER | WEAVER | PHANTOM | HANDLESS | BELLRINGER
                        )
                        && !reborn_fodder(m)
                        && m.max_hp <= m.def().hp + m.pbonus_hp + 18
                        && (key != BRANN || p.stats.undead_atk >= 150)
                        && (waiting_fodder
                            || [MAW_CASTER, GRAVEROBBER, BELLRINGER].into_iter().any(|c| {
                                holding(p, c)
                                    && p.board.iter().any(|b| {
                                        b.eid != m.eid && reborn_target_score(p, c, b, 1).is_some()
                                    })
                            })))
                    .then_some((action, cadence::minion_strength(m)))
                })
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(action, _)| action);
        }
    }
    let action = action?;
    if phantom == right {
        return Some(action);
    }
    let position = Action::Move {
        from: phantom,
        to: right,
    };
    (remaining >= 2 && legal(position)).then_some(position)
}

pub fn copy_core_target(p: &Player, m: &Minion) -> bool {
    let card = catalog::key(m.card);
    let base = m.def();
    matches!(card, WEAVER | PHANTOM)
        && m.has(Kw::REBORN)
        && (p.stats.undead_atk >= 24 || card == WEAVER && apm::economy(&p.board))
        && (perfect_reborn(m)
            || m.max_hp <= base.hp + m.pbonus_hp + 12
                && (has_prosthetic_hand(m) || m.atk <= base.atk + m.pbonus_atk + 12))
        && p.board.iter().filter(|b| catalog::is(b, card)).count()
            + p.hand
                .iter()
                .filter(|h| catalog::key(h.item.card()) == card)
                .count()
            < 3
}

pub fn sacrifice_score(p: &Player, card: &str, m: &Minion, slots: usize) -> Option<i32> {
    let key = catalog::key(m.card);
    if !m.is(Races::UNDEAD) || key == card && card == BELLRINGER {
        return None;
    }
    if card == GRAVEROBBER && copy_core_target(p, m) {
        return Some(if key == WEAVER { 530 } else { 495 });
    }
    // A minion's Deathrattle summons before Reborn and can take its space.
    let summons = summon_count(m);
    let reborn = (card == BELLRINGER || m.has(Kw::REBORN)) && slots > summons;
    let base = m.def();
    let extra_attack = (m.atk - base.atk - m.pbonus_atk).max(0);
    let extra_health = (m.max_hp - base.hp - m.pbonus_hp).max(0);
    if matches!(
        key,
        WEAVER | PHANTOM | HANDLESS | BELLRINGER | ROGERS | PRIVATEER | BRANN
    ) || m.golden
        || extra_attack > 12
        || extra_health > 12
        || !reborn && summons == 0
    {
        return None;
    }
    if reborn && phantom_growth_ready(p) {
        // A plain Reborn keeps global auras. It loses its local enchantments and Reborn keyword.
        let attack = if perfect_reborn(m) {
            m.atk
        } else {
            base.atk + m.pbonus_atk
        };
        return Some(
            100 + attack.min(350)
                + 5 * slots.min(summons + 1) as i32
                + if card == BELLRINGER && !m.has(Kw::REBORN) {
                    12
                } else {
                    0
                }
                - extra_attack
                - extra_health
                - m.max_hp.min(16),
        );
    }
    if card == BELLRINGER {
        return None;
    }
    Some(
        24 + 4 * slots.min(summons) as i32 + if reborn { 15 } else { 0 }
            - extra_attack
            - extra_health
            - m.max_hp.min(15),
    )
}

pub fn best_sacrifice(p: &Player, card: &str, slots: usize) -> Option<i32> {
    p.board
        .iter()
        .filter_map(|m| sacrifice_score(p, card, m, slots))
        .max()
}

pub fn sacrifice_purchase(p: &Player, card: &str, board_size: usize) -> i32 {
    let uses_slot = card != BUTCHERING;
    let slots = board_size.saturating_sub(p.board.len() + usize::from(uses_slot)) + 1;
    let value = best_sacrifice(p, card, slots).unwrap_or(0);
    match card {
        GRAVEROBBER if value >= 495 => 54,
        BELLRINGER if value >= 100 && !owned(p, BELLRINGER) => 39,
        MAW_CASTER if value >= 100 => 41,
        GRAVEROBBER if value >= 100 => 33,
        BUTCHERING if value >= 100 => 46,
        MAW_CASTER | BUTCHERING if value > 0 => 21,
        PROSTHETIC_HAND if prosthetic_value(p) > 0 => 33,
        _ => 0,
    }
}

pub fn fodder_value(p: &Player, m: &Minion) -> i32 {
    if !phantom_growth_ready(p) || !m.is(Races::UNDEAD) {
        return 0;
    }
    let held_tool = p
        .hand
        .iter()
        .any(|h| h.locked == 0 && is_sacrifice(catalog::key(h.item.card())));
    let has_bell = apm::has(&p.board, BELLRINGER);
    match catalog::key(m.card) {
        "BG25_001" | "BG25_022" if held_tool || has_bell => 21,
        "BG28_300" | "BG30_125" if held_tool || has_bell => 17,
        _ => 0,
    }
}

pub fn prosthetic_target(p: &Player, m: &Minion) -> i32 {
    if !m.is(Races::UNDEAD) || m.has(Kw::REBORN) {
        return 0;
    }
    let robber = owned(p, GRAVEROBBER)
        || p.shop
            .iter()
            .any(|s| catalog::key(s.item.card()) == GRAVEROBBER);
    match catalog::key(m.card) {
        WEAVER if robber => 155,
        PHANTOM
            if robber
                && (!phantom_growth_ready(p) || m.max_hp <= m.def().hp + m.pbonus_hp + 12) =>
        {
            145
        }
        HANDLESS => 120,
        WEAVER if apm::economy(&p.board) => 78,
        PHANTOM => 65,
        _ if phantom_growth_ready(p) && m.max_hp <= 8 => 35,
        _ => 0,
    }
}

fn prosthetic_value(p: &Player) -> i32 {
    p.board
        .iter()
        .map(|m| prosthetic_target(p, m))
        .max()
        .unwrap_or(0)
}

pub fn cycle_sale(game: &Game, seat: usize, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
    let p = &game.players[seat];
    if !apm::economy(&p.board)
        || !apm::has(&p.board, WEAVER)
        || p.hand.len() + 3 >= game.cfg.hand_size
        || p.actions_this_turn + 7 >= game.action_cap_of(seat)
        || p.board.len() <= 4
    {
        return None;
    }
    p.board
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            let action = Action::Sell(i);
            let key = catalog::key(m.card);
            if !legal(action)
                || m.max_hp > m.def().hp + m.pbonus_hp + 12
                || m.atk > m.def().atk + m.pbonus_atk + 15
            {
                return None;
            }
            let value = match key {
                BLUE_SHELL if p.stats.lost_last => 38,
                "BG36_098" | "BGS_115" | "BG24_715" | "BG36_100" | "BG36_308" | "BG22_202" => 29,
                TENTACLE | WATER_DROPLET | BUSKER => 15,
                GRAVEROBBER | MAW_CASTER if phantom_growth_ready(p) => 12,
                _ if finance_minion(p, m) > 0 => 11,
                _ => return None,
            };
            // Keep a playable replacement if the cycle would otherwise leave a short warband.
            (p.board.len() > 5
                || p.gold < 3
                    && (value >= 29
                        || matches!(key, TENTACLE | WATER_DROPLET) && p.health + p.armor >= 16)
                || p.hand
                    .iter()
                    .any(|h| h.locked == 0 && h.item.as_minion().is_some()))
            .then_some((action, value))
        })
        .max_by_key(|(_, value)| *value)
        .map(|(action, _)| action)
}

pub fn prepare_sacrifice(
    game: &Game,
    seat: usize,
    legal: &dyn Fn(Action) -> bool,
) -> Option<Action> {
    let p = &game.players[seat];
    if !phantom_growth_ready(p) || p.actions_this_turn + 3 >= game.action_cap_of(seat) {
        return None;
    }
    let phantom = p
        .board
        .iter()
        .enumerate()
        .filter(|(_, m)| catalog::is(m, PHANTOM))
        .max_by_key(|(_, m)| (m.max_hp, m.atk))
        .map(|(i, _)| i)?;
    let right = p.board.iter().rposition(|m| m.is(Races::UNDEAD))?;
    let slots = game.cfg.board_size.saturating_sub(p.board.len()) + 1;
    let spells_to_cast = p.hand.iter().any(|h| {
        h.locked == 0
            && is_tavern_spell(h.item.card())
            && !matches!(catalog::key(h.item.card()), BUTCHERING | "BG28_849")
    });
    let bell = p.board.iter().enumerate().find_map(|(i, m)| {
        let action = Action::Activate(i);
        (catalog::is(m, BELLRINGER)
            && legal(action)
            && (!spells_to_cast || p.actions_this_turn + 7 >= game.action_cap_of(seat))
            && best_sacrifice(p, BELLRINGER, slots).is_some_and(|v| v >= 100))
        .then_some(action)
    });
    if phantom == right {
        return bell;
    }
    let playable = p.hand.iter().any(|h| {
        let key = catalog::key(h.item.card());
        h.locked == 0
            && matches!(key, BUTCHERING | MAW_CASTER | GRAVEROBBER)
            && (key == BUTCHERING || slots > 1)
            && best_sacrifice(p, key, slots - usize::from(key != BUTCHERING))
                .is_some_and(|v| v >= 100)
    });
    let action = Action::Move {
        from: phantom,
        to: right,
    };
    (legal(action) && (bell.is_some() || playable)).then_some(action)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RulesConfig;
    use crate::engine::Phase;
    use crate::state::{Pending, ShopSlot};
    use crate::strategy::guide_bot::Plan as _;

    fn setup(cards: &[&str]) -> Game {
        let mut game = Game::new(99, RulesConfig::default());
        game.lobby_races = crate::strategy::pivot::LOBBY_MASK;
        game.phase = Phase::Recruit;
        game.turn = 10;
        let p = &mut game.players[0];
        p.pending.clear();
        p.tier = 6;
        p.health = 30;
        p.gold = 3;
        p.shop.clear();
        p.board = cards
            .iter()
            .enumerate()
            .map(|(i, key)| Minion::new(i as u32 + 300, db().id(key)))
            .collect();
        game
    }

    #[test]
    fn knights_can_take_an_affordable_weaver_when_the_economy_is_ready() {
        let mut game = setup(&[ROGERS, PRIVATEER, "BG25_008", HANDLESS]);
        game.players[0].shop.push(ShopSlot::new(
            400,
            Item::Minion(Minion::new(400, db().id(WEAVER))),
        ));
        let knight = Build::Catalog(catalog::Route::Knight);
        let legal = |action: Action| game.legal_mask(0)[action.encode()];
        assert_eq!(select(&game, 0, knight), PLAN);
        assert_eq!(buy_core(&game, 0, &legal), Some(Action::Buy(0)));
        game.players[0].gold = 2;
        assert_eq!(select(&game, 0, knight), knight);
    }

    #[test]
    fn repeated_bounty_casts_feed_both_gold_and_weaver() {
        for (privateer, weaver, gold, attack) in [
            (PRIVATEER, WEAVER, 4, 6),
            ("BG33_825_G", "BG34_692_G", 6, 18),
        ] {
            let mut game = setup(&[ROGERS, privateer, weaver]);
            game.players[0].gold = 0;
            game.with_recruit(0, |ctx| ctx.give_card(0, cardid!("BG33_815")));
            game.apply(0, Action::Play { hand: 0, pos: 0 }).unwrap();
            assert_eq!(game.players[0].gold, gold);
            assert_eq!(game.players[0].stats.undead_atk, attack);
        }
    }

    #[test]
    fn tomb_turning_core_waits_until_the_death_mark_expires() {
        let mut weaver = Minion::new(400, db().id(WEAVER));
        weaver.doomed_turn = 11;
        assert!(!can_play_minion(&weaver, 11));
        assert!(can_play_minion(&weaver, 12));
    }

    #[test]
    fn bellringer_moves_phantom_right_and_uses_a_disposable_target() {
        let mut game = setup(&[BELLRINGER, PHANTOM, "BG_ICC_026t", HANDLESS]);
        game.players[0].stats.undead_atk = 12;
        let choose = |game: &Game| {
            let mask = game.legal_mask(0);
            prepare_bellringer(&game.players[0], &|action| mask[action.encode()]).unwrap()
        };
        let first = choose(&game);
        assert_eq!(first, Action::Move { from: 1, to: 3 });
        game.apply(0, first).unwrap();
        let p = &game.players[0];
        let hp = p.board.last().unwrap().hp;
        let fodder = p
            .board
            .iter()
            .find(|m| bellringer_target(p, m))
            .unwrap()
            .eid;
        let next = choose(&game);
        assert_eq!(next, Action::Activate(0));
        game.apply(0, next).unwrap();
        let Some(Pending::Target { options, .. }) = game.players[0].pending.last() else {
            panic!("Bellringer target menu");
        };
        let target = options.iter().position(|eid| *eid == fodder).unwrap();
        game.apply(0, Action::Choose(target)).unwrap();
        assert!(game.players[0].board.last().unwrap().hp > hp);
        assert!(apm::has(&game.players[0].board, HANDLESS));
    }

    #[test]
    fn finance_cycles_return_gold_and_busker_needs_the_next_turn() {
        let mut game = setup(&[ROGERS, PRIVATEER, WEAVER, PHANTOM, "BG36_098"]);
        game.players[0].gold = 0;
        let mask = game.legal_mask(0);
        let action = cycle_sale(&game, 0, &|action| mask[action.encode()]).unwrap();
        assert_eq!(action, Action::Sell(4));
        game.apply(0, action).unwrap();
        assert_eq!(game.players[0].gold, 1);
        assert_eq!(catalog::key(game.players[0].hand[0].item.card()), TENTACLE);
        game.apply(0, Action::Play { hand: 0, pos: 0 }).unwrap();
        game.apply(0, Action::Sell(0)).unwrap();
        assert_eq!(game.players[0].gold, 2);

        let mut game = setup(&[BRANN]);
        let busker = Minion::new(500, db().id(BUSKER));
        assert!(
            finance_minion(&game.players[0], &busker)
                > finance_minion(&setup(&[]).players[0], &busker)
        );
        game.with_recruit(0, |ctx| ctx.give_card(0, db().id(BUSKER)));
        game.apply(0, Action::Play { hand: 0, pos: 0 }).unwrap();
        assert_eq!(game.players[0].gold, 3);
        assert_eq!(game.players[0].gold_next_turn, 2);
    }

    #[test]
    fn graverobber_copies_magnetized_weaver_and_grows_phantom() {
        let mut game = setup(&[ROGERS, PRIVATEER, WEAVER, PHANTOM]);
        game.players[0].stats.undead_atk = 60;
        game.with_recruit(0, |ctx| {
            ctx.give_card(0, db().id(PROSTHETIC_HAND));
            ctx.give_card(0, db().id(GRAVEROBBER));
        });
        fn plan(p: &Player) -> apm::Plan<'_> {
            apm::Plan::new(p, Route::Undead).with_undead_specialist(true, true, 7)
        }
        let mask = game.legal_mask(0);
        let p = &game.players[0];
        let hand = p.hand[0].item.as_minion().unwrap();
        let magnet = plan(p)
            .magnetic_target(0, hand, &|a| mask[a.encode()])
            .unwrap();
        assert_eq!(magnet, Action::Magnetize { hand: 0, board: 2 });
        game.apply(0, magnet).unwrap();
        let p = &game.players[0];
        let hp = p.board.last().unwrap().hp;
        let original = p.board[2].eid;
        assert!(p.board[2].has(Kw::REBORN));
        game.apply(0, Action::Play { hand: 0, pos: 0 }).unwrap();
        let mask = game.legal_mask(0);
        let p = &game.players[0];
        let pending = p.pending.last().unwrap();
        let action = plan(p).pending(pending, &|a| mask[a.encode()]).unwrap();
        game.apply(0, action).unwrap();
        let p = &game.players[0];
        let revived = p.board.iter().find(|m| catalog::is(m, WEAVER)).unwrap();
        assert_ne!(revived.eid, original);
        assert!(!revived.has(Kw::REBORN));
        assert!(!revived.has_attached(PROSTHETIC_HAND));
        let copy = p.hand[0].item.as_minion().unwrap();
        assert!(catalog::is(copy, WEAVER));
        assert!(!copy.has(Kw::REBORN));
        assert!(p.board.last().unwrap().hp > hp + 60);
    }

    #[test]
    fn sacrifice_avoids_grown_phantom_and_respects_summon_space() {
        let mut game = setup(&[PHANTOM, "BG25_001", "BG25_009"]);
        let p = &mut game.players[0];
        p.stats.undead_atk = 70;
        p.board[0].give(Kw::REBORN);
        p.board[0].max_hp = 500;
        assert!(sacrifice_score(p, GRAVEROBBER, &p.board[0], 1).is_none());
        assert!(sacrifice_score(p, BUTCHERING, &p.board[1], 1).unwrap() >= 100);
        assert!(sacrifice_score(p, BUTCHERING, &p.board[2], 1).unwrap() < 100);
        assert!(sacrifice_score(p, BUTCHERING, &p.board[2], 2).unwrap() >= 100);
    }

    #[test]
    fn full_board_plus_795_converts_a_real_reborn_into_triple_phantom_growth() {
        let mut game = setup(&[
            "BG25_010_G",
            "BG36_515_G",
            "BG25_001",
            PHANTOM,
            WEAVER,
            ROGERS,
            PRIVATEER,
        ]);
        game.players[0].board[0].give(Kw::REBORN);
        game.with_recruit(0, |ctx| {
            ctx.bump_player_stat(0, |stats| stats.undead_atk = 795);
            ctx.give_card(0, db().id(BUTCHERING));
        });
        let p = &game.players[0];
        let handless = p.board[0].eid;
        let phantom = p.board[1].eid;
        let reborn = p.board[2].eid;
        let returned_attack = p.board[2].def().atk + 795;
        let (atk, hp) = (p.board[1].atk, p.board[1].hp);
        assert!(reborn_target_score(p, BUTCHERING, &p.board[0], 1).is_none());
        assert!(reborn_target_score(p, BUTCHERING, &p.board[4], 1).is_none());
        let prepare = |game: &Game| {
            let mask = game.legal_mask(0);
            prepare_reborn_sacrifice(game, 0, &|action| mask[action.encode()], false).unwrap()
        };
        let action = prepare(&game);
        assert_eq!(action, Action::Move { from: 1, to: 4 });
        game.apply(0, action).unwrap();
        let play = prepare(&game);
        assert_eq!(play, Action::Play { hand: 0, pos: 0 });
        game.apply(0, play).unwrap();
        let p = &game.players[0];
        let mask = game.legal_mask(0);
        let plan = apm::Plan::new(p, Route::Undead)
            .with_undead_specialist(true, true, 7)
            .with_reborn_conversion(true);
        let pending = p.pending.last().unwrap();
        let action = plan.pending(pending, &|a| mask[a.encode()]).unwrap();
        let Action::Choose(index) = action else {
            panic!("expected a target")
        };
        let Pending::Target { options, .. } = pending else {
            panic!("expected a target menu")
        };
        assert_eq!(options[index], reborn);
        game.apply(0, action).unwrap();
        let p = &game.players[0];
        let grown = p.board.iter().find(|m| m.eid == phantom).unwrap();
        assert_eq!(grown.hp - hp, returned_attack * 3);
        assert!(grown.atk - atk >= returned_attack * 3);
        assert!(
            p.board
                .iter()
                .any(|m| m.eid == handless && m.has(Kw::REBORN))
        );
        assert!(
            p.board
                .iter()
                .any(|m| catalog::is(m, "BG25_001") && m.eid != reborn)
        );
    }
}
