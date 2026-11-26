use crate::actions::{Action, N_HAND, N_POWERS};
use crate::bot;
use crate::cards::{CardId, db};
use crate::script::script;
use crate::state::{Continuation, Item, Minion, Pending, Player};
use crate::types::{Kw, Races};
use crate::{Game, cardid};

use super::{
    apm, azerite, cadence, catalog, catalog_plan, demon, pivot, quilboar, tea, undead_apm, v2,
};

const TURN_ACTION_LIMIT: u32 = 61;
const GOLD_VALUE: f64 = 1.6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Build {
    General,
    Azerite,
    Tea,
    Quilboar,
    Demon,
    Apm(apm::Route),
    Catalog(catalog::Route),
}

impl Build {
    pub fn evidence(self, player: &Player, turn: u32) -> i32 {
        match self {
            Self::Azerite => azerite::evidence(player),
            Self::Tea => tea::evidence(player),
            Self::Quilboar => quilboar::evidence(player),
            Self::Demon => demon::evidence(player, turn),
            _ => 0,
        }
    }

    // Scores from separate families are on different scales. A real engine on board
    // should beat a speculative catalog pivot built around an attractive shop offer.
    pub fn v1_incumbent(self, game: &Game, seat: usize) -> i32 {
        let p = &game.players[seat];
        let board = &p.board;
        if let Self::Apm(route) = self {
            let score = apm::candidate(game, seat, Some(route)).map_or(18, |(_, n)| 17 + n / 2);
            return score
                + if apm::economy(board) && route.payoff(board) {
                    18
                } else if apm::economy(board) {
                    8
                } else {
                    0
                };
        }
        let commitment = match self {
            Self::Azerite if azerite::pieces(board) & azerite::AZERITE != 0 => {
                12 + 8 * i32::from(azerite::pieces(board) & azerite::CORE == azerite::CORE)
            }
            Self::Tea if tea::has_core(tea::pieces(board)) => 24,
            Self::Tea if tea::pieces(board) & tea::GATE != 0 => 15,
            Self::Tea if p.tier >= 5 => 7,
            Self::Quilboar if quilboar::has_core(quilboar::pieces(board)) => 22,
            Self::Quilboar if quilboar::pieces(board) & quilboar::RIDER != 0 => 14,
            Self::Quilboar if p.stats.gem_atk + p.stats.gem_hp >= 5 => 8,
            Self::Demon if demon::has_engine(demon::pieces(board)) => 22,
            Self::Demon if demon::has_rewind(demon::pieces(board)) && game.turn <= 8 => 9,
            _ => 0,
        };
        self.evidence(p, game.turn) + commitment
    }

    pub fn adaptive(
        self,
        player: &Player,
        turn: u32,
        with_quilboar: bool,
        with_demon: bool,
    ) -> Self {
        let azerite = azerite::evidence(player);
        let tea = tea::evidence(player);
        let quills = if with_quilboar {
            quilboar::evidence(player)
        } else {
            0
        };
        let demons = if with_demon {
            demon::evidence(player, turn)
        } else {
            0
        };
        // Keep an established engine unless the other route is convincingly ahead.
        match self {
            Self::Azerite if azerite >= 10 && tea.max(quills).max(demons) < azerite + 7 => {
                Self::Azerite
            }
            Self::Tea if tea >= 10 && azerite.max(quills).max(demons) < tea + 7 => Self::Tea,
            Self::Quilboar if quills >= 10 && azerite.max(tea).max(demons) < quills + 7 => {
                Self::Quilboar
            }
            Self::Demon if demons >= 10 && azerite.max(tea).max(quills) < demons + 7 => Self::Demon,
            _ if azerite >= 13 && azerite >= tea.max(quills).max(demons) => Self::Azerite,
            _ if tea >= 13 && tea >= quills.max(demons) => Self::Tea,
            _ if quills >= 13 && quills >= demons => Self::Quilboar,
            _ if demons >= 13 => Self::Demon,
            _ => Self::General,
        }
    }
}

pub trait Plan {
    fn protects(&self, minion: &Minion) -> bool;
    fn keep_bonus(&self, minion: &Minion) -> f64;
    fn acquisition_bonus(&self, minion: &Minion, copies: usize) -> f64;
    fn spell_bonus(&self, card: CardId) -> f64;
    fn hand_priority(&self, item: &Item) -> f64;
    fn can_cast(&self, card: CardId) -> bool;
    fn activate(&self, legal: &dyn Fn(Action) -> bool) -> Option<Action>;
    fn pending(&self, menu: &Pending, legal: &dyn Fn(Action) -> bool) -> Option<Action>;
    fn upgrade(&self, game: &Game, seat: usize, ordinary: bool) -> bool;
    fn search_missing(&self, game: &Game, seat: usize) -> bool;
    fn attack_bonus(&self, _minion: &Minion) -> f64 {
        0.0
    }
}

/// A seat-local policy for the upstream 37.0.2 rules. It sees no opponent's private state.
pub struct Rules<'a> {
    pub cadence: bool,
    pub rush_six: bool,
    pub demon: bool,
    pub apm: bool,
    pub undead_apm: bool,
    pub undead_finance: bool,
    pub reborn_conversion: bool,
    pub v2: bool,
    pub transition: Option<&'a pivot::State>,
}

pub fn choose(game: &mut Game, seat: usize, build: Build, rules: Rules<'_>) -> Option<Action> {
    let Rules {
        cadence: with_cadence,
        rush_six,
        demon: with_demon,
        apm: with_apm,
        undead_apm: specialize_undead,
        undead_finance,
        reborn_conversion,
        v2: use_v2,
        transition,
    } = rules;
    let mask = game.legal_mask(seat);
    let legal = |action: Action| mask[action.encode()];
    let player = &game.players[seat];
    let undead_plan = specialize_undead && build == Build::Apm(apm::Route::Undead);
    let undead_finance = undead_plan && undead_finance;
    let reborn_conversion = undead_finance && reborn_conversion;
    let board_full = player.board.len() >= game.cfg.board_size;
    let azerite_plan;
    let tea_plan;
    let quilboar_plan;
    let demon_plan = (build == Build::Demon).then(|| demon::Plan::new(player, game.turn));
    let apm_plan = match build {
        Build::Apm(route) => Some(
            apm::Plan::new(player, route)
                .with_undead_specialist(specialize_undead, undead_finance, game.cfg.board_size)
                .with_reborn_conversion(reborn_conversion)
                .with_advanced(use_v2),
        ),
        Build::Catalog(catalog::Route::PirateGold) => {
            Some(apm::Plan::new(player, apm::Route::Golden).with_advanced(use_v2))
        }
        _ => None,
    };
    let catalog = match build {
        Build::Catalog(route) if route != catalog::Route::PirateGold => {
            Some(catalog_plan::Plan::new(player, route, game.turn).with_advanced(use_v2))
        }
        _ => None,
    };
    let plan: Option<&dyn Plan> = match build {
        Build::General => None,
        Build::Azerite => {
            azerite_plan = azerite::Plan::new(player, game.turn);
            Some(&azerite_plan)
        }
        Build::Tea => {
            tea_plan = tea::Plan::new(player, game.turn);
            Some(&tea_plan)
        }
        Build::Quilboar => {
            quilboar_plan = quilboar::Plan::new(player, game.turn);
            Some(&quilboar_plan)
        }
        Build::Demon => demon_plan.as_ref().map(|plan| plan as &dyn Plan),
        Build::Apm(_) => apm_plan.as_ref().map(|plan| plan as &dyn Plan),
        Build::Catalog(_) => apm_plan
            .as_ref()
            .map(|plan| plan as &dyn Plan)
            .or_else(|| catalog.as_ref().map(|plan| plan as &dyn Plan)),
    };
    let timing = with_cadence.then(|| cadence::Cadence::new(game, seat).with_six_rush(rush_six));
    let rush_upgrade = timing.as_ref().is_some_and(|t| t.rush_upgrade(game, seat));
    let reserve_six_refreshes = timing.as_ref().is_some_and(|t| t.reserve_six_refreshes());
    let v2 = use_v2.then(|| v2::Recruit::new(game, seat));
    let defer_gift = v2.as_ref().is_some_and(v2::Recruit::defer_gift);
    let mut valuation = Valuation::new(player, game.turn, plan, with_cadence);
    valuation.v2 = v2.as_ref();

    if let Some(menu) = player.pending.last() {
        if with_apm
            && let Some(action) =
                apm::tier_seven_choice(game, seat, menu, apm_plan.as_ref().map(|p| p.route), &legal)
        {
            return Some(action);
        }
        if with_demon
            && build != Build::Demon
            && let Some(action) = demon::cupcake_target(player, menu, &legal)
        {
            return Some(action);
        }
        if let Some(action) = plan.and_then(|plan| plan.pending(menu, &legal)) {
            return Some(action);
        }
        if let Some(action) = v2
            .as_ref()
            .and_then(|recruit| recruit.pending(menu, &legal))
        {
            return Some(action);
        }
        if let Some(action) = timing
            .as_ref()
            .and_then(|timing| timing.trinket(game, seat, menu, &legal))
        {
            return Some(action);
        }
        let ranked = match menu {
            Pending::Discover { options, .. } => options
                .iter()
                .enumerate()
                .filter(|(i, _)| legal(Action::Choose(*i)))
                .max_by(|(_, a), (_, b)| {
                    let score = |item| {
                        valuation.offer_value(item)
                            + if specialize_undead {
                                undead_apm::discover_bonus(game, seat, build, item)
                            } else {
                                0.0
                            }
                    };
                    score(a).total_cmp(&score(b))
                })
                .map(|(i, _)| Action::Choose(i)),
            Pending::Target {
                options,
                cont: Continuation::HandSpell { hid, .. },
            } if player.hand.iter().any(|h| {
                h.hid == *hid && h.item.card() == cardid!("BG28_604") // Butchering
            }) =>
            {
                options
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| legal(Action::Choose(*i)))
                    .filter_map(|(i, id)| {
                        player.board.iter().find(|m| m.eid == *id).map(|m| (i, m))
                    })
                    .min_by(|(_, a), (_, b)| {
                        valuation.keep_value(a).total_cmp(&valuation.keep_value(b))
                    })
                    .map(|(i, _)| Action::Choose(i))
            }
            _ => None,
        };
        if ranked.is_some() {
            return ranked;
        }
        let original = bot::choose_action(game, seat);
        return if legal(original) {
            Some(original)
        } else {
            mask.iter()
                .position(|allowed| *allowed)
                .and_then(Action::decode)
        };
    }

    let ordinary_limit = match game.action_cap_of(seat) {
        u32::MAX => TURN_ACTION_LIMIT,
        cap => cap,
    };
    let action_limit = apm_plan
        .as_ref()
        .filter(|p| p.is_cycling())
        .map_or(ordinary_limit, |p| p.action_budget(game, seat));
    if player.actions_this_turn >= action_limit {
        return legal(Action::EndTurn).then_some(Action::EndTurn);
    }
    let upgrade = if legal(Action::Upgrade) {
        let ordinary = should_upgrade(game, seat, &valuation);
        let ordinary = plan.map_or(ordinary, |plan| plan.upgrade(game, seat, ordinary));
        let level = timing.as_ref().map_or(ordinary, |timing| {
            timing.wants_upgrade(
                game,
                seat,
                build,
                ordinary,
                transition.is_some() && pivot::search_for_six(player, build),
            )
        });
        v2.as_ref()
            .map_or(level, |r| r.upgrade(game, seat, build, level))
    } else {
        false
    };
    valuation.delay_triple = upgrade;
    if undead_plan && let Some(action) = undead_apm::buy_core(game, seat, &legal) {
        return Some(action);
    }
    let sacrifice = if reborn_conversion {
        undead_apm::prepare_reborn_sacrifice(game, seat, &legal, false)
    } else if undead_finance {
        undead_apm::prepare_sacrifice(game, seat, &legal)
    } else {
        None
    };
    if let Some(action) = sacrifice {
        return Some(action);
    }
    if let Some(action) = transition.and_then(|state| state.buy)
        && legal(action)
    {
        // Bring the new component into hand before opening a board slot.
        return Some(action);
    }
    if let Some(action) = apm_plan
        .as_ref()
        .and_then(|p| p.prepare(game, seat, &legal))
        .filter(|action| *action != Action::Refresh || !rush_upgrade && !reserve_six_refreshes)
    {
        return Some(action);
    }
    if let Some(action) = catalog
        .as_ref()
        .and_then(|plan| plan.prepare(game, seat, &legal))
        .filter(|action| *action != Action::Refresh || !rush_upgrade && !reserve_six_refreshes)
    {
        return Some(action);
    }

    let carry = transition
        .filter(|state| state.preserve_carry)
        .and_then(|_| {
            player
                .board
                .iter()
                .max_by(|a, b| cadence::minion_strength(a).total_cmp(&cadence::minion_strength(b)))
                .map(|m| m.eid)
        });
    let weakest = player
        .board
        .iter()
        .enumerate()
        .filter(|(i, m)| {
            legal(Action::Sell(*i)) && carry != Some(m.eid) && !plan.is_some_and(|p| p.protects(m))
        })
        .map(|(i, m)| (i, valuation.keep_value(m)))
        .min_by(|(_, a), (_, b)| a.total_cmp(b));
    if legal(Action::Refresh)
        && !rush_upgrade
        && !reserve_six_refreshes
        && player.actions_this_turn < 40
        && demon_plan
            .as_ref()
            .is_some_and(|plan| plan.replenish(game, seat))
    {
        return Some(Action::Refresh);
    }
    if let Some(action) = play_hand(
        player,
        &valuation,
        &legal,
        weakest.filter(|_| board_full),
        demon_plan.as_ref(),
        apm_plan.as_ref(),
        catalog.as_ref(),
    ) {
        return Some(action);
    }
    if undead_finance && let Some(action) = undead_apm::cycle_sale(game, seat, &legal) {
        return Some(action);
    }

    if legal(Action::DarkGift)
        && !defer_gift
        && timing
            .as_ref()
            .is_some_and(|timing| timing.wants_early_gift(game, seat))
    {
        return Some(Action::DarkGift);
    }

    if let Some(action) = timing
        .as_ref()
        .and_then(|t| t.rush_cash_sale(game, seat))
        .filter(|&a| legal(a))
    {
        return Some(action);
    }

    if upgrade && timing.is_none() {
        return Some(Action::Upgrade);
    }

    // Free effects can add Gold or fill a gap before spending on the shop.
    for i in 0..N_POWERS {
        if legal(Action::HeroPower(i))
            && game.hero_power_price(seat, i) == Some(0)
            && timing
                .as_ref()
                .is_none_or(|timing| timing.can_power(game, seat, i))
        {
            return Some(Action::HeroPower(i));
        }
    }

    if !upgrade && let Some(action) = plan.and_then(|plan| plan.activate(&legal)) {
        return Some(action);
    }
    if !upgrade
        && let Some(action) = apm_plan
            .as_ref()
            .and_then(|plan| plan.recycle(game, seat, &legal))
    {
        return Some(action);
    }
    if !upgrade
        && let Some(action) = catalog
            .as_ref()
            .and_then(|plan| plan.cycle(game, seat, &legal))
    {
        return Some(action);
    }

    let mut best_purchase = None;
    let mut best_score = 0.0;
    let mut freeze = false;
    for (index, slot) in player.shop.iter().enumerate() {
        let Some((price, costs_health, _)) = game.slot_price(seat, index) else {
            continue;
        };
        let mut action = Action::Buy(index);
        let mut gain = valuation.offer_value(&slot.item)
            + timing
                .as_ref()
                .map_or(0.0, |timing| timing.shop_bonus(&slot.item));
        if let Item::Minion(minion) = &slot.item
            && board_full
            && valuation.copies(minion) != 2
            && !(minion.has(Kw::MAGNETIC) && player.board.iter().any(|m| m.is(Races::MECH)))
        {
            let Some((index, keep)) = weakest else {
                continue;
            };
            action = Action::Sell(index);
            // One board slot must be cleared before the purchase is useful.
            gain -= keep + 2.0;
        }
        let cost = f64::from(price) * if costs_health { 2.0 } else { GOLD_VALUE };
        let score = gain - cost;
        let can_buy = legal(Action::Buy(index));
        let before_level = !upgrade
            || timing.as_ref().is_some_and(|timing| {
                if rush_upgrade {
                    let replacement = if let Action::Sell(i) = action {
                        Some(i)
                    } else {
                        None
                    };
                    timing.before_rush_upgrade(game, seat, index, score, build, replacement)
                } else {
                    timing.pre_level_buy(game, seat, index, score, build)
                }
            });
        if score > best_score && can_buy && legal(action) && before_level {
            best_score = score;
            best_purchase = Some(action);
        }
        if !can_buy
            && !costs_health
            && price > player.gold
            && score >= 7.0
            && slot.item.as_minion().is_some()
        {
            freeze = true;
        }
    }
    if let Some(action) = best_purchase {
        return Some(action);
    }
    if upgrade {
        return Some(Action::Upgrade);
    }

    for i in 0..N_POWERS {
        if legal(Action::HeroPower(i))
            && timing
                .as_ref()
                .is_none_or(|timing| timing.can_power(game, seat, i))
        {
            return Some(Action::HeroPower(i));
        }
    }
    if legal(Action::DarkGift) && !board_full && !defer_gift {
        return Some(Action::DarkGift);
    }
    for i in 0..player.board.len() {
        if legal(Action::Activate(i))
            && !(undead_plan && catalog::is(&player.board[i], undead_apm::BELLRINGER))
            && catalog
                .as_ref()
                .is_none_or(|p| p.allow_activate(&player.board[i]))
        {
            return Some(Action::Activate(i));
        }
    }
    if freeze && !player.is_frozen() && legal(Action::Freeze) {
        return Some(Action::Freeze);
    }
    let ordinary_refresh = player.gold >= game.refresh_price(seat) + game.minion_price(seat)
        || plan.is_some_and(|plan| plan.search_missing(game, seat));
    let refresh = timing.as_ref().map_or(ordinary_refresh, |timing| {
        timing.can_refresh(game, seat)
            && (ordinary_refresh || player.tier >= 5 && player.free_refreshes > 0)
    });
    let refresh = demon_plan
        .as_ref()
        .filter(|plan| plan.health_refresh(game, seat))
        .map_or(refresh, |plan| plan.wants_health_refresh(game, seat));
    let apm_refresh = apm_plan
        .as_ref()
        .is_some_and(|plan| plan.refresh(game, seat));
    if !freeze
        && !reserve_six_refreshes
        && legal(Action::Refresh)
        && (refresh && player.actions_this_turn < 40 || apm_refresh)
    {
        return Some(Action::Refresh);
    }
    if reborn_conversion
        && let Some(action) = undead_apm::prepare_reborn_sacrifice(game, seat, &legal, true)
    {
        return Some(action);
    }
    if undead_plan
        && !undead_finance
        && player.actions_this_turn + 3 < action_limit
        && let Some(action) = undead_apm::prepare_bellringer(player, &legal)
    {
        return Some(action);
    }
    let position = if let Some(p) = &catalog {
        p.position(true, &legal)
    } else {
        improve_attack_order(player, &legal, plan, with_cadence)
    };
    if let Some(action) = position {
        return Some(action);
    }
    legal(Action::EndTurn).then_some(Action::EndTurn)
}

fn play_hand(
    player: &Player,
    valuation: &Valuation<'_>,
    legal: &impl Fn(Action) -> bool,
    replace: Option<(usize, f64)>,
    demon: Option<&demon::Plan<'_>>,
    apm: Option<&apm::Plan<'_>>,
    catalog: Option<&catalog_plan::Plan<'_>>,
) -> Option<Action> {
    let mut best = None;
    let mut best_priority = f64::NEG_INFINITY;
    for (hand, card) in player.hand.iter().enumerate().take(N_HAND) {
        if card.locked > 0 {
            continue;
        }
        let (action, priority) = match &card.item {
            Item::Spell { card, .. } => {
                if valuation.v2.is_some_and(|r| !r.can_cast(*card))
                    || valuation
                        .plan
                        .as_ref()
                        .is_some_and(|plan| !plan.can_cast(*card))
                {
                    continue;
                }
                (Action::Play { hand, pos: 0 }, 0.0)
            }
            Item::Minion(minion) => {
                if valuation
                    .v2
                    .is_some_and(|r| r.save_triple(minion, valuation.delay_triple))
                    || demon.is_some_and(|plan| !plan.can_play_minion(minion))
                    || apm.is_some_and(|plan| !plan.can_play_minion(minion, valuation.turn))
                    || catalog.is_some_and(|p| !p.can_play_minion(minion))
                {
                    continue;
                }
                let magnetic = apm
                    .and_then(|plan| plan.magnetic_target(hand, minion, legal))
                    .or_else(|| catalog.and_then(|p| p.magnetic(player, hand, legal)))
                    .or_else(|| {
                        (0..player.board.len())
                            .map(|board| Action::Magnetize { hand, board })
                            .find(|action| legal(*action))
                    });
                let summon = Action::Play {
                    hand,
                    pos: apm.map_or(player.board.len(), |plan| plan.play_position(minion)),
                };
                let value = valuation.offer_value(&card.item);
                if let Some(action) = magnetic {
                    (action, value)
                } else if legal(summon) {
                    // Play engines before the spells and battlecries they reward.
                    (summon, value + valuation.growth_value(minion))
                } else if let Some((index, keep)) = replace
                    && value > keep + 4.0
                {
                    (Action::Sell(index), value - keep)
                } else {
                    continue;
                }
            }
        };
        let priority = priority
            + valuation
                .plan
                .as_ref()
                .map_or(0.0, |plan| plan.hand_priority(&card.item));
        if legal(action) && priority > best_priority {
            best = Some(action);
            best_priority = priority;
        }
    }
    best
}

fn should_upgrade(game: &Game, seat: usize, valuation: &Valuation<'_>) -> bool {
    let player = valuation.player;
    let tier = usize::from(player.tier);
    let price = game.upgrade_price(seat);
    let Some((&target_turn, &minimum_board)) = [2, 5, 7, 9, 11]
        .get(tier - 1)
        .zip([1, 2, 4, 5, 6].get(tier - 1))
    else {
        return price <= 2 && valuation.future_weight >= 1.0;
    };
    if game.turn < target_turn || player.board.len() < minimum_board {
        return false;
    }
    // On low health, leave enough gold to buy tempo or wait for a cheap upgrade.
    if tier > 1 && valuation.future_weight < 1.0 {
        return price <= 2 || player.gold - price >= game.minion_price(seat);
    }
    tier < 5 || price <= 6 || player.gold - price >= game.minion_price(seat)
}

struct Valuation<'a> {
    player: &'a Player,
    turn: u32,
    future_weight: f64,
    knights: usize,
    azerite: bool,
    undead: usize,
    plan: Option<&'a dyn Plan>,
    with_cadence: bool,
    v2: Option<&'a v2::Recruit<'a>>,
    delay_triple: bool,
}

impl<'a> Valuation<'a> {
    fn new(player: &'a Player, turn: u32, plan: Option<&'a dyn Plan>, with_cadence: bool) -> Self {
        let durability = player.health + player.armor;
        let hurt = player.stats.last_damage <= -8;
        let future_weight = if turn >= 6 && (durability <= 12 || (durability <= 20 && hurt)) {
            0.55
        } else {
            1.0
        };
        Self {
            player,
            turn,
            future_weight,
            knights: player
                .board
                .iter()
                .filter(|m| m.is_card_id(cardid!("BG25_008")))
                .count(),
            plan,
            with_cadence,
            v2: None,
            delay_triple: false,
            azerite: player
                .board
                .iter()
                .any(|m| m.is_card_id(cardid!("BG28_707"))),
            undead: player.board.iter().filter(|m| m.is(Races::UNDEAD)).count(),
        }
    }

    fn copies(&self, m: &Minion) -> usize {
        if m.golden {
            return 0;
        }
        self.player
            .board
            .iter()
            .chain(self.player.hand.iter().filter_map(|h| h.item.as_minion()))
            .filter(|other| other.card == m.card && other.eid != m.eid)
            .count()
    }

    fn offer_value(&self, item: &Item) -> f64 {
        match item {
            Item::Minion(m) => {
                let copies = self.copies(m);
                let triple = match copies {
                    2 => 11.0,
                    1 => 2.0 * self.future_weight,
                    _ => 0.0,
                };
                let battlecry =
                    script(m.card)
                        .filter(|s| s.battlecry.is_some())
                        .map_or(0.0, |_| {
                            if self
                                .player
                                .board
                                .iter()
                                .any(|b| b.is_card_id(cardid!("BG_LOE_077")))
                            {
                                3.0
                            } else {
                                1.5
                            }
                        });
                self.keep_value(m)
                    + triple
                    + battlecry
                    + self
                        .plan
                        .as_ref()
                        .map_or(0.0, |plan| plan.acquisition_bonus(m, copies))
            }
            Item::Spell { card, .. } => {
                self.spell_value(*card)
                    + self
                        .plan
                        .as_ref()
                        .map_or(0.0, |plan| plan.spell_bonus(*card))
            }
        }
    }

    fn keep_value(&self, m: &Minion) -> f64 {
        let attack = f64::from(m.atk.max(0)).sqrt();
        let health = f64::from(m.hp.max(0)).sqrt();
        let mut value = 1.8 * attack + 1.6 * health + 0.4 * f64::from(m.tier);
        for (keyword, bonus) in [
            (Kw::DIVINE_SHIELD, 1.5 + 1.1 * attack),
            (Kw::REBORN, 2.1),
            (Kw::CLEAVE, 1.2 * attack),
            (Kw::WINDFURY, 0.5 * attack),
            (Kw::VENOMOUS, 3.0 + 0.35 * f64::from(self.turn.min(18))),
        ] {
            if m.has(keyword) {
                value += bonus;
            }
        }
        if script(m.card).is_some_and(|s| s.deathrattle.is_some()) {
            value += 1.5;
        }
        value
            + self.growth_value(m)
            + self.v2.map_or(0.0, |r| r.keep_bonus(m))
            + self.plan.as_ref().map_or(0.0, |plan| plan.keep_bonus(m))
            + if self.with_cadence {
                cadence::gift_value(m)
            } else {
                0.0
            }
    }

    fn growth_value(&self, m: &Minion) -> f64 {
        let id = db().get(m.card).normal.unwrap_or(m.card);
        let score = match db().get(id).id.as_str() {
            // Eternal Knight gets stronger every time one dies in combat.
            "BG25_008" => {
                if self.turn <= 7 {
                    5.0
                } else {
                    1.0
                }
            }
            "BG25_009" => 6.0 + (self.knights.min(3) as f64) * 2.0,
            "BG34_403" => 2.0 + self.undead.min(6) as f64,
            // Azerite grows the shop, and cheap spells let it do so several times a turn.
            "BG28_707" => 9.0,
            "BG35_881" | "BG36_701" | "BG35_883" if self.azerite => 7.0,
            "BG_LOE_077" => {
                if self.azerite {
                    8.0
                } else {
                    4.0
                }
            }
            // Scale other boards without requiring either featured tribe in the lobby.
            _ => {
                let effects = script(m.card);
                if effects.is_some_and(|s| s.end_of_turn.is_some()) {
                    4.0
                } else {
                    0.0
                }
            }
        };
        self.future_weight * score * if m.golden { 1.4 } else { 1.0 }
    }

    fn spell_value(&self, card: CardId) -> f64 {
        let bonus = self.v2.map_or(0.0, |r| r.spell_bonus(card));
        let card = db().get(card);
        let board = self.player.board.len() as f64;
        let value = match card.id.as_str() {
            "BG28_504" if self.player.board.len() < 5 => 8.0, // Recruit a Trainee
            "BG28_805" if self.turn <= 8 => 9.0 * self.future_weight, // Strike Oil
            "BG28_800" => 3.5 * self.future_weight,           // Careful Investment
            "BG28_168" | "BG35_951" => 1.0 + 0.9 * board,     // Board-wide stats
            "BG28_604" => 1.7 * self.undead.saturating_sub(1) as f64, // Butchering
            "BG28_810" => 0.0, // Buying and playing the Tavern Coin at sticker price nets zero.
            _ => 1.8 + 0.4 * f64::from(card.tier),
        };
        value
            + bonus
            + if self.azerite {
                4.5 * self.future_weight
            } else {
                0.0
            }
    }
}

fn improve_attack_order(
    player: &Player,
    legal: &impl Fn(Action) -> bool,
    plan: Option<&dyn Plan>,
    with_cadence: bool,
) -> Option<Action> {
    let priority = |m: &Minion| {
        if m.atk == 0 {
            return f64::NEG_INFINITY;
        }
        let mut score = f64::from(m.atk.max(0)).sqrt();
        if m.has(Kw::CLEAVE) {
            score += 10.0;
        }
        if m.has(Kw::DIVINE_SHIELD) {
            score += 3.0;
        }
        if script(m.card).is_some_and(|s| s.rally.is_some()) {
            score += 4.0;
        }
        if script(m.card).is_some_and(|s| s.avenge.is_some()) {
            score -= 8.0;
        }
        score
            + plan.map_or(0.0, |plan| plan.attack_bonus(m))
            + if with_cadence && cadence::refresh_deathrattle(m) {
                5.0
            } else {
                0.0
            }
    };
    // Put the best attacker ahead of a weaker one. Strict comparisons keep ties stable.
    for (to, front) in player.board.iter().enumerate() {
        if let Some((from, _)) = player
            .board
            .iter()
            .enumerate()
            .skip(to + 1)
            .filter(|(from, m)| {
                priority(m) > priority(front) && legal(Action::Move { from: *from, to })
            })
            .max_by(|(_, a), (_, b)| priority(a).total_cmp(&priority(b)))
        {
            return Some(Action::Move { from, to });
        }
    }
    None
}
