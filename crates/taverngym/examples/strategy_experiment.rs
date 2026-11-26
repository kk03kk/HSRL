//! Matched eight-player games with the simulator's default random lobby types.
//! Usage: cargo run --release -p taverngym --example strategy_experiment -- <first-seed> <pairs> <workers> <new-output-dir> v1.0-only

#[path = "strategy/apm_report.rs"]
mod apm_report;

#[path = "strategy/azerite_report.rs"]
mod azerite_report;
#[path = "strategy/catalog_report.rs"]
mod catalog_report;
#[path = "strategy/demon_report.rs"]
mod demon_report;
#[path = "strategy/quilboar_report.rs"]
mod quilboar_report;
#[path = "strategy/tea_report.rs"]
mod tea_report;
#[path = "strategy/undead_apm_report.rs"]
mod undead_apm_report;

use guide_bot::Build;
use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use taverngym::engine::Phase;
use taverngym::strategy::{
    Policy, PolicyState, apm, azerite, catalog, demon, guide_bot, hsreplay_picks, pivot, quilboar,
    tea, undead_apm,
};
use taverngym::types::Races;
use taverngym::{Action, Game, RulesConfig, bot, workers};

struct ResultRow {
    seed: u64,
    seat: usize,
    outcomes: Vec<Outcome>,
}

impl ResultRow {
    fn lobby(&self) -> Races {
        self.outcomes[0].lobby
    }
}

struct Outcome {
    lobby: Races,
    placement: u8,
    turn: u32,
    problem: Option<&'static str>,
    azerite: azerite_report::Formation,
    tea: tea_report::Formation,
    quilboar: quilboar_report::Formation,
    demon: demon_report::Formation,
    apm: apm_report::Formation,
    undead_proof: Option<undead_apm_report::Proof>,
    picks: hsreplay_picks::Decisions,
    catalog: catalog_report::Log,
    pivot: pivot::Trace,
    used_azerite: bool,
    used_tea: bool,
    used_quilboar: bool,
    used_demon: bool,
    bought_turn_one: bool,
    gifts_by_five: u32,
    tier_two_turn: Option<u32>,
    tier_three_turn: Option<u32>,
    tier_five_turn: Option<u32>,
    tier_six_turn: Option<u32>,
    last_recruit_turn: u32,
    free_rolls_before_five: u32,
    free_rolls_at_five_or_six: u32,
}

impl Outcome {
    fn formed_routes(&self) -> impl Iterator<Item = &'static str> + '_ {
        [
            (
                "azerite",
                self.lobby.contains(Races::ELEMENTAL.union(Races::MURLOC))
                    && self.azerite.full_turn.is_some(),
            ),
            ("tea", self.tea.four_turn.is_some()),
            (
                "quilboar",
                self.lobby.contains(Races::QUILBOAR) && self.quilboar.five_turn.is_some(),
            ),
            (
                "demon",
                self.lobby.contains(Races::DEMON) && self.demon.six_turn.is_some(),
            ),
        ]
        .into_iter()
        .filter_map(|(route, full)| full.then_some(route))
        .chain(
            apm::Route::ALL
                .into_iter()
                .filter_map(|route| self.apm.routes[route.index()].full.map(|_| route.name())),
        )
        .chain(
            catalog::Route::ALL
                .into_iter()
                .filter(|route| {
                    self.catalog.routes[route.index()].full.is_some() && route.full_pool(self.lobby)
                })
                .map(|route| route.spec().key),
        )
    }

    fn pivot_engine_worked(&self, build: Build) -> bool {
        match build {
            Build::Tea => self.tea.working_turn.is_some(),
            Build::Quilboar => self.quilboar.working_turn.is_some(),
            Build::Apm(route) => self.apm.routes[route.index()].working.is_some(),
            Build::Catalog(route) => self.catalog.routes[route.index()].started.is_some(),
            _ => false,
        }
    }

    fn record_turn(&mut self, game: &Game, seat: usize, turn: u32, tier: u8) {
        self.azerite.record_turn(game, seat, turn, tier);
        self.tea.record_turn(game, seat, turn, tier);
        self.quilboar.record_turn(game, seat, turn, tier);
        self.demon.record_turn(game, seat, turn, tier);
        self.apm.record_turn(game, seat, turn, tier);
        if let Some(proof) = &mut self.undead_proof {
            proof.record_turn(&game.players[seat], turn, tier);
        }
        self.last_recruit_turn = turn;
        if turn == 1 {
            self.bought_turn_one = game.players[seat].stats.minions_bought > 0;
        }
        if turn <= 5 {
            self.gifts_by_five = game.players[seat].dark_gift_uses;
        }
        if tier >= 2 {
            self.tier_two_turn.get_or_insert(turn);
        }
        if tier >= 3 {
            self.tier_three_turn.get_or_insert(turn);
        }
        if tier >= 5 {
            self.tier_five_turn.get_or_insert(turn);
        }
        if tier >= 6 {
            self.tier_six_turn.get_or_insert(turn);
        }
    }
}

fn run(
    seed: u64,
    seat: usize,
    policy: Policy,
    output: &Path,
    record_picks: bool,
    fixed_pivot_lobby: bool,
) -> io::Result<Outcome> {
    let mut config = RulesConfig::hsrl();
    if fixed_pivot_lobby {
        config.lobby_types = pivot::LOBBY.map(str::to_owned).to_vec();
    }
    let lobby_size = config.lobby_races;
    let mut game = Game::new(seed, config);
    assert_eq!(game.lobby_races.iter().count(), lobby_size);
    if fixed_pivot_lobby {
        assert_eq!(game.lobby_races, pivot::LOBBY_MASK);
    }
    let mut outcome = Outcome {
        lobby: game.lobby_races,
        placement: 0,
        turn: 0,
        problem: None,
        azerite: Default::default(),
        tea: Default::default(),
        quilboar: Default::default(),
        demon: Default::default(),
        apm: Default::default(),
        undead_proof: fixed_pivot_lobby.then(undead_apm_report::Proof::default),
        picks: Default::default(),
        catalog: Default::default(),
        pivot: Default::default(),
        used_azerite: false,
        used_tea: false,
        used_quilboar: false,
        used_demon: false,
        bought_turn_one: false,
        gifts_by_five: 0,
        tier_two_turn: None,
        tier_three_turn: None,
        tier_five_turn: None,
        tier_six_turn: None,
        last_recruit_turn: 0,
        free_rolls_before_five: 0,
        free_rolls_at_five_or_six: 0,
    };
    outcome.problem = drive(
        &mut game,
        seat,
        policy,
        &mut outcome,
        record_picks,
        fixed_pivot_lobby,
    )
    .err()
    .or_else(|| {
        if game.turn_limit_end {
            return Some("turn_limit");
        }
        if game.truncated_combats > 0 || game.truncated_recruit_contexts > 0 {
            return Some("engine_truncation");
        }
        let mut placements: Vec<_> = game.players.iter().map(|p| p.placement).collect();
        placements.sort_unstable();
        (placements != [1, 2, 3, 4, 5, 6, 7, 8]).then_some("invalid_ranks")
    });
    if game.stopped_reason.is_some() {
        outcome.problem.get_or_insert("simulator_stopped");
    }
    if outcome.problem.is_some() {
        serde_json::to_writer(
            File::create(output.join(format!("{}-{seed}.json", policy.name())))?,
            &game,
        )?;
    }
    outcome.placement = game.players[seat].placement;
    outcome.turn = game.turn;
    Ok(outcome)
}

fn drive(
    game: &mut Game,
    subject: usize,
    policy: Policy,
    outcome: &mut Outcome,
    record_picks: bool,
    trace_pivots: bool,
) -> Result<(), &'static str> {
    let mut strategy = PolicyState::new(policy);
    for _ in 0..10_000 {
        if game.is_over() {
            return Ok(());
        }
        let mut progressed = false;
        for seat in 0..game.players.len() {
            if !game.needs_action(seat) {
                continue;
            }
            let turn = game.turn;
            if seat != subject || policy == Policy::Original {
                let recruit = game.phase == Phase::Recruit;
                let acted = bot::play_turn(game, seat);
                if !acted && game.needs_action(seat) {
                    return Err("original_bot_stalled");
                }
                progressed |= acted;
                let player = &game.players[seat];
                if seat == subject && recruit && acted && (player.ended_turn || game.turn != turn) {
                    outcome.record_turn(game, seat, turn, player.tier);
                }
                if acted && (game.turn != turn || game.is_over()) {
                    outcome.apm.after_combat(&game.players[subject], turn);
                    if trace_pivots {
                        outcome.pivot.after_combat(game, subject, turn);
                    }
                    if policy.uses_catalog() {
                        outcome.catalog.after_combat(game, subject, turn);
                    }
                }
                continue;
            }
            for _ in 0..200 {
                if game.turn != turn || !game.needs_action(seat) {
                    break;
                }
                let player = &game.players[seat];
                let tier = player.tier;
                let build = strategy.update(game, seat);
                let bank = player.free_refreshes;
                outcome.used_azerite |= build == Build::Azerite;
                outcome.used_tea |= build == Build::Tea;
                outcome.used_quilboar |= build == Build::Quilboar;
                outcome.used_demon |= build == Build::Demon;
                if let Build::Apm(route) = build {
                    outcome.apm.routes[route.index()].selected = true;
                }
                let action = strategy
                    .choose(game, seat, record_picks.then_some(&mut outcome.picks))
                    .ok_or("candidate_no_legal_action")?;
                let azerite = azerite_report::Witness::before_action(game, seat, action);
                let tea = tea_report::Cast::before_action(game, seat, action);
                let quills = quilboar_report::Choice::before_action(game, seat, action);
                let demons = demon_report::Witness::before_action(game, seat, action);
                let apm = apm_report::Witness::before_action(game, seat, action);
                let undead = outcome
                    .undead_proof
                    .as_ref()
                    .and_then(|_| undead_apm_report::Witness::before(game, seat, action));
                if trace_pivots && action == Action::EndTurn {
                    outcome.pivot.recruit_end(game, seat, build);
                }
                let route_witness = if let Build::Catalog(route) = build {
                    outcome.catalog.begin(route, game, seat, action)
                } else {
                    if action == Action::EndTurn && policy.uses_catalog() {
                        outcome.catalog.stop();
                    }
                    None
                };
                if !game.legal_mask(seat)[action.encode()] {
                    return Err("candidate_action_not_in_mask");
                }
                game.apply(seat, action)
                    .map_err(|_| "candidate_rejected_action")?;
                if action == Action::Refresh && bank > game.players[seat].free_refreshes {
                    if tier < 5 {
                        outcome.free_rolls_before_five += 1;
                    } else {
                        outcome.free_rolls_at_five_or_six += 1;
                    }
                }
                if let Some(witness) = azerite {
                    outcome.azerite.record_action(witness, game, seat);
                }
                if let Some(cast) = tea {
                    outcome.tea.record_cast(cast, &game.players[seat]);
                }
                if let Some(choice) = quills {
                    outcome.quilboar.record_choice(choice, &game.players[seat]);
                }
                if let Some(witness) = demons {
                    witness.record(&mut outcome.demon, &game.players[seat]);
                }
                if let Some(witness) = apm {
                    witness.record(&mut outcome.apm, &game.players[seat], turn);
                }
                if let (Some(witness), Some(proof)) = (undead, &mut outcome.undead_proof) {
                    witness.apply(proof, &game.players[seat], turn);
                }
                if let Some(witness) = route_witness {
                    outcome.catalog.finish(witness, &game.players[seat]);
                }
                if action == Action::EndTurn {
                    outcome.record_turn(game, seat, turn, tier);
                }
                if game.turn != turn || game.is_over() {
                    outcome.apm.after_combat(&game.players[subject], turn);
                    if trace_pivots {
                        outcome.pivot.after_combat(game, subject, turn);
                    }
                    if policy.uses_catalog() {
                        outcome.catalog.after_combat(game, subject, turn);
                    }
                }
                progressed = true;
            }
        }
        if !progressed && !game.is_over() {
            return Err("no_progress");
        }
    }
    Err("driver_limit")
}

fn write_pairs(out: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    let mut csv = BufWriter::new(File::create(out.join("pairs.csv"))?);
    write!(csv, "seed,seat,lobby_types")?;
    let columns = [
        "rank",
        "turn",
        "problem",
        "azerite_core_turn",
        "azerite_full_turn",
        "azerite_100_100_turn",
        "tea_core_turn",
        "tea_working_turn",
        "tea_four_turn",
        "tea_targeted_casts",
        "tea_gem_casts",
        "tea_other_attack_gain",
        "tea_other_health_gain",
        "tea_final_piece_mask",
        "used_azerite",
        "used_tea",
        "bought_turn_one",
        "gifts_by_five",
        "tier_three_turn",
        "tier_five_turn",
        "tier_six_turn",
        "last_recruit_turn",
        "free_rolls_before_five",
        "free_rolls_at_five_or_six",
        "quilboar_core_turn",
        "quilboar_working_turn",
        "quilboar_five_turn",
        "quilboar_gem_waves",
        "quilboar_gem_day_waves",
        "quilboar_added_gems",
        "used_quilboar",
        "azerite_working_turn",
    ];
    let demon_columns = [
        "demon_rewinder_turn",
        "demon_engine_turn",
        "demon_working_turn",
        "demon_six_turn",
        "demon_final_mask",
        "demon_safely_rewound",
        "demon_unrecovered_damage",
        "demon_cupcakes_gained",
        "demon_targeted",
        "demon_off_target",
        "demon_missed_distractor",
        "demon_health_refreshes",
        "demon_percussion_plays",
        "demon_spell_shop_atk",
        "demon_spell_shop_hp",
        "demon_cupcakes_consumed",
        "used_demon",
    ];
    let demon_mode = arms.contains(&Policy::FourBuilds);
    let apm_mode = arms.contains(&Policy::EightBuilds);
    let pick_mode = arms.contains(&Policy::HsBoth);
    for policy in arms {
        for column in columns {
            write!(csv, ",{}_{column}", policy.name())?;
        }
        if demon_mode {
            for column in demon_columns {
                write!(csv, ",{}_{column}", policy.name())?;
            }
        }
        if apm_mode {
            for column in [
                "loop_turn",
                "peak_spend",
                "peak_refund",
                "bounties_received",
                "bounties_cast",
                "wealthy_gold",
            ] {
                write!(csv, ",{}_apm_{column}", policy.name())?;
            }
            for route in ["undead", "tavern", "mech", "deity"] {
                for column in [
                    "selected",
                    "core_turn",
                    "working_turn",
                    "full_turn",
                    "final_full",
                    "gain",
                    "extra",
                    "capstone",
                ] {
                    write!(csv, ",{}_apm_{route}_{column}", policy.name())?;
                }
            }
        }
        if pick_mode {
            for column in [
                "hero_dbf",
                "hero_best_dbf",
                "hero_rated",
                "hero_changed",
                "lesser_choices",
                "lesser_rated",
                "lesser_changed",
                "greater_choices",
                "greater_rated",
                "greater_changed",
            ] {
                write!(csv, ",{}_picks_{column}", policy.name())?;
            }
        }
    }
    writeln!(csv)?;
    let turn = |value: Option<u32>| value.map(|v| v.to_string()).unwrap_or_default();
    for row in rows {
        write!(csv, "{},{},{:?}", row.seed, row.seat, row.lobby())?;
        for (policy, game) in arms.iter().zip(&row.outcomes) {
            let measured = *policy != Policy::Original;
            let metric = |value: i64| {
                if measured {
                    value.to_string()
                } else {
                    String::new()
                }
            };
            write!(
                csv,
                ",{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                game.placement,
                game.turn,
                game.problem.unwrap_or(""),
                turn(game.azerite.core_turn),
                turn(game.azerite.full_turn),
                turn(game.azerite.big_full_turn),
                turn(game.tea.core_turn),
                turn(game.tea.working_turn),
                turn(game.tea.four_turn),
                metric(i64::from(game.tea.targeted_casts)),
                metric(i64::from(game.tea.gem_casts)),
                metric(game.tea.other_attack_gain),
                metric(game.tea.other_health_gain),
                game.tea.last.as_ref().map_or(0, |board| board.pieces),
                game.used_azerite,
                game.used_tea,
                game.bought_turn_one,
                game.gifts_by_five,
                turn(game.tier_three_turn),
                turn(game.tier_five_turn),
                turn(game.tier_six_turn),
                game.last_recruit_turn,
                metric(i64::from(game.free_rolls_before_five)),
                metric(i64::from(game.free_rolls_at_five_or_six))
            )?;
            write!(
                csv,
                ",{},{},{},{},{},{},{}",
                turn(game.quilboar.core_turn),
                turn(game.quilboar.working_turn),
                turn(game.quilboar.five_turn),
                metric(i64::from(game.quilboar.gem_waves)),
                metric(i64::from(game.quilboar.gem_day_waves)),
                metric(game.quilboar.added_gems as i64),
                game.used_quilboar
            )?;
            write!(csv, ",{}", turn(game.azerite.working_turn))?;
            if demon_mode {
                let demon = &game.demon;
                write!(
                    csv,
                    ",{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    turn(demon.rewinder_turn),
                    turn(demon.engine_turn),
                    turn(demon.working_turn),
                    turn(demon.six_turn),
                    demon.last.as_ref().map_or(0, |b| b.pieces),
                    metric(i64::from(demon.safely_rewound)),
                    metric(i64::from(demon.unrecovered_damage)),
                    metric(i64::from(demon.cupcakes_gained)),
                    metric(i64::from(demon.targeted)),
                    metric(i64::from(demon.off_target)),
                    metric(i64::from(demon.missed_distractor)),
                    metric(i64::from(demon.health_refreshes)),
                    metric(i64::from(demon.percussion_plays)),
                    metric(i64::from(demon.spell_shop_atk)),
                    metric(i64::from(demon.spell_shop_hp)),
                    metric(i64::from(demon.cupcakes_consumed)),
                    game.used_demon,
                )?;
            }
            if apm_mode {
                let apm = &game.apm;
                write!(
                    csv,
                    ",{},{},{},{},{},{}",
                    turn(apm.loop_turn),
                    apm.peak_spend,
                    apm.peak_refund,
                    apm.bounties_received,
                    apm.bounties_cast,
                    apm.wealthy_gold
                )?;
                for progress in &apm.routes {
                    write!(
                        csv,
                        ",{},{},{},{},{},{},{},{}",
                        progress.selected,
                        turn(progress.core),
                        turn(progress.working),
                        turn(progress.full),
                        progress.final_full,
                        progress.gain,
                        progress.extra,
                        progress.capstone
                    )?;
                }
            }
            if pick_mode {
                let picks = &game.picks;
                write!(
                    csv,
                    ",{},{},{},{},{},{},{},{},{},{}",
                    picks.hero_dbf,
                    picks.hero_best_dbf,
                    picks.hero_rated,
                    picks.hero_changed,
                    picks.lesser[0],
                    picks.lesser[1],
                    picks.lesser[2],
                    picks.greater[0],
                    picks.greater[1],
                    picks.greater[2],
                )?;
            }
        }
        writeln!(csv)?;
    }
    csv.flush()
}

fn summarize_arm(label: &str, games: &[&Outcome], out: &mut impl Write) -> io::Result<()> {
    let valid: Vec<_> = games
        .iter()
        .copied()
        .filter(|g| g.problem.is_none())
        .collect();
    let n = valid.len();
    writeln!(out, "{label}: {n}/{} valid games", games.len())?;
    if n == 0 {
        return Ok(());
    }
    let count = |predicate: fn(&Outcome) -> bool| valid.iter().filter(|g| predicate(g)).count();
    let rate = |total: usize| 100.0 * total as f64 / n as f64;
    let mean = valid.iter().map(|g| f64::from(g.placement)).sum::<f64>() / n as f64;
    writeln!(
        out,
        "  mean rank {mean:.3}, top 4 {:.2}%, top 2 {:.2}%, first {:.2}%",
        rate(count(|g| g.placement <= 4)),
        rate(count(|g| g.placement <= 2)),
        rate(count(|g| g.placement == 1))
    )?;
    writeln!(
        out,
        "  cadence: bought a minion T1 {:.1}%; 3 gifts by T5 {:.1}%; tier 3 by T5 {:.1}%; tier 5 by T8 {:.1}%, by T9 {:.1}%; tier 6 by T10 {:.1}%; last recruit at T9 or earlier and out of top four {:.1}%",
        rate(count(|g| g.bought_turn_one)),
        rate(count(|g| g.gifts_by_five >= 3)),
        rate(count(|g| g.tier_three_turn.is_some_and(|t| t <= 5))),
        rate(count(|g| g.tier_five_turn.is_some_and(|t| t <= 8))),
        rate(count(|g| g.tier_five_turn.is_some_and(|t| t <= 9))),
        rate(count(|g| g.tier_six_turn.is_some_and(|t| t <= 10))),
        rate(count(|g| g.last_recruit_turn <= 9 && g.placement > 4)),
    )?;
    if label != Policy::Original.name() {
        let low: u32 = valid.iter().map(|g| g.free_rolls_before_five).sum();
        let high: u32 = valid.iter().map(|g| g.free_rolls_at_five_or_six).sum();
        writeln!(
            out,
            "  banked free refreshes used per game: {:.2} below tier 5, {:.2} at tier 5+",
            f64::from(low) / n as f64,
            f64::from(high) / n as f64
        )?;
    }
    let core = count(|g| g.tea.core_turn.is_some());
    let working = count(|g| g.tea.working_turn.is_some());
    let guide_four = count(|g| g.tea.four_turn.is_some());
    let final_core = count(|g| g.tea.last.as_ref().is_some_and(|b| tea::has_core(b.pieces)));
    writeln!(
        out,
        "  tea: core {core} ({:.1}%), guide four {guide_four}, final core {final_core}",
        rate(core)
    )?;
    if label != Policy::Original.name() {
        let casts: u32 = valid.iter().map(|g| g.tea.targeted_casts).sum();
        writeln!(
            out,
            "  tea working {working} ({:.1}%); spells aimed from hand at Gatekeeper: {casts}; games using Azerite plan {}; tea plan {}; both {}",
            rate(working),
            count(|g| g.used_azerite),
            count(|g| g.used_tea),
            count(|g| g.used_azerite && g.used_tea)
        )?;
    } else {
        writeln!(
            out,
            "  tea activation telemetry unavailable for the unmodified original bot"
        )?;
    }
    Ok(())
}

fn summarize_pick_choices(games: &[&Outcome], out: &mut impl Write) -> io::Result<()> {
    let mut heroes = [0; 3];
    let mut lesser = [0; 3];
    let mut greater = [0; 3];
    for g in games.iter().filter(|g| g.problem.is_none()) {
        let p = &g.picks;
        heroes[0] += u32::from(p.hero_dbf != 0);
        heroes[1] += u32::from(p.hero_rated);
        heroes[2] += u32::from(p.hero_changed);
        for i in 0..3 {
            lesser[i] += p.lesser[i];
            greater[i] += p.greater[i];
        }
    }
    writeln!(
        out,
        "  Choice events (picked / HSReplay-rated / changed vs previous rule in the same state): hero {} / {} / {}; lesser {} / {} / {}; greater {} / {} / {}",
        heroes[0],
        heroes[1],
        heroes[2],
        lesser[0],
        lesser[1],
        lesser[2],
        greater[0],
        greater[1],
        greater[2],
    )
}

fn summarize_pair(
    rows: &[ResultRow],
    control: usize,
    candidate: usize,
    label: &str,
    out: &mut impl Write,
) -> io::Result<()> {
    let (mut count, mut sum, mut squares) = (0, 0.0, 0.0);
    for row in rows {
        let a = &row.outcomes[control];
        let b = &row.outcomes[candidate];
        if a.problem.is_some() || b.problem.is_some() {
            continue;
        }
        let delta = f64::from(a.placement) - f64::from(b.placement);
        count += 1;
        sum += delta;
        squares += delta * delta;
    }
    if count == 0 {
        return writeln!(out, "{label}: no valid pairs");
    }
    let n = count as f64;
    let mean = sum / n;
    let margin = if count > 1 {
        1.96 * ((squares / n - mean * mean).max(0.0) / (n - 1.0)).sqrt()
    } else {
        f64::NAN
    };
    writeln!(
        out,
        "{label}: {count} pairs, rank improvement {mean:+.3} (approx. 95% CI [{:+.3}, {:+.3}])",
        mean - margin,
        mean + margin
    )
}

fn summarize_quilboar(policy: Policy, games: &[&Outcome], out: &mut impl Write) -> io::Result<()> {
    let valid: Vec<_> = games
        .iter()
        .copied()
        .filter(|g| g.problem.is_none())
        .collect();
    let n = valid.len();
    if n == 0 {
        return Ok(());
    }
    let count = |predicate: fn(&Outcome) -> bool| valid.iter().filter(|g| predicate(g)).count();
    let core = count(|g| g.quilboar.core_turn.is_some());
    let five = count(|g| g.quilboar.five_turn.is_some());
    let working = count(|g| g.quilboar.working_turn.is_some());
    let final_core = count(|g| {
        g.quilboar
            .last
            .as_ref()
            .is_some_and(|b| quilboar::has_core(b.pieces))
    });
    writeln!(
        out,
        "  Quilboar: core {core} ({:.1}%), guide five {five}, final core {final_core}",
        100.0 * core as f64 / n as f64
    )?;
    if policy != Policy::Original {
        let waves: u64 = valid.iter().map(|g| u64::from(g.quilboar.gem_waves)).sum();
        let gem_days: u64 = valid
            .iter()
            .map(|g| u64::from(g.quilboar.gem_day_waves))
            .sum();
        writeln!(
            out,
            "  Quilboar working {working} ({:.1}%); wins among working {}; observed gem waves {waves}, of which Gem Day {gem_days}; games using Quilboar plan {}",
            100.0 * working as f64 / n as f64,
            count(|g| g.placement == 1 && g.quilboar.working_turn.is_some()),
            count(|g| g.used_quilboar)
        )?;
    }
    Ok(())
}

fn summarize_demon(policy: Policy, games: &[&Outcome], out: &mut impl Write) -> io::Result<()> {
    let valid: Vec<_> = games
        .iter()
        .copied()
        .filter(|g| g.problem.is_none())
        .collect();
    if valid.is_empty() {
        return Ok(());
    }
    let count = |predicate: fn(&Outcome) -> bool| valid.iter().filter(|g| predicate(g)).count();
    let sum = |value: fn(&demon_report::Formation) -> i64| -> i64 {
        valid.iter().map(|g| value(&g.demon)).sum()
    };
    writeln!(
        out,
        "  Demons: Soul Rewinder by T6 {}; three-piece engine {}; six-piece board {}; six pieces on final board {}",
        count(|g| g.demon.rewinder_turn.is_some_and(|turn| turn <= 6)),
        count(|g| g.demon.engine_turn.is_some()),
        count(|g| g.demon.six_turn.is_some()),
        count(|g| g
            .demon
            .last
            .as_ref()
            .is_some_and(|b| b.pieces & demon::SIX == demon::SIX)),
    )?;
    if policy != Policy::Original {
        let working = count(|g| g.demon.working_turn.is_some());
        writeln!(
            out,
            "  Demon working {working} ({:.1}%); wins among working {}; games using Demon plan {}; cupcakes gained during self-damage {}; cupcakes aimed at Distractor {}, at other targets {}; permanent shop gain on aimed cupcakes +{}/+{}; shop minions removed in those casts {}",
            100.0 * working as f64 / valid.len() as f64,
            count(|g| g.placement == 1 && g.demon.working_turn.is_some()),
            count(|g| g.used_demon),
            sum(|f| i64::from(f.cupcakes_gained)),
            sum(|f| i64::from(f.targeted)),
            sum(|f| i64::from(f.off_target)),
            sum(|f| i64::from(f.spell_shop_atk)),
            sum(|f| i64::from(f.spell_shop_hp)),
            sum(|f| i64::from(f.cupcakes_consumed)),
        )?;
        writeln!(
            out,
            "  Observed recruit damage {} points; {} on fully rewound actions; {} net unrecovered",
            sum(|f| i64::from(f.damage)),
            sum(|f| i64::from(f.safely_rewound)),
            sum(|f| i64::from(f.unrecovered_damage)),
        )?;
        writeln!(
            out,
            "  Cupcakes aimed elsewhere with Distractor available {}; Malchezaar health refreshes {}; Percussionist plays {}",
            sum(|f| i64::from(f.missed_distractor)),
            sum(|f| i64::from(f.health_refreshes)),
            sum(|f| i64::from(f.percussion_plays)),
        )?;
    }
    Ok(())
}

struct RouteCoverage {
    name: &'static str,
    start_types: Races,
    full_types: Races,
    started: fn(&Outcome) -> bool,
    assembled: fn(&Outcome) -> bool,
    final_board: fn(&Outcome) -> bool,
}

fn write_route_coverage(output: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    let routes = [
        RouteCoverage {
            name: "活化艾泽里特",
            start_types: Races::ELEMENTAL.union(Races::MURLOC),
            full_types: Races::ELEMENTAL.union(Races::MURLOC),
            started: |g| g.azerite.working_turn.is_some(),
            assembled: |g| g.azerite.full_turn.is_some(),
            final_board: |g| {
                g.azerite
                    .last
                    .as_ref()
                    .is_some_and(|b| b.pieces == azerite::FULL)
            },
        },
        RouteCoverage {
            name: "喝茶流",
            start_types: Races::NONE,
            full_types: Races::QUILBOAR,
            started: |g| g.tea.working_turn.is_some(),
            assembled: |g| g.tea.four_turn.is_some(),
            final_board: |g| {
                g.tea
                    .last
                    .as_ref()
                    .is_some_and(|b| b.pieces & tea::GUIDE_FOUR == tea::GUIDE_FOUR)
            },
        },
        RouteCoverage {
            name: "野猪人抉择流",
            start_types: Races::QUILBOAR,
            full_types: Races::QUILBOAR,
            started: |g| g.quilboar.working_turn.is_some(),
            assembled: |g| g.quilboar.five_turn.is_some(),
            final_board: |g| {
                g.quilboar
                    .last
                    .as_ref()
                    .is_some_and(|b| b.pieces == quilboar::GUIDE_FIVE)
            },
        },
        RouteCoverage {
            name: "恶魔自伤流",
            start_types: Races::DEMON,
            full_types: Races::DEMON,
            started: |g| g.demon.working_turn.is_some(),
            assembled: |g| g.demon.six_turn.is_some(),
            final_board: |g| {
                g.demon
                    .last
                    .as_ref()
                    .is_some_and(|b| b.pieces & demon::SIX == demon::SIX)
            },
        },
    ];
    let lobbies: BTreeSet<_> = rows.iter().map(|row| row.lobby().0).collect();
    let mut md = BufWriter::new(File::create(output.join("route-coverage.md"))?);
    writeln!(
        md,
        "# 随机大厅套路覆盖\n\n{} 个种子覆盖 {} 种大厅组合。各项分母使用满足本项种族条件的有效对局。",
        rows.len(),
        lobbies.len(),
    )?;
    writeln!(
        md,
        "\n- 活化艾泽里特要求元素和鱼人。启动要求四件循环组件同场，海藻护卫实际对魔网浮灵进击，且元素酒馆属性已增长。齐集为五件组件同场。"
    )?;
    writeln!(
        md,
        "- 喝茶流启动适用于所有大厅。要求守门融合怪与巴琳达或黑暗傀儡师同场，并观察到至少五次定向法术以及其他随从至少 60 点生命增量。攻略四件含热气球测绘员，齐集和终局四件的分母使用有野猪人的大厅。"
    )?;
    writeln!(
        md,
        "- 抉择流要求野猪人。启动要求核心在场，且已观察到至少三次抉择结算为多只野猪人添加宝石。齐集为攻略五件同场。"
    )?;
    writeln!(
        md,
        "- 自伤流要求恶魔。启动要求三个核心在场，实测自伤产糕点，且至少两次在三核同时在场时将糕点喂给恶魔干扰者并提高永久酒馆属性。齐集为六核同场。"
    )?;
    writeln!(
        md,
        "\n齐集统计曾在一次招募结束时同时持有完整组件的局数。终局统计最后一次招募结束时仍持有完整组件的局数。"
    )?;
    let share = |count, total| {
        if total == 0 {
            "—".to_owned()
        } else {
            format!(
                "{count}/{total}（{:.2}%）",
                100.0 * count as f64 / total as f64
            )
        }
    };
    for (arm, policy) in arms
        .iter()
        .enumerate()
        .filter(|(_, p)| **p != Policy::Original)
    {
        writeln!(md, "\n## {}\n", policy.name())?;
        writeln!(
            md,
            "| 路线 | 启动 | 启动后夺冠 | 齐集 | 终局完整 |\n| --- | ---: | ---: | ---: | ---: |"
        )?;
        for route in &routes {
            let valid = rows
                .iter()
                .map(|row| &row.outcomes[arm])
                .filter(|g| g.problem.is_none());
            let start = valid
                .clone()
                .filter(|g| route.start_types.is_none() || g.lobby.contains(route.start_types));
            let full = valid.filter(|g| g.lobby.contains(route.full_types));
            let started = start.clone().filter(|g| (route.started)(g));
            let full_count = full.clone().count();
            writeln!(
                md,
                "| {} | {} | {} | {} | {} |",
                route.name,
                share(started.clone().count(), start.count()),
                started.filter(|g| g.placement == 1).count(),
                share(
                    full.clone().filter(|g| (route.assembled)(g)).count(),
                    full_count
                ),
                share(full.filter(|g| (route.final_board)(g)).count(), full_count),
            )?;
        }
    }
    md.flush()
}

fn write_demon_showcase(out: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    let mut md = BufWriter::new(File::create(out.join("showcase-demon.md"))?);
    writeln!(
        md,
        "# 恶魔自伤流终局战团\n\n优先展示完整六核，再按排名和定向糕点次数选择达成实战启动的案例。"
    )?;
    for (arm, policy) in arms
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p, Policy::Demon | Policy::FourBuilds))
    {
        let mut examples: Vec<_> = rows
            .iter()
            .filter_map(|row| {
                let game = &row.outcomes[arm];
                let board = game.demon.last.as_ref()?;
                let turn = game.demon.working_turn?;
                (game.problem.is_none() && demon::has_engine(board.pieces)).then_some((
                    game.placement,
                    row,
                    turn,
                    board,
                ))
            })
            .collect();
        examples.sort_by_key(|(rank, row, _, board)| {
            (
                board.pieces & demon::SIX != demon::SIX,
                *rank,
                board.cupcake_donors_consumed == 0,
                Reverse(board.targeted),
                Reverse(board.attack + board.health),
                row.seed,
            )
        });
        writeln!(md, "\n## {} 策略", policy.name())?;
        if examples.is_empty() {
            writeln!(md, "\n本轮没有达到条件的终局战团。")?;
        }
        let with_cycle_minion = examples.iter().skip(1).find(|(rank, _, _, board)| {
            *rank == 1 && board.pieces & demon::SIX == demon::SIX && board.preferred_cycle_minion
        });
        for (rank, row, turn, board) in examples
            .first()
            .into_iter()
            .chain(with_cycle_minion.or_else(|| examples.get(1)))
        {
            writeln!(
                md,
                "\n### 种子 {}，座位 {}，第 {} 名\n\n种族池 {:?}。第 {} 回合首次达到实战启动标准。",
                row.seed,
                row.seat,
                rank,
                row.lobby(),
                turn,
            )?;
            demon_report::write_board(&mut md, board)?;
            serde_json::to_writer_pretty(
                File::create(out.join(format!(
                    "{}-demon-warband-{}.json",
                    policy.name(),
                    row.seed
                )))?,
                board,
            )?;
        }
    }
    md.flush()
}

fn write_apm_report(output: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    use apm::Route;
    let mut report = BufWriter::new(File::create(output.join("apm-coverage.md"))?);
    let mut showcase = BufWriter::new(File::create(output.join("showcase-apm.md"))?);
    writeln!(
        report,
        "# 海盗经济与四条手速路线\n\n每条路线使用海盗和对应种族同时开放的有效对局作为分母。可以有多条路线在同一局被尝试。"
    )?;
    writeln!(
        report,
        "\n共同经济标准要求罗杰斯和私掠者同场时，某回合实际花费至少 15 金、回收至少 4 金，且累计施放至少两张悬赏令。回收包含卖牌与法术等实际金币流入。"
    )?;
    writeln!(
        report,
        "\n- 亡灵记录纺织工在场时通过法术增加至少 12 点全局亡灵攻击。"
    )?;
    writeln!(
        report,
        "- 酒馆强化记录邪能野猪人吞食至少三只酒馆随从，并观察到其中至少两次吞食时已存在酒馆加成。"
    )?;
    writeln!(
        report,
        "- 磁力记录有起床号时累计至少六次磁力，其中至少两次由购买触发起床号。"
    )?;
    writeln!(
        report,
        "- 神明记录有疑之影时买牌让神明的攻生命合计至少增长 72 点，且实战中神明确实苏醒。"
    )?;
    writeln!(
        report,
        "\n启动同时要求该路线的收益牌仍在场。毕业为报告中的攻略核心同场，磁力与神明分别包含七本拳手与惧之煞。毕业和启动分别计数。"
    )?;
    for (arm, p) in arms
        .iter()
        .enumerate()
        .filter(|(_, p)| **p != Policy::Original)
    {
        writeln!(report, "\n## {}\n", p.name())?;
        writeln!(
            report,
            "| 路线 | 有效机会 | 选线 | 经济二件与收益牌 | 启动 | 启动后夺冠 | 曾齐集攻略核心 | 终局核心齐集 | 七本效果实测 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )?;
        for route in Route::ALL {
            let eligible = rows
                .iter()
                .map(|r| &r.outcomes[arm])
                .filter(|g| g.problem.is_none() && g.lobby.contains(route.required()));
            let counts = eligible.clone().fold([0; 7], |mut counts, g| {
                let p = &g.apm.routes[route.index()];
                counts[0] += usize::from(p.selected);
                counts[1] += usize::from(p.core.is_some());
                counts[2] += usize::from(p.working.is_some());
                counts[3] += usize::from(p.working.is_some() && g.placement == 1);
                counts[4] += usize::from(p.full.is_some());
                counts[5] += usize::from(p.final_full);
                counts[6] += usize::from(p.capstone > 0);
                counts
            });
            writeln!(
                report,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                route.name(),
                eligible.count(),
                counts[0],
                counts[1],
                counts[2],
                counts[3],
                counts[4],
                counts[5],
                if matches!(route, Route::Mech | Route::Deity) {
                    counts[6].to_string()
                } else {
                    "—".to_owned()
                },
            )?;
        }
        let games = rows
            .iter()
            .map(|r| &r.outcomes[arm])
            .filter(|g| g.problem.is_none() && g.lobby.intersects(Races::PIRATE));
        let loops = games.clone().filter(|g| g.apm.loop_turn.is_some()).count();
        let spent20 = games.clone().filter(|g| g.apm.peak_spend >= 20).count();
        let peak = games.clone().map(|g| g.apm.peak_spend).max().unwrap_or(0);
        writeln!(
            report,
            "\n海盗大厅共 {} 局，{} 局达到经济标准，{} 局在双经济牌同场时单回合花费至少 20 金，单回合最高 {} 金。",
            games.count(),
            loops,
            spent20,
            peak
        )?;
    }
    let arm = arms
        .iter()
        .position(|p| *p == Policy::EightBuilds)
        .expect("APM policy");
    writeln!(
        showcase,
        "# 四条手速路线的终局战团\n\n优先展示已启动同时终局攻略核心齐集的案例。"
    )?;
    for route in Route::ALL {
        let i = route.index();
        let best = rows
            .iter()
            .filter_map(|row| {
                let game = &row.outcomes[arm];
                let progress = &game.apm.routes[i];
                let board = game.apm.last.as_ref()?;
                (game.problem.is_none() && progress.working.is_some())
                    .then_some((row, game, progress, board))
            })
            .min_by_key(|(row, game, p, board)| {
                (
                    !p.final_full,
                    matches!(route, Route::Mech | Route::Deity) && p.capstone == 0,
                    game.placement,
                    Reverse(game.apm.peak_spend),
                    Reverse(board.attack + board.health),
                    row.seed,
                )
            });
        writeln!(showcase, "\n## {}\n", route.name())?;
        if let Some((row, game, p, board)) = best {
            writeln!(
                showcase,
                "种子 {}，座位 {}，种族池 {:?}，第 {} 名。英雄 {}，第 {} 回合结束，{} 本。第 {} 回合达到启动标准。终局核心{}。\n",
                row.seed,
                row.seat,
                row.lobby(),
                game.placement,
                board.hero.unwrap_or("未知"),
                board.turn,
                board.tier,
                p.working.expect("started"),
                if p.final_full {
                    "齐集"
                } else {
                    "仍缺组件"
                },
            )?;
            writeln!(
                showcase,
                "双经济牌同场时单回合最高花费 {} 金、最高回收 {} 金。期间共施放 {} 张悬赏令，财富悬赏令累计实际产出 {} 金。战团合计 {}/{}。\n",
                game.apm.peak_spend,
                game.apm.peak_refund,
                game.apm.bounties_cast,
                game.apm.wealthy_gold,
                board.attack,
                board.health,
            )?;
            match route {
                Route::Undead => writeln!(
                    showcase,
                    "见证法术期间亡灵全局攻击 +{}，终局全局亡灵攻击 +{}。\n",
                    p.gain, board.undead_attack
                )?,
                Route::Tavern => writeln!(
                    showcase,
                    "邪能野猪人累计吞食 {} 个酒馆随从，其中 {} 次结算时存在酒馆加成，终局通用永久酒馆属性 +{}/+{}。\n",
                    p.gain, p.extra, board.tavern_attack, board.tavern_health
                )?,
                Route::Mech => writeln!(
                    showcase,
                    "起床号在场时见证 {} 次磁力，{} 次购买触发起床号吸附。\n",
                    p.gain, p.extra
                )?,
                Route::Deity => writeln!(
                    showcase,
                    "买牌期间神明攻击与生命合计增长 {} 点，达到启动条件后的苏醒战斗数 {}，终局神明 {}/{}。\n",
                    p.gain, p.extra, board.deity_attack, board.deity_health
                )?,
                Route::Golden => unreachable!("tracked by catalog"),
            }
            if matches!(route, Route::Mech | Route::Deity) {
                writeln!(showcase, "实测七本核心效果触发 {} 次。\n", p.capstone)?;
            }
            writeln!(showcase, "| 随从 | 攻击 | 生命 |\n| --- | ---: | ---: |")?;
            for m in &board.minions {
                writeln!(
                    showcase,
                    "| {}{}{} | {} | {} |",
                    m.name,
                    if m.golden { "〈金色〉" } else { "" },
                    if m.reborn { "〈复生〉" } else { "" },
                    m.attack,
                    m.health
                )?;
            }
            serde_json::to_writer_pretty(
                File::create(output.join(format!("apm-{}-warband-{}.json", i, row.seed)))?,
                board,
            )?;
        } else {
            writeln!(showcase, "本轮没有达到启动标准的对局。")?;
        }
    }
    report.flush()?;
    showcase.flush()
}

fn write_quilboar_showcase(out: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    let mut md = BufWriter::new(File::create(out.join("showcase-quilboar.md"))?);
    writeln!(
        md,
        "# 野猪人抉择流终局战团\n\n以下案例达到实战启动标准，并把骑士和至少一张抉择来源保留到最后一次招募结束。"
    )?;
    for (arm, policy) in arms
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p, Policy::Quilboar | Policy::AllBuilds | Policy::FourBuilds))
    {
        let mut examples: Vec<_> = rows
            .iter()
            .filter_map(|row| {
                let game = &row.outcomes[arm];
                let board = game.quilboar.last.as_ref()?;
                let turn = game.quilboar.working_turn?;
                (game.problem.is_none() && quilboar::has_core(board.pieces)).then_some((
                    game.placement,
                    row,
                    turn,
                    board,
                ))
            })
            .collect();
        examples.sort_by_key(|(rank, row, _, board)| {
            (*rank, Reverse(board.attack + board.health), row.seed)
        });
        writeln!(md, "\n## {} 策略", policy.name())?;
        if examples.is_empty() {
            writeln!(md, "\n本轮没有符合条件的终局战团。")?;
        }
        let full = examples
            .iter()
            .skip(1)
            .find(|(_, _, _, b)| b.pieces == quilboar::GUIDE_FIVE);
        for (rank, row, turn, board) in examples
            .first()
            .into_iter()
            .chain(full.or_else(|| examples.get(1)))
        {
            writeln!(
                md,
                "\n### 种子 {}，座位 {}，第 {} 名\n\n种族池 {:?}。第 {} 回合首次达到实战启动标准。",
                row.seed,
                row.seat,
                rank,
                row.lobby(),
                turn
            )?;
            quilboar_report::write_board(&mut md, board)?;
            serde_json::to_writer_pretty(
                File::create(out.join(format!(
                    "{}-quill-warband-{}.json",
                    policy.name(),
                    row.seed
                )))?,
                board,
            )?;
        }
    }
    md.flush()
}

fn write_tea_showcase(out: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    let mut md = BufWriter::new(File::create(out.join("showcase-tea.md"))?);
    writeln!(
        md,
        "# 喝茶流终局战团\n\n以下案例保留喝茶核心至终局，并达到实战启动标准。优先展示吃鸡局。"
    )?;
    for (arm, policy) in arms.iter().enumerate().filter(|(_, p)| {
        matches!(
            p,
            Policy::Tea | Policy::Combined | Policy::AllBuilds | Policy::FourBuilds
        )
    }) {
        let mut examples: Vec<_> = rows
            .iter()
            .filter_map(|row| {
                let game = &row.outcomes[arm];
                let board = game.tea.last.as_ref()?;
                let turn = game.tea.working_turn?;
                (game.problem.is_none() && tea::has_core(board.pieces)).then_some((
                    game.placement,
                    row,
                    turn,
                    board,
                ))
            })
            .collect();
        examples.sort_by_key(|(placement, row, _, board)| {
            (*placement, Reverse(board.attack + board.health), row.seed)
        });
        writeln!(md, "\n## {} 策略", policy.name())?;
        if examples.is_empty() {
            writeln!(md, "\n本轮没有达到条件的终局战团。")?;
        }
        let guide_board = examples
            .iter()
            .skip(1)
            .find(|(_, _, _, board)| board.pieces & tea::GUIDE_FOUR == tea::GUIDE_FOUR);
        for (placement, row, turn, board) in examples
            .first()
            .into_iter()
            .chain(guide_board.or_else(|| examples.get(1)))
        {
            writeln!(
                md,
                "\n### 种子 {}，座位 {}，第 {} 名\n\n种族池 {:?}。第 {} 回合首次达到启动标准。",
                row.seed,
                row.seat,
                placement,
                row.lobby(),
                turn
            )?;
            tea_report::write_board(&mut md, board)?;
            serde_json::to_writer_pretty(
                File::create(out.join(format!("{}-warband-{}.json", policy.name(), row.seed)))?,
                board,
            )?;
        }
    }
    md.flush()
}

fn write_azerite_showcase(out: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    let mut report = BufWriter::new(File::create(out.join("showcase-azerite.md"))?);
    writeln!(
        report,
        "# 活化艾泽里特终局战团\n\n五张组件在最后一次招募结束时仍同时在场。"
    )?;
    for (arm, policy) in arms.iter().enumerate().filter(|(_, p)| {
        matches!(
            p,
            Policy::Azerite | Policy::Combined | Policy::AllBuilds | Policy::FourBuilds
        )
    }) {
        let best = rows
            .iter()
            .filter_map(|row| {
                let game = &row.outcomes[arm];
                let board = game
                    .azerite
                    .last
                    .as_ref()
                    .filter(|b| b.pieces == azerite::FULL)?;
                (game.problem.is_none()).then_some((game.placement, row, board))
            })
            .min_by_key(|(placement, row, board)| {
                (*placement, Reverse(board.attack + board.health), row.seed)
            });
        if let Some((placement, row, board)) = best {
            writeln!(
                report,
                "\n## {} 策略，种子 {}，座位 {}，第 {} 名\n",
                policy.name(),
                row.seed,
                row.seat,
                placement
            )?;
            azerite_report::write_board(&mut report, board)?;
        } else {
            writeln!(report, "\n{} 策略本轮没有五件齐的终局战团。", policy.name())?;
        }
    }
    report.flush()
}

fn write_undead_apm_report(output: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    let mut report = BufWriter::new(File::create(output.join("undead-apm.md"))?);
    let mut csv = BufWriter::new(File::create(output.join("undead-apm.csv"))?);
    writeln!(
        report,
        "# 手速亡灵专项\n\n固定海盗、亡灵、机械、野猪人、畸变怪。每席每回合 100 次招募操作，菜单可收尾。"
    )?;
    writeln!(
        report,
        "\n三核心指罗杰斯、私掠者、纺织工同场。旧口径启动沿用八流派评测。严格启动要求三核心在场时同一回合花费至少 15 金、实际流入至少 4 金、施放至少两张不同手牌实例的悬赏令，并观察到至少 +12 全局亡灵攻击。五核心额外要求断手和魅影同场。"
    )?;
    writeln!(
        report,
        "\n| 策略 | 有效局 | 曾选线 | 经济与成长三核心 | 旧口径启动 | 严格启动 | 曾齐五核心 | 严格启动且当轮齐五核心 | 五核心且金色复生断手 | 终局五核心 | 齐五核心且夺冠 | 招募阶段丧钟给魅影成长 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    writeln!(
        csv,
        "seed,seat,policy,placement,selected,trio_turn,old_working_turn,strict_turn,five_core_turn,strict_full_turn,golden_reborn_turn,final_full,bellringer_growth_count,phantom_health_added,peak_spend,peak_spell_undead_attack,finance_buys,trio_gold_spent,trio_gold_inflow,butchering_growth,maw_caster_growth,graverobber_growth,sacrifice_phantom_hp,graverobber_core_copies,max_phantom_hp,peak_phantom_gain"
    )?;
    let mut finance_rows = Vec::with_capacity(arms.len());
    for (i, policy) in arms.iter().enumerate() {
        let mut counts = [0; 11];
        let mut finance = [0u64; 10];
        for row in rows {
            let g = &row.outcomes[i];
            if g.problem.is_some() {
                continue;
            }
            let proof = g.undead_proof.as_ref().expect("fixed lobby observer");
            let route = &g.apm.routes[apm::Route::Undead.index()];
            for (total, value) in finance.iter_mut().zip([
                u64::from(proof.finance_buys),
                proof.trio_gold_spent as u64,
                proof.trio_gold_inflow as u64,
                u64::from(proof.sacrifice_growth[0]),
                u64::from(proof.sacrifice_growth[1]),
                u64::from(proof.sacrifice_growth[2]),
                u64::from(proof.sacrifice_growth[3]),
                u64::from(proof.permanent_phantom_hp > 0),
                u64::from(proof.max_phantom_hp >= 1000),
                u64::from(proof.core_copies),
            ]) {
                *total += value;
            }
            for (count, matches) in counts.iter_mut().zip([
                true,
                route.selected,
                proof.trio.is_some(),
                route.working.is_some(),
                proof.same_turn_loop.is_some(),
                route.full.is_some(),
                proof.full_with_loop.is_some(),
                proof.golden_reborn.is_some(),
                proof.final_full,
                route.full.is_some() && g.placement == 1,
                proof.bellringer_gains > 0,
            ]) {
                *count += usize::from(matches);
            }
            write!(
                csv,
                "{},{},{},{},{}",
                row.seed,
                row.seat,
                policy.name(),
                g.placement,
                route.selected
            )?;
            for t in [
                proof.trio,
                route.working,
                proof.same_turn_loop,
                route.full,
                proof.full_with_loop,
                proof.golden_reborn,
            ] {
                write!(
                    csv,
                    ",{}",
                    t.map_or_else(String::new, |turn| turn.to_string())
                )?;
            }
            writeln!(
                csv,
                ",{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                proof.final_full,
                proof.bellringer_gains,
                proof.phantom_health_added,
                proof.peak_spend,
                proof.peak_spell_growth,
                proof.finance_buys,
                proof.trio_gold_spent,
                proof.trio_gold_inflow,
                proof.sacrifice_growth[1],
                proof.sacrifice_growth[2],
                proof.sacrifice_growth[3],
                proof.permanent_phantom_hp,
                proof.core_copies,
                proof.max_phantom_hp,
                proof.peak_phantom_gain,
            )?;
        }
        write!(report, "| {} |", policy.name())?;
        for count in counts {
            write!(report, " {count} |")?;
        }
        writeln!(report)?;
        finance_rows.push((policy.name(), finance));
    }

    writeln!(
        report,
        "\n| 策略 | 三核心在场理财购买 | 三核心在场总花费 | 三核心在场总金币流入 | 丧钟成长次数 | 宰割成长次数 | 噬渊成长次数 | 盗墓成长次数 | 四种消灭效果给魅影成长的局数 | 招募结束魅影至少千血的局数 | 盗墓净增核心副本 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for (policy, finance) in finance_rows {
        write!(report, "| {policy} |")?;
        for value in finance {
            write!(report, " {value} |")?;
        }
        writeln!(report)?;
    }

    let arm = arms
        .iter()
        .position(|p| *p == Policy::RushSix)
        .or_else(|| arms.iter().position(|p| *p == Policy::UndeadReborn))
        .or_else(|| arms.iter().position(|p| *p == Policy::UndeadFinance))
        .or_else(|| arms.iter().position(|p| *p == Policy::UndeadApm))
        .expect("undead specialist");
    if let Some((row, trace, board)) = rows
        .iter()
        .filter_map(|row| {
            let result = &row.outcomes[arm];
            let trace = result.undead_proof.as_ref()?;
            let board = trace.tallest_phantom.as_ref()?;
            (result.problem.is_none() && result.apm.routes[apm::Route::Undead.index()].selected)
                .then_some((row, trace, board))
        })
        .max_by_key(|(row, trace, _)| {
            (
                trace.final_full,
                trace.permanent_phantom_hp,
                trace.max_phantom_hp,
                Reverse(row.seed),
            )
        })
    {
        writeln!(
            report,
            "\n手速亡灵招募成长代表局为种子 {}，座位 {}，第 {} 回合，魅影 {} 血。按每次结算前后最高血量的魅影计算，四类消灭效果累计净增 {} 生命，单次最多净增 {} 生命。该局最终第 {} 名。",
            row.seed,
            row.seat,
            board.turn,
            trace.max_phantom_hp,
            trace.permanent_phantom_hp,
            trace.peak_phantom_gain,
            row.outcomes[arm].placement
        )?;
        serde_json::to_writer_pretty(
            File::create(output.join(format!("phantom-{}.json", row.seed)))?,
            board,
        )?;
    }
    let best = rows
        .iter()
        .filter_map(|row| {
            let result = &row.outcomes[arm];
            let trace = result.undead_proof.as_ref()?;
            let board = trace.best.as_ref()?;
            result
                .problem
                .is_none()
                .then_some((row, result, trace, board))
        })
        .min_by_key(|(row, result, trace, board)| {
            (
                !trace.final_full,
                result.placement,
                trace.golden_reborn.is_none(),
                trace.full_with_loop.is_none(),
                Reverse(board.undead_attack),
                row.seed,
            )
        });
    if let Some((row, result, trace, board)) = best {
        writeln!(
            report,
            "\n## 代表战团\n\n种子 {}，座位 {}，最终第 {} 名。第 {} 回合、{} 本，英雄 {}。本局终局五核心 {}。该战团的全局亡灵攻击 +{}。三核心同时在场的单回合最高花费 {} 金，单回合最多记录 +{} 法术亡灵攻击。",
            row.seed,
            row.seat,
            result.placement,
            board.turn,
            board.tier,
            board.hero.unwrap_or("未知"),
            trace.final_full,
            board.undead_attack,
            trace.peak_spend,
            trace.peak_spell_growth
        )?;
        writeln!(
            report,
            "\n| 站位 | 随从 | 攻击 | 生命 | 复生 |\n| --- | --- | ---: | ---: | --- |"
        )?;
        for (index, m) in board.minions.iter().enumerate() {
            writeln!(
                report,
                "| {} | {}{} | {} | {} | {} |",
                index + 1,
                m.name,
                if m.golden { "〈金色〉" } else { "" },
                m.attack,
                m.health,
                if m.reborn { "有" } else { "—" }
            )?;
        }
        serde_json::to_writer_pretty(
            File::create(output.join(format!("undead-apm-{}.json", row.seed)))?,
            board,
        )?;
    }
    report.flush()?;
    csv.flush()
}

fn write_pivot_report(output: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    use std::collections::BTreeMap;

    let mut report = BufWriter::new(File::create(output.join("pivot-report.md"))?);
    let mut csv = BufWriter::new(File::create(output.join("pivot-games.csv"))?);
    writeln!(
        report,
        "# 五族转型专项\n\n前期过渡指第 8 回合及以前使用永恒骑士、腐泥或手牌宝石策略结束招募，且主牌在场。兑现转型指第 9 回合起在五本及以上使用另一套后期策略结束招募，且该路线指定的启动组件全在场。每局只计第一次兑现。"
    )?;
    writeln!(
        report,
        "\n| 策略 | 有效局 | 前期走过渡 | 前期骑士 | 兑现转型 | 新路线验证启动 | 前期骑士局兑现 | 骑士策略打到最后一战 | 转型当轮战斗胜率 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    writeln!(
        csv,
        "seed,seat,policy,rank,early_bridge,early_knight_turn,knight_at_end,tier_five_turn,tier_six_turn,from,to,turn,previous_board_strength,pivot_board_strength,combat_won,engine_worked"
    )?;
    let mut paths = BTreeMap::new();
    for (i, policy) in arms.iter().enumerate() {
        let (
            mut valid,
            mut early_bridge,
            mut early_knight,
            mut conversions,
            mut knight_conversions,
            mut knight_at_end,
        ) = (0, 0, 0, 0, 0, 0);
        let (mut wins, mut fought) = (0, 0);
        let mut working = 0;
        for row in rows {
            let g = &row.outcomes[i];
            if g.problem.is_some() {
                continue;
            }
            valid += 1;
            let trace = &g.pivot;
            early_bridge += usize::from(trace.had_bridge());
            early_knight += usize::from(trace.knight_first.is_some());
            knight_at_end += usize::from(trace.knight_at_end);
            let turn = |v: Option<u32>| v.map(|t| t.to_string()).unwrap_or_default();
            write!(
                csv,
                "{},{},{},{},{},{},{},{},{}",
                row.seed,
                row.seat,
                policy.name(),
                g.placement,
                trace.had_bridge(),
                turn(trace.knight_first),
                trace.knight_at_end,
                turn(g.tier_five_turn),
                turn(g.tier_six_turn)
            )?;
            if let Some(c) = &trace.conversion {
                conversions += 1;
                let engine_worked = g.pivot_engine_worked(c.to);
                working += usize::from(engine_worked);
                knight_conversions += usize::from(trace.knight_first.is_some());
                if let Some(won) = c.won {
                    wins += usize::from(won);
                    fought += 1;
                }
                let (from, to) = (c.from.spec().name, pivot::name(c.to));
                *paths.entry((policy.name(), from, to)).or_insert(0) += 1;
                writeln!(
                    csv,
                    ",{from},{to},{},{:.2},{:.2},{},{engine_worked}",
                    c.turn,
                    c.before,
                    c.after,
                    c.won.map(|v| v.to_string()).unwrap_or_default()
                )?;
            } else {
                writeln!(csv, ",,,,,,,")?;
            }
        }
        let win_rate = if fought == 0 {
            "—".to_string()
        } else {
            format!("{:.1}%", 100.0 * wins as f64 / fought as f64)
        };
        writeln!(
            report,
            "| {} | {valid} | {early_bridge} | {early_knight} | {conversions} | {working} | {knight_conversions} | {knight_at_end} | {win_rate} |",
            policy.name()
        )?;
    }
    writeln!(
        report,
        "\n| 策略 | 前期起点 | 后期路线 | 首次兑现局数 |\n| --- | --- | --- | ---: |"
    )?;
    for ((policy, from, to), count) in paths {
        writeln!(report, "| {policy} | {from} | {to} | {count} |")?;
    }
    writeln!(
        report,
        "\n验证启动沿用各路线的实战验收条件，按整局核查。逐局 CSV 包含转型前一回合及转型当回合招募结束的战团估值，和转型当轮是否获胜。估值基于攻血、圣盾、复生等基础因素。"
    )?;
    report.flush()?;
    csv.flush()
}

fn write_v1_report(
    output: &Path,
    arms: &[Policy],
    arm: usize,
    rows: &[ResultRow],
) -> io::Result<()> {
    let mut report = BufWriter::new(File::create(output.join("v1-coverage.md"))?);
    let mut cases = BufWriter::new(File::create(output.join("v1-showcase.md"))?);
    let mut raw = BufWriter::new(File::create(output.join("v1-routes.csv"))?);
    writeln!(
        report,
        "# AutoTavern 策略 Bot {}\n\n{} 个大厅种子，覆盖 {} 种种族组合。所有对手均为原版 Bot。",
        arms[arm].name(),
        rows.len(),
        rows.iter()
            .map(|r| r.lobby().0)
            .collect::<BTreeSet<_>>()
            .len()
    )?;
    writeln!(
        report,
        "\n表格统计有效局。新路线的启动要求战团核心已进战斗，同时观测到该路线的成长、操作结算或出战见证。严格列要求启动和完整核心同回合。烈毒搅局的启动表示交易阵容已完成至少一场有胜负的实战。"
    )?;
    writeln!(
        report,
        "\n| 新路线 | 可用 | 选线路 | 核心出战 | 启动 | 齐集 | 启动且齐集 | 最终仍完整 | 启动且齐集夺冠 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    writeln!(
        cases,
        "# {} 实战代表战团\n\n每条路线优先展示启动且完整的战团。战团来自记录回合，排名为全局终局排名。",
        arms[arm].name(),
    )?;
    writeln!(
        raw,
        "seed,seat,route,rank,selected_turn,core_turn,start_turn,full_turn,started_full_turn,final_full,gain"
    )?;
    let turn = |t: Option<u32>| t.map(|v| v.to_string()).unwrap_or_default();
    for route in catalog::Route::ALL {
        let spec = route.spec();
        let eligible = rows
            .iter()
            .filter(|r| r.outcomes[arm].problem.is_none() && route.has_eligible_pool(r.lobby()));
        let idx = route.index();
        let mut counts = [0; 7];
        for row in eligible.clone() {
            let g = &row.outcomes[arm];
            let c = &g.catalog.routes[idx];
            counts[0] += usize::from(c.selected.is_some());
            counts[1] += usize::from(c.core.is_some());
            counts[2] += usize::from(c.started.is_some());
            counts[3] += usize::from(c.full.is_some() && route.full_pool(row.lobby()));
            counts[4] += usize::from(c.started_full.is_some() && route.full_pool(row.lobby()));
            counts[5] += usize::from(c.final_full && route.full_pool(row.lobby()));
            counts[6] += usize::from(
                g.placement == 1 && c.started_full.is_some() && route.full_pool(row.lobby()),
            );
            if c.selected.is_some() {
                writeln!(
                    raw,
                    "{},{},{},{},{},{},{},{},{},{},{}",
                    row.seed,
                    row.seat,
                    spec.key,
                    g.placement,
                    turn(c.selected),
                    turn(c.core),
                    turn(c.started),
                    turn(c.full),
                    turn(c.started_full),
                    c.final_full,
                    c.gain
                )?;
            }
        }
        writeln!(
            report,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            spec.name,
            eligible.clone().count(),
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            counts[4],
            counts[5],
            counts[6]
        )?;
        let best = eligible
            .filter_map(|row| {
                let g = &row.outcomes[arm];
                let board = g.catalog.routes[idx].best.as_ref()?;
                Some((row, g, board))
            })
            .min_by_key(|(row, g, board)| {
                (
                    !(board.started && board.full),
                    !board.started,
                    !board.full,
                    g.placement,
                    Reverse(board.attack + board.health),
                    row.seed,
                )
            });
        writeln!(cases, "\n## {}\n", spec.name)?;
        if let Some((row, g, b)) = best {
            writeln!(
                cases,
                "种子 {}，座位 {}，第 {} 名。第 {} 回合、{} 本，英雄 {}。已启动 {}，组件完整 {}。",
                row.seed,
                row.seat,
                g.placement,
                b.turn,
                b.tavern,
                b.hero.unwrap_or("未知"),
                b.started,
                b.full
            )?;
            writeln!(cases, "\n| 随从 | 攻击 | 生命 |\n| --- | ---: | ---: |")?;
            for m in &b.board {
                writeln!(
                    cases,
                    "| {}{} | {} | {} |",
                    m.name,
                    if m.golden { "〈金色〉" } else { "" },
                    m.attack,
                    m.health
                )?;
            }
            serde_json::to_writer_pretty(
                File::create(output.join(format!("v1-{}-{}.json", spec.key, row.seed)))?,
                b,
            )?;
        } else {
            writeln!(cases, "本批种子暂无启动或齐集案例。")?;
        }
    }
    writeln!(
        report,
        "\n## 已有八条路线\n\n选过指至少一次由该流派的专项策略决策。启动与完整沿用各路线的验收规则，范围为满足牌池条件的对局，即使当时由其他策略控制。活化艾泽里特按元素和鱼人同时在场统计启动与完整，单元素局也可能走艾泽里特。"
    )?;
    writeln!(
        report,
        "\n| 版本 | 路线 | 可用 | 全局选过 | 可用局选过 | 启动 | 曾完整 | 选过且曾完整 |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    struct Prior {
        name: &'static str,
        pool: Races,
        selected: fn(&Outcome) -> bool,
        working: fn(&Outcome) -> bool,
        full: fn(&Outcome) -> bool,
    }
    let mut prior = vec![
        Prior {
            name: "活化艾泽里特",
            pool: Races::ELEMENTAL.union(Races::MURLOC),
            selected: |g| g.used_azerite,
            working: |g| g.azerite.working_turn.is_some(),
            full: |g| g.azerite.full_turn.is_some(),
        },
        Prior {
            name: "喝茶流",
            pool: Races::NONE,
            selected: |g| g.used_tea,
            working: |g| g.tea.working_turn.is_some(),
            full: |g| g.tea.four_turn.is_some(),
        },
        Prior {
            name: "野猪人抉择",
            pool: Races::QUILBOAR,
            selected: |g| g.used_quilboar,
            working: |g| g.quilboar.working_turn.is_some(),
            full: |g| g.quilboar.five_turn.is_some(),
        },
        Prior {
            name: "恶魔自伤",
            pool: Races::DEMON,
            selected: |g| g.used_demon,
            working: |g| g.demon.working_turn.is_some(),
            full: |g| g.demon.six_turn.is_some(),
        },
    ];
    prior.extend([
        Prior {
            name: "手速亡灵",
            pool: apm::Route::Undead.required(),
            selected: |g| g.apm.routes[0].selected,
            working: |g| g.apm.routes[0].working.is_some(),
            full: |g| g.apm.routes[0].full.is_some(),
        },
        Prior {
            name: "手速酒馆强化",
            pool: apm::Route::Tavern.required(),
            selected: |g| g.apm.routes[1].selected,
            working: |g| g.apm.routes[1].working.is_some(),
            full: |g| g.apm.routes[1].full.is_some(),
        },
        Prior {
            name: "手速磁力",
            pool: apm::Route::Mech.required(),
            selected: |g| g.apm.routes[2].selected,
            working: |g| g.apm.routes[2].working.is_some(),
            full: |g| g.apm.routes[2].full.is_some(),
        },
        Prior {
            name: "手速神明",
            pool: apm::Route::Deity.required(),
            selected: |g| g.apm.routes[3].selected,
            working: |g| g.apm.routes[3].working.is_some(),
            full: |g| g.apm.routes[3].full.is_some(),
        },
    ]);
    for (i, policy) in arms.iter().enumerate() {
        for r in &prior {
            let games = rows
                .iter()
                .map(|r| &r.outcomes[i])
                .filter(|g| g.problem.is_none());
            let selected_all = games.clone().filter(|g| (r.selected)(g)).count();
            let mut counts = [0; 5];
            for g in games.filter(|g| r.pool.is_none() || g.lobby.contains(r.pool)) {
                let selected = (r.selected)(g);
                let full = (r.full)(g);
                counts[0] += 1;
                counts[1] += usize::from(selected);
                counts[2] += usize::from((r.working)(g));
                counts[3] += usize::from(full);
                counts[4] += usize::from(selected && full);
            }
            writeln!(
                report,
                "| {} | {} | {} | {selected_all} | {} | {} | {} | {} |",
                policy.name(),
                r.name,
                counts[0],
                counts[1],
                counts[2],
                counts[3],
                counts[4]
            )?;
        }
    }
    report.flush()?;
    cases.flush()?;
    raw.flush()
}

fn write_progress_report(
    output: &Path,
    arms: &[Policy],
    rows: &[ResultRow],
    name: &str,
) -> io::Result<()> {
    let mut report = BufWriter::new(File::create(output.join(format!("{name}.md")))?);
    let mut csv = BufWriter::new(File::create(output.join(format!("{name}.csv")))?);
    writeln!(
        report,
        "# 升本、成型与夺冠对照\n\n每个种子同座位对七个原版 Bot。各项比率以有效局为分母。成型指至少一条已实现路线在任意回合达到既有完整条件。同局多条成型只计一局。"
    )?;
    writeln!(
        report,
        "\n| 策略 | 有效局 | T2 到 2 本 | T5 到 3 本 | T8 到 6 本 | T9 到 6 本 | T10 到 6 本 | T11 到 6 本 | 曾到 6 本 | 至少一套成型 | 吃鸡 | 成型且吃鸡 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    writeln!(
        csv,
        "seed,seat,policy,rank,tier_two_turn,tier_three_turn,tier_five_turn,tier_six_turn,last_recruit_turn,formed_routes"
    )?;
    let mut route_counts = std::collections::BTreeMap::<&str, Vec<[usize; 2]>>::new();
    for (i, policy) in arms.iter().enumerate() {
        let mut count = [0; 11];
        let turn = |t: Option<u32>| t.map_or_else(String::new, |v| v.to_string());
        for row in rows {
            let g = &row.outcomes[i];
            if g.problem.is_some() {
                continue;
            }
            let routes: Vec<_> = g.formed_routes().collect();
            let formed = !routes.is_empty();
            let winner = g.placement == 1;
            for &route in &routes {
                let stats = &mut route_counts
                    .entry(route)
                    .or_insert_with(|| vec![[0; 2]; arms.len()])[i];
                stats[0] += 1;
                stats[1] += usize::from(winner);
            }
            for (n, condition) in count.iter_mut().zip([
                true,
                g.tier_two_turn.is_some_and(|t| t <= 2),
                g.tier_three_turn.is_some_and(|t| t <= 5),
                g.tier_six_turn.is_some_and(|t| t <= 8),
                g.tier_six_turn.is_some_and(|t| t <= 9),
                g.tier_six_turn.is_some_and(|t| t <= 10),
                g.tier_six_turn.is_some_and(|t| t <= 11),
                g.tier_six_turn.is_some(),
                formed,
                winner,
                formed && winner,
            ]) {
                *n += usize::from(condition);
            }
            writeln!(
                csv,
                "{},{},{},{},{},{},{},{},{},{}",
                row.seed,
                row.seat,
                policy.name(),
                g.placement,
                turn(g.tier_two_turn),
                turn(g.tier_three_turn),
                turn(g.tier_five_turn),
                turn(g.tier_six_turn),
                g.last_recruit_turn,
                routes.join("|")
            )?;
        }
        write!(report, "| {} | {} |", policy.name(), count[0])?;
        for n in &count[1..] {
            let percent = 100.0 * *n as f64 / count[0].max(1) as f64;
            write!(report, " {n} ({percent:.2}%) |")?;
        }
        writeln!(report)?;
    }
    writeln!(
        report,
        "\n| 路线 | 策略 | 成型 | 成型且吃鸡 |\n| --- | --- | ---: | ---: |"
    )?;
    for (route, counts) in route_counts {
        for (policy, [full, first]) in arms.iter().zip(counts) {
            writeln!(report, "| {route} | {} | {full} | {first} |", policy.name())?;
        }
    }
    report.flush()?;
    csv.flush()
}

fn write_formed_apm_examples(output: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    let mut examples = Vec::new();
    for (index, policy) in arms.iter().enumerate() {
        for route in apm::Route::ALL {
            let case = rows
                .iter()
                .filter_map(|row| {
                    let outcome = &row.outcomes[index];
                    (outcome.problem.is_none()
                        && outcome.placement == 1
                        && outcome.apm.routes[route.index()].final_full)
                        .then(|| outcome.apm.last.as_ref().map(|board| (row, board)))
                        .flatten()
                })
                .max_by_key(|(_, board)| (board.attack + board.health, board.turn));
            if let Some((row, board)) = case {
                examples.push(serde_json::json!({
                    "policy": policy.name(), "route": route.name(), "seed": row.seed,
                    "seat": row.seat, "rank": 1, "warband": board
                }));
            }
        }
    }
    serde_json::to_writer_pretty(
        File::create(output.join("formed-apm-boards.json"))?,
        &examples,
    )?;
    Ok(())
}

fn write_v2_route_comparison(output: &Path, arms: &[Policy], rows: &[ResultRow]) -> io::Result<()> {
    struct RouteEval {
        name: &'static str,
        eligible: fn(Races) -> bool,
        started: fn(&Outcome) -> bool,
        full: fn(&Outcome) -> bool,
    }
    let specialized = [
        RouteEval {
            name: "活化艾泽里特",
            eligible: |l| l.contains(Races::ELEMENTAL.union(Races::MURLOC)),
            started: |g| g.azerite.working_turn.is_some(),
            full: |g| g.azerite.full_turn.is_some(),
        },
        RouteEval {
            name: "喝茶流",
            eligible: |_| true,
            started: |g| g.tea.working_turn.is_some(),
            full: |g| g.tea.four_turn.is_some(),
        },
        RouteEval {
            name: "野猪人抉择",
            eligible: |l| l.contains(Races::QUILBOAR),
            started: |g| g.quilboar.working_turn.is_some(),
            full: |g| g.quilboar.five_turn.is_some(),
        },
        RouteEval {
            name: "恶魔自伤",
            eligible: |l| l.contains(Races::DEMON),
            started: |g| g.demon.working_turn.is_some(),
            full: |g| g.demon.six_turn.is_some(),
        },
    ];
    let mut csv = BufWriter::new(File::create(output.join("v2-route-comparison.csv"))?);
    writeln!(csv, "route,policy,eligible,started,full,full_and_first")?;
    for (i, policy) in arms.iter().enumerate() {
        let count = |eligible: &dyn Fn(Races) -> bool,
                     evaluate: &dyn Fn(&Outcome) -> (bool, bool)| {
            let mut counts = [0; 4];
            for g in rows
                .iter()
                .map(|r| &r.outcomes[i])
                .filter(|g| g.problem.is_none() && eligible(g.lobby))
            {
                let (started, full) = evaluate(g);
                for (n, matches) in
                    counts
                        .iter_mut()
                        .zip([true, started, full, full && g.placement == 1])
                {
                    *n += usize::from(matches);
                }
            }
            counts
        };
        let mut write = |name: &str, [eligible, started, full, first]: [usize; 4]| {
            writeln!(
                csv,
                "{name},{},{eligible},{started},{full},{first}",
                policy.name()
            )
        };
        for s in &specialized {
            write(
                s.name,
                count(&s.eligible, &|g| ((s.started)(g), (s.full)(g))),
            )?;
        }
        for route in apm::Route::ALL {
            write(
                route.name(),
                count(&|l| l.contains(route.required()), &|g| {
                    let p = &g.apm.routes[route.index()];
                    (p.working.is_some(), p.full.is_some())
                }),
            )?;
        }
        for route in catalog::Route::ALL {
            write(
                route.spec().key,
                count(&|l| route.has_eligible_pool(l), &|g| {
                    let p = &g.catalog.routes[route.index()];
                    (
                        p.started.is_some(),
                        p.full.is_some() && route.full_pool(g.lobby),
                    )
                }),
            )?;
        }
    }
    csv.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let focus_arm;
    let (start, count, threads, directory, arms): (_, _, _, _, &[Policy]) = match args.as_slice() {
        [start, count, threads, directory] => (start, count, threads, directory, &[Policy::Original, Policy::Guide]),
        [start, count, threads, directory, mode] if mode == "azerite" => (start, count, threads, directory,
            &[Policy::Original, Policy::Guide, Policy::Azerite]),
        [start, count, threads, directory, mode] if mode == "tea" => (start, count, threads, directory,
            &[Policy::Original, Policy::Guide, Policy::Tea]),
        [start, count, threads, directory, mode] if mode == "combined" => (start, count, threads, directory,
            &[Policy::Original, Policy::Guide, Policy::Azerite, Policy::Tea, Policy::Combined]),
        [start, count, threads, directory, mode] if mode == "cadence" => (start, count, threads, directory,
            &[Policy::Original, Policy::Guide, Policy::Combined, Policy::Cadence]),
        [start, count, threads, directory, mode] if mode == "quilboar" => (start, count, threads, directory,
            &[Policy::Original, Policy::Guide, Policy::Cadence, Policy::Quilboar, Policy::AllBuilds]),
        [start, count, threads, directory, mode] if mode == "demon" => (start, count, threads, directory,
            &[Policy::Original, Policy::AllBuilds, Policy::Demon, Policy::FourBuilds]),
        [start, count, threads, directory, mode] if mode == "four-builds" => (start, count, threads, directory,
            &[Policy::Original, Policy::AllBuilds, Policy::FourBuilds]),
        [start, count, threads, directory, mode] if mode == "eight-builds" => (start, count, threads, directory,
            &[Policy::Original, Policy::FourBuilds, Policy::EightBuilds]),
        [start, count, threads, directory, mode] if mode == "hsreplay-picks" => (start, count, threads, directory,
            &[Policy::EightBuilds, Policy::HsHeroes, Policy::HsTrinkets, Policy::HsBoth]),
        [start, count, threads, directory, mode] if mode == "v1.0" => (start, count, threads, directory,
            &[Policy::HsBoth, Policy::V1]),
        [start, count, threads, directory, mode] if mode == "v1.0-only" => (start, count, threads, directory,
            &[Policy::V1]),
        [start, count, threads, directory, mode] if mode == "pivot" => (start, count, threads, directory,
            &[Policy::V1, Policy::Pivot]),
        [start, count, threads, directory, mode] if mode == "pivot-only" => (start, count, threads, directory,
            &[Policy::Pivot]),
        [start, count, threads, directory, mode] if mode == "undead-apm" => (start, count, threads, directory,
            &[Policy::Pivot, Policy::UndeadApm]),
        [start, count, threads, directory, mode] if mode == "undead-apm-only" => (start, count, threads, directory,
            &[Policy::UndeadApm]),
        [start, count, threads, directory, mode] if mode == "undead-finance" => (start, count, threads, directory,
            &[Policy::UndeadApm, Policy::UndeadFinance]),
        [start, count, threads, directory, mode] if mode == "undead-reborn" => (start, count, threads, directory,
            &[Policy::UndeadFinance, Policy::UndeadReborn]),
        [start, count, threads, directory, mode] if mode == "rush-six" => (start, count, threads, directory,
            &[Policy::UndeadReborn, Policy::RushSix]),
        [start, count, threads, directory, mode] if mode == "hsrl" => (start, count, threads, directory,
            &[Policy::Original, Policy::V2, Policy::V2Placed]),
        [start, count, threads, directory, mode] if mode == "v2-history" => (start, count, threads, directory,
            &[Policy::Original, Policy::HsBoth, Policy::V1, Policy::Pivot, Policy::UndeadApm,
              Policy::UndeadFinance, Policy::UndeadReborn, Policy::RushSix]),
        [start, count, threads, directory, mode] if mode == "v2-dev" => (start, count, threads, directory,
            &[Policy::V1, Policy::UndeadReborn, Policy::V2]),
        [start, count, threads, directory, mode] if mode == "v2" => (start, count, threads, directory,
            &[Policy::V1, Policy::UndeadFinance, Policy::UndeadReborn, Policy::V2]),
        [start, count, threads, directory, mode, name] if mode == "v1.0-focus" => {
            let route = catalog::Route::ALL.into_iter().find(|r| r.spec().key == name)
                .ok_or("unknown catalog route")?;
            focus_arm = [Policy::Focus(route)];
            (start, count, threads, directory, &focus_arm)
        }
        _ => return Err("usage: strategy_experiment <first-seed> <pairs> <workers> <new-output-dir> [azerite|tea|combined|cadence|quilboar|demon|four-builds|eight-builds|hsreplay-picks|v1.0|v1.0-only|v1.0-focus <route>|pivot|pivot-only|undead-apm|undead-apm-only|undead-finance|undead-reborn|rush-six|v2-history|v2-dev|v2]".into()),
    };
    let start = start.parse::<u64>()?;
    let count = count.parse::<usize>()?;
    let threads = threads.parse::<usize>()?.min(count);
    if count == 0 || threads == 0 || start.checked_add(count as u64).is_none() {
        return Err("expected nonzero counts and a seed range within u64".into());
    }
    let output = Path::new(directory);
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(output)?;
    let now = Instant::now();
    let record_picks = arms.contains(&Policy::HsBoth) || arms.iter().any(|p| p.uses_catalog());
    let fixed_pivot_lobby = args.get(4).is_none_or(|mode| !mode.starts_with("v2"))
        && arms.iter().any(|p| {
            matches!(
                p,
                Policy::Pivot
                    | Policy::UndeadApm
                    | Policy::UndeadFinance
                    | Policy::UndeadReborn
                    | Policy::RushSix
            )
        });
    let mut rows = std::thread::scope(|scope| -> io::Result<Vec<ResultRow>> {
        let jobs: Vec<_> = (0..threads)
            .map(|worker| {
                workers::spawn(scope, move || {
                    (worker..count)
                        .step_by(threads)
                        .map(|i| {
                            let seed = start + i as u64;
                            let seat = (seed % 8) as usize;
                            let outcomes = arms
                                .iter()
                                .copied()
                                .map(|p| {
                                    run(seed, seat, p, output, record_picks, fixed_pivot_lobby)
                                })
                                .collect::<io::Result<Vec<_>>>()?;
                            assert!(
                                outcomes
                                    .iter()
                                    .all(|outcome| outcome.lobby == outcomes[0].lobby)
                            );
                            Ok(ResultRow {
                                seed,
                                seat,
                                outcomes,
                            })
                        })
                        .collect::<io::Result<Vec<_>>>()
                })
            })
            .collect();
        let mut rows = Vec::with_capacity(count);
        for job in jobs {
            rows.extend(job.join().expect("strategy worker panicked")?);
        }
        Ok(rows)
    })?;
    rows.sort_unstable_by_key(|row| row.seed);
    write_pairs(output, arms, &rows)?;

    let mut summary = Vec::new();
    writeln!(
        summary,
        "TavernGym {} / rules {}. {}",
        taverngym::VERSION,
        taverngym::SERVER_RULES_VERSION,
        if fixed_pivot_lobby {
            "Fixed lobby types: PIRATE, UNDEAD, MECHANICAL, QUILBOAR, ABERRATION."
        } else {
            "Random lobby types (simulator default, no forced types)."
        },
    )?;
    writeln!(
        summary,
        "{count} matched seeds starting at {start}; {} games; {threads} workers; {:.2}s. Every tested seat faces seven original bots.",
        count * arms.len(),
        now.elapsed().as_secs_f64()
    )?;
    writeln!(
        summary,
        "Shared recruit action cap: {} per seat and turn. Pending choices can finish after it.",
        RulesConfig::hsrl().action_cap(1)
    )?;
    writeln!(
        summary,
        "Tea core: Gatekeeper with Balinda or Dark Puppeteer at recruit end. Working: same core, at least 5 direct casts on Gatekeeper and at least 60 Health gained by other minions during those actions. Guide four: Gatekeeper, Balinda, Hot-Air Surveyor and Brann."
    )?;
    if arms.contains(&Policy::Quilboar) || arms.contains(&Policy::FourBuilds) {
        writeln!(
            summary,
            "Quilboar core: Turbo Hogrider with Gem Rat or Bramble Tunneler at recruit end. Working: core plus >=3 observed Choose One resolutions with Hogrider present and at least two existing Quilboar gaining Blood Gems each time."
        )?;
    }
    if arms.contains(&Policy::FourBuilds) {
        writeln!(
            summary,
            "Demon engine: Ashen Corruptor, Eredar Escapist and Devilish Distractor at recruit end. Working: same core, at least one Cupcake gained during self-damage with Escapist present, and >=2 observed hand-played Cupcakes on Distractor while those three were present, each increasing permanent Tavern stats."
        )?;
    }
    for (index, policy) in arms.iter().enumerate() {
        let games: Vec<_> = rows.iter().map(|row| &row.outcomes[index]).collect();
        summarize_arm(policy.name(), &games, &mut summary)?;
        if record_picks {
            summarize_pick_choices(&games, &mut summary)?;
        }
        if arms.contains(&Policy::Quilboar) || arms.contains(&Policy::FourBuilds) {
            summarize_quilboar(*policy, &games, &mut summary)?;
        }
        if arms.contains(&Policy::FourBuilds) {
            summarize_demon(*policy, &games, &mut summary)?;
        }
        if arms.iter().any(|p| {
            matches!(
                p,
                Policy::Azerite | Policy::Combined | Policy::AllBuilds | Policy::FourBuilds
            )
        }) {
            let label = if *policy == Policy::Original {
                "Original"
            } else {
                policy.name()
            };
            azerite_report::summarize(
                label,
                games
                    .iter()
                    .filter(|g| g.problem.is_none())
                    .map(|g| &g.azerite),
                &mut summary,
            )?;
        }
        if index > 0 {
            summarize_pair(
                &rows,
                0,
                index,
                &format!("{} vs {}", policy.name(), arms[0].name()),
                &mut summary,
            )?;
        }
        if index > 1 {
            summarize_pair(
                &rows,
                1,
                index,
                &format!("{} vs {}", policy.name(), arms[1].name()),
                &mut summary,
            )?;
        }
    }
    if let Some(tea_index) = arms.iter().position(|p| *p == Policy::Tea) {
        for chunk in rows.chunks(count.div_ceil(4)) {
            summarize_pair(
                chunk,
                1,
                tea_index,
                &format!(
                    "Tea vs guide, seeds {}–{}",
                    chunk[0].seed,
                    chunk[chunk.len() - 1].seed
                ),
                &mut summary,
            )?;
        }
        write_tea_showcase(output, arms, &rows)?;
    }
    if let (Some(cadence_index), Some(combined_index)) = (
        arms.iter().position(|p| *p == Policy::Cadence),
        arms.iter().position(|p| *p == Policy::Combined),
    ) {
        summarize_pair(
            &rows,
            combined_index,
            cadence_index,
            "Cadence vs previous combined",
            &mut summary,
        )?;
        for chunk in rows.chunks(count.div_ceil(4)) {
            summarize_pair(
                chunk,
                combined_index,
                cadence_index,
                &format!(
                    "Cadence vs previous combined, seeds {}–{}",
                    chunk[0].seed,
                    chunk[chunk.len() - 1].seed
                ),
                &mut summary,
            )?;
        }
    }
    if let Some(quil_index) = arms.iter().position(|p| *p == Policy::Quilboar) {
        let cadence = arms.iter().position(|p| *p == Policy::Cadence).unwrap();
        let all = arms.iter().position(|p| *p == Policy::AllBuilds).unwrap();
        summarize_pair(
            &rows,
            cadence,
            quil_index,
            "Quilboar specialist vs previous cadence",
            &mut summary,
        )?;
        summarize_pair(
            &rows,
            cadence,
            all,
            "Three routes vs previous cadence",
            &mut summary,
        )?;
        for chunk in rows.chunks(count.div_ceil(4)) {
            summarize_pair(
                chunk,
                cadence,
                all,
                &format!(
                    "Three routes vs previous cadence, seeds {}–{}",
                    chunk[0].seed,
                    chunk[chunk.len() - 1].seed
                ),
                &mut summary,
            )?;
        }
        write_quilboar_showcase(output, arms, &rows)?;
        write_tea_showcase(output, arms, &rows)?;
    }
    if arms
        .iter()
        .any(|p| matches!(p, Policy::Azerite | Policy::AllBuilds | Policy::FourBuilds))
    {
        write_azerite_showcase(output, arms, &rows)?;
    }
    if arms.contains(&Policy::FourBuilds) {
        write_demon_showcase(output, arms, &rows)?;
        write_quilboar_showcase(output, arms, &rows)?;
        write_tea_showcase(output, arms, &rows)?;
        write_route_coverage(output, arms, &rows)?;
    }
    if arms.contains(&Policy::EightBuilds) {
        write_apm_report(output, arms, &rows)?;
    }
    if let Some(arm) = arms.iter().rposition(|p| p.uses_catalog()) {
        write_v1_report(output, arms, arm, &rows)?;
    }
    if fixed_pivot_lobby {
        write_pivot_report(output, arms, &rows)?;
        if arms.len() == 2 {
            for block in rows.chunks(count.div_ceil(4)) {
                summarize_pair(
                    block,
                    0,
                    1,
                    &format!(
                        "Pivot, seeds {}–{}",
                        block[0].seed,
                        block[block.len() - 1].seed
                    ),
                    &mut summary,
                )?;
            }
        }
    }
    if fixed_pivot_lobby
        && arms.iter().any(|p| {
            matches!(
                p,
                Policy::UndeadApm | Policy::UndeadFinance | Policy::UndeadReborn | Policy::RushSix
            )
        })
    {
        write_undead_apm_report(output, arms, &rows)?;
    }
    if arms.contains(&Policy::V2) {
        write_progress_report(output, arms, &rows, "v2-progress")?;
        write_v2_route_comparison(output, arms, &rows)?;
        write_formed_apm_examples(output, arms, &rows)?;
    } else if arms.contains(&Policy::RushSix) {
        write_progress_report(output, arms, &rows, "rush-six")?;
    }
    fs::write(output.join("summary.txt"), &summary)?;
    io::stdout().write_all(&summary)?;
    if rows
        .iter()
        .any(|row| row.outcomes.iter().any(|o| o.problem.is_some()))
    {
        return Err("invalid games; see pairs.csv and saved snapshots".into());
    }
    Ok(())
}
