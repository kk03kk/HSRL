//! Complete-game qualification for the shared HSRL research profile.
use std::{collections::BTreeSet, error::Error, fs, path::Path};
use taverngym::actions::{A_CANCEL, A_CHOOSE, A_END, A_MOVE, A_PLAY};
use taverngym::{Game, RulesConfig, bot, invariants, rng::Rng};

#[derive(Default)]
struct Totals {
    games: usize,
    actions: usize,
    lobbies: BTreeSet<u16>,
    failures: Vec<serde_json::Value>,
}

fn play(g: &mut Game, random: bool, seed: u64) -> Result<usize, String> {
    let mut rng = Rng::new(seed ^ 0x4853524c);
    let random_seat = (seed % 8) as usize;
    let mut actions = 0;
    if g.players.len() != 8 || g.lobby_races.0.count_ones() != 5 {
        return Err("invalid lobby size or tribe count".into());
    }
    while !g.is_over() && actions < 40_000 {
        let before = actions;
        for p in 0..g.players.len() {
            if !g.needs_action(p) {
                continue;
            }
            let mask = g.legal_mask(p);
            if mask[A_CANCEL] || mask[A_MOVE..].iter().any(|&allowed| allowed) {
                return Err(format!("seat {p}: disabled action exposed"));
            }
            let legal: Vec<_> = (0..mask.len()).filter(|&a| mask[a]).collect();
            if legal.is_empty() {
                return Err(format!("seat {p}: needs action with empty mask"));
            }
            if g.players[p].actions_this_turn >= RulesConfig::HSRL_RECRUIT_ACTIONS
                && legal
                    .iter()
                    .any(|&a| a != A_END && !(A_CHOOSE..A_PLAY).contains(&a))
            {
                return Err(format!("seat {p}: action budget bypass"));
            }
            let action = if random && p == random_seat {
                if mask[A_END] && rng.chance(1, 10) {
                    A_END
                } else {
                    legal[rng.below(legal.len())]
                }
            } else {
                bot::choose_action(g, p).encode()
            };
            if !mask.get(action).copied().unwrap_or(false) {
                return Err(format!("seat {p}: chose unmasked action {action}"));
            }
            g.apply_index(p, action)
                .map_err(|err| format!("seat {p}, action {action}: {err}"))?;
            actions += 1;
            if random && p == random_seat {
                let issues = invariants::check(g);
                if !issues.is_empty() {
                    return Err(format!("seat {p}: {}", issues.join(", ")));
                }
            }
        }
        if before == actions {
            return Err("game did not progress".into());
        }
    }

    let mut ranks: Vec<_> = g.players.iter().map(|p| p.placement).collect();
    ranks.sort_unstable();
    if g.stopped_reason.is_some()
        || !g.is_over()
        || g.turn_limit_end
        || g.truncated_combats != 0
        || g.truncated_recruit_contexts != 0
        || ranks != [1, 2, 3, 4, 5, 6, 7, 8]
    {
        return Err(format!(
            "invalid terminal: stopped={:?} turn_limit={} combat_truncations={} recruit_truncations={} ranks={ranks:?}",
            g.stopped_reason, g.turn_limit_end, g.truncated_combats, g.truncated_recruit_contexts
        ));
    }
    Ok(actions)
}

fn panel(
    kind: &str,
    count: usize,
    seed: u64,
    threads: usize,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut totals = Totals::default();
    let random = kind == "one_random_seat";
    let local = std::thread::scope(|scope| {
        let lanes: Vec<_> = (0..threads)
            .map(|lane| {
                std::thread::Builder::new()
                    .stack_size(16 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        let mut totals = Totals::default();
                        for i in (lane..count).step_by(threads) {
                            let seed = seed + i as u64;
                            let mut game = Game::new(seed, RulesConfig::hsrl());
                            match play(&mut game, random, seed) {
                                Ok(actions) => {
                                    totals.games += 1;
                                    totals.actions += actions;
                                    totals.lobbies.insert(game.lobby_races.0);
                                }
                                Err(error) => {
                                    eprintln!("{kind} seed {seed} turn {}: {error}", game.turn);
                                    totals.failures.push(serde_json::json!({
                                "seed": seed, "turn": game.turn, "error": error, "game": game
                            }));
                                }
                            }
                        }
                        totals
                    })
                    .expect("worker starts")
            })
            .collect();
        lanes
            .into_iter()
            .map(|lane| lane.join().expect("simulation worker succeeds"))
            .collect::<Vec<_>>()
    });
    for lane in local {
        totals.games += lane.games;
        totals.actions += lane.actions;
        totals.lobbies.extend(lane.lobbies);
        totals.failures.extend(lane.failures);
    }
    let report = serde_json::json!({
        "panel": kind, "requested": count, "complete": totals.games, "actions": totals.actions,
        "failures": totals.failures.len(), "unique_lobbies": totals.lobbies.len(),
        "seed_start": seed, "seed_end_exclusive": seed + count as u64,
    });
    println!("{report}");
    fs::write(
        output.join(format!("{kind}.json")),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if !totals.failures.is_empty() {
        fs::write(
            output.join(format!("{kind}-failures.json")),
            serde_json::to_vec(&totals.failures)?,
        )?;
        return Err(format!("{kind}: {} failures", totals.failures.len()).into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: hsrl_validate OUTPUT SCRIPTED_GAMES RANDOM_GAMES THREADS".into());
    }
    let output = Path::new(&args[0]);
    let scripted = args[1].parse()?;
    let random = args[2].parse()?;
    let threads = args[3].parse::<usize>()?.max(1);
    let first = panel("scripted", scripted, 7_000_000, threads, output);
    let second = panel("one_random_seat", random, 7_100_000, threads, output);
    first?;
    second
}
