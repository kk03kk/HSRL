use std::{path::Path, sync::Arc};
use taverngym::{
    Action,
    bc::{Arena, WarbandPlan},
    invariants,
    warband::{Bank, Mode},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let bank = Arc::new(Bank::open(Path::new(&args[1]))?);
    let seed: u64 = args[2].parse()?;
    let mut audits = 0;
    let mut result_count = 0;
    for (seats, start, mode) in [
        (7, 1, Mode::Continuous),
        (3, 8, Mode::Continuous),
        (7, 1, Mode::Rolling),
    ] {
        if args.get(3).is_some() && !matches!(mode, Mode::Rolling) {
            continue;
        }
        let seeds: Vec<_> = (seed..seed + 64).collect();
        let plan = WarbandPlan {
            bank: bank.clone(),
            seats,
            start,
            mode,
            level: 2,
        };
        let mut arena = Arena::with_plan(seeds, true, 8, Some(plan), false)?;
        let mut last = [0; 64];
        loop {
            let seen = arena.observe();
            if seen.indices.is_empty() {
                break;
            }
            for &i in &seen.indices {
                let i = i as usize;
                let g = arena.snapshot(i).unwrap();
                let player = &g.players[((seed + i as u64) % 8) as usize];
                if player.pending.is_empty()
                    && g.players.iter().all(|p| p.pending.is_empty())
                    && last[i] != g.turn
                {
                    last[i] = g.turn;
                    let errors = invariants::check(g);
                    let species = invariants::pool_species_violations(g);
                    let (held, initial) = invariants::pool_balance(g);
                    if !(errors.is_empty() && species.is_empty() && held == initial) {
                        return Err(format!(
                            "seed {} turn {} pool {held}/{initial}: {errors:?}; {species:?}",
                            seed + i as u64,
                            g.turn
                        )
                        .into());
                    }
                    audits += 1;
                }
            }
            let acts: Vec<_> = arena
                .teacher_targets()?
                .into_iter()
                .map(|a| a[0] as usize)
                .collect();
            if acts.iter().any(|&i| {
                matches!(
                    Action::decode(i),
                    Some(Action::Move { .. } | Action::Cancel)
                )
            }) {
                return Err("forbidden action".into());
            }
            arena.step(&acts)?;
        }
        result_count += arena.results()?.len();
    }
    println!("{{\"complete_games\":{result_count},\"pool_and_structural_checks\":{audits}}}");
    Ok(())
}
