//! Export deterministic demonstrations from the shared V2.0 policy.
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde_json::json;
use taverngym::bc::Episode;
use taverngym::engine::Phase;
use taverngym::obs::{self, OBS_CARDS, OBS_FLOATS};
use taverngym::strategy::TEACHER_POLICY;
use taverngym::{N_ACTIONS, RulesConfig, workers};

const ROW_BYTES: usize = 4 * (OBS_FLOATS + OBS_CARDS) + N_ACTIONS + 6;

fn shard(
    out: &Path,
    first: u64,
    count: usize,
    worker: usize,
    mixed: bool,
    capture: bool,
) -> io::Result<(usize, usize)> {
    let mut data = BufWriter::with_capacity(
        4 << 20,
        File::create(out.join(format!("rows-{worker:03}.bin")))?,
    );
    let mut games = BufWriter::new(File::create(out.join(format!("games-{worker:03}.jsonl")))?);
    let mut valid = 0;
    let mut rows = 0;
    for seed in first..first + count as u64 {
        let mut episode = Episode::new(seed, mixed).map_err(io::Error::other)?;
        let mut buffer = Vec::with_capacity(if capture { 250 * ROW_BYTES } else { 0 });
        let game_result = (|| -> Result<_, String> {
            while !episode.game.is_over() {
                let action = episode.teacher_action()?;
                let mask = episode.game.legal_mask(episode.seat);
                if !mask[action] {
                    return Err(format!("teacher produced an illegal action {action}"));
                }
                let multi = mask.iter().filter(|&&bit| bit).take(2).count() > 1;
                if capture {
                    let view = obs::observe(&episode.game, episode.seat);
                    let flags = u8::from(episode.game.phase == Phase::HeroSelect)
                        | (u8::from(!episode.game.players[episode.seat].pending.is_empty()) << 1)
                        | (u8::from(multi) << 2);
                    buffer.extend(view.floats.into_iter().flat_map(f32::to_le_bytes));
                    buffer.extend(view.cards.into_iter().flat_map(i32::to_le_bytes));
                    buffer.extend(mask.into_iter().map(u8::from));
                    buffer.extend((action as u16).to_le_bytes());
                    buffer.extend((episode.game.turn as u16).to_le_bytes());
                    buffer.extend([0, flags]);
                }
                episode.step(action, multi)?;
            }
            episode.result()
        })();
        match game_result {
            Ok(record) => {
                let rank = episode.game.players[episode.seat].placement;
                for row in buffer.chunks_exact_mut(ROW_BYTES) {
                    row[ROW_BYTES - 2] = rank;
                }
                let count = buffer.len() / ROW_BYTES;
                serde_json::to_writer(
                    &mut games,
                    &json!({"game": record, "offset": rows, "rows": count}),
                )?;
                writeln!(games)?;
                data.write_all(&buffer)?;
                rows += count;
                valid += 1;
            }
            Err(error) => {
                let failure = File::create(out.join(format!("failure-{seed}.json")))?;
                serde_json::to_writer(failure, &episode.game)?;
                return Err(io::Error::other(format!("seed {seed}: {error}")));
            }
        }
    }
    data.flush()?;
    games.flush()?;
    Ok((valid, rows))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (first, count, threads, directory, field, capture) =
        match args.as_slice() {
            [first, count, threads, directory, field] => {
                (first, count, threads, directory, field, true)
            }
            [first, count, threads, directory, field, mode] if mode == "metrics" => {
                (first, count, threads, directory, field, false)
            }
            _ => return Err(
                "usage: bc_collect FIRST_SEED GAMES WORKERS NEW_OUTPUT original|mixed [metrics]"
                    .into(),
            ),
        };
    let first: u64 = first.parse()?;
    let count: usize = count.parse()?;
    let threads = threads.parse::<usize>()?.min(count);
    if threads == 0
        || first.checked_add(count as u64).is_none()
        || !matches!(field.as_str(), "original" | "mixed")
    {
        return Err("invalid seed range, workers or opponent field".into());
    }
    let out = Path::new(directory);
    fs::create_dir(out)?;
    let mixed = field == "mixed";
    let part = count.div_ceil(threads);
    let (games, rows) = std::thread::scope(|scope| -> io::Result<_> {
        let mut jobs = Vec::new();
        for (worker, offset) in (0..count).step_by(part).enumerate() {
            jobs.push(workers::spawn(scope, move || {
                shard(
                    out,
                    first + offset as u64,
                    (count - offset).min(part),
                    worker,
                    mixed,
                    capture,
                )
            }));
        }
        let mut total = (0, 0);
        for job in jobs {
            let (games, rows) = job.join().expect("teacher worker panicked")?;
            total.0 += games;
            total.1 += rows;
        }
        Ok(total)
    })?;
    let manifest = json!({"format": 2, "teacher": TEACHER_POLICY.name(), "rules": taverngym::SERVER_RULES_VERSION,
        "first_seed": first, "games": games, "rows": rows, "field": field, "row_bytes": ROW_BYTES,
        "floats": OBS_FLOATS, "cards": OBS_CARDS, "actions": N_ACTIONS, "turn_action_cap": RulesConfig::HSRL_RECRUIT_ACTIONS, "allow_move": false, "allow_cancel": false,
        "lobby": "five random of ten"});
    serde_json::to_writer_pretty(File::create(out.join("manifest.json"))?, &manifest)?;
    println!("{manifest}");
    Ok(())
}
