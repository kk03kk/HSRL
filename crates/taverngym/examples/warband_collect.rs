use std::{
    env,
    io::{self, Write},
};
use taverngym::{
    Action, Game, RulesConfig, bot,
    cards::db,
    engine::Phase,
    strategy::{PolicyState, TEACHER_POLICY},
    types::Races,
    warband::{Header, Track},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    let start: u64 = args[1].parse()?;
    let block: usize = args[2].parse()?;
    let blocks: usize = args[3].parse()?;
    let repeats: u64 = args[4].parse()?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    serde_json::to_writer(
        &mut out,
        &Header {
            schema: 1,
            patch: taverngym::SERVER_RULES_VERSION.into(),
            card_table: db().fingerprint,
        },
    )?;
    writeln!(out)?;
    for (index, mask) in (1u16..1 << 10).filter(|m| m.count_ones() == 5).enumerate() {
        if index % blocks != block {
            continue;
        }
        let names: Vec<_> = Races(mask)
            .iter()
            .map(|r| Races::NAMES[r.index().unwrap()].into())
            .collect();
        for run in 0..repeats {
            let seed = start + index as u64 * 128 + run;
            let mut cfg = RulesConfig::hsrl();
            cfg.lobby_types.clone_from(&names);
            let mut game = Game::new(seed, cfg);
            if game.lobby_races.0 != mask {
                return Err("unexpected fixed lobby".into());
            }
            game.warband_capture = Some(Default::default());
            let mut drivers: Vec<_> = (0..8)
                .map(|p| ((run + p as u64) % 4 != 0).then(|| PolicyState::new(TEACHER_POLICY)))
                .collect();
            for _ in 0..10_000 {
                if game.is_over() {
                    break;
                }
                for (p, policy) in drivers.iter_mut().enumerate() {
                    if !game.needs_action(p) {
                        continue;
                    }
                    let tick = game.turn;
                    if let Some(policy) = policy {
                        for _ in 0..256 {
                            if game.turn != tick || !game.needs_action(p) {
                                break;
                            }
                            policy.update(&game, p);
                            let action =
                                policy.choose(&mut game, p, None).ok_or("teacher stalled")?;
                            if !game.legal_mask(p)[action.encode()] {
                                return Err("teacher action masked".into());
                            }
                            game.apply(p, action)?;
                            if action == Action::EndTurn && game.phase == Phase::Recruit {
                                break;
                            }
                        }
                    } else if !bot::play_turn(&mut game, p) {
                        return Err("original bot stalled".into());
                    }
                }
            }
            if !game.is_over()
                || game.turn_limit_end
                || game.truncated_recruit_contexts > 0
                || game.truncated_combats > 0
            {
                return Err(format!("invalid source lobby {seed}").into());
            }
            for frames in game.warband_capture.take().unwrap() {
                if frames.len() < 5 {
                    continue;
                }
                serde_json::to_writer(
                    &mut out,
                    &Track {
                        seed,
                        lobby: mask,
                        source: "script".into(),
                        frames,
                    },
                )?;
                writeln!(out)?;
            }
        }
    }
    Ok(())
}
