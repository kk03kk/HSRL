//! Matched complete lobbies for imitation learning. Each seat finishes its recruit phase
//! in seating order, as in strategy_experiment, before advancing to the next seat.

use crate::engine::Phase;
use crate::obs::{self, OBS_CARDS, OBS_FLOATS};
use crate::strategy::{PolicyState, TEACHER_POLICY, hsreplay_picks::Decisions};
use crate::warband::{Bank, Mode, Track, mix};
use crate::{Action, Game, N_ACTIONS, RulesConfig, bot, workers};
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct WarbandPlan {
    pub bank: Arc<Bank>,
    pub seats: usize,
    pub start: u32,
    pub level: u8,
    pub mode: Mode,
}

#[derive(Default, Serialize)]
struct WarbandCounts {
    frames: usize,
    splices: usize,
    repeats: usize,
}

#[derive(Serialize)]
pub struct Round {
    turn: u32,
    tier: u8,
    attack: i64,
    health: i64,
}

#[derive(Serialize)]
pub struct ResultRow {
    seed: u64,
    seat: usize,
    rank: u8,
    lobby: u16,
    actions: u32,
    decisions: u32,
    budget_turns: u32,
    rounds: Vec<Round>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warband: Option<WarbandCounts>,
}

pub struct Episode {
    pub game: Game,
    pub seat: usize,
    seed: u64,
    next: usize,
    actions: u32,
    decisions: u32,
    budget_turns: u32,
    rounds: Vec<Round>,
    teacher: PolicyState,
    opponents: [Option<PolicyState>; 8],
    picks: Decisions,
    warband: Option<WarbandPlan>,
    warband_tracks: [Option<usize>; 8],
    warband_counts: WarbandCounts,
}

impl Episode {
    pub fn new(seed: u64, mixed: bool) -> Result<Self, String> {
        Self::with_plan(seed, mixed, None, false)
    }

    fn with_plan(
        seed: u64,
        mixed: bool,
        warband: Option<WarbandPlan>,
        capture: bool,
    ) -> Result<Self, String> {
        let seat = (seed % 8) as usize;
        let opponents = std::array::from_fn(|i| {
            let offset = (i + 8 - seat) % 8;
            let first = (seed / 8 % 7) as usize + 1;
            let second = first % 7 + 1;
            (mixed && (offset == first || offset == second))
                .then(|| PolicyState::new(TEACHER_POLICY))
        });
        let mut game = Game::new(seed, RulesConfig::hsrl());
        if capture {
            game.warband_capture = Some(Default::default());
        }
        if let Some(plan) = &warband {
            if !(1..=7).contains(&plan.seats) || plan.start < 1 || plan.level > 3 {
                return Err("invalid warband seats, first turn or difficulty".into());
            }
            game.warband_start = plan.start;
            let mut order: Vec<_> = (0..8).filter(|&i| i != seat).collect();
            // Hybrid preserves first preference for its two V2.0 script seats.
            order.sort_by_key(|&i| {
                (
                    opponents[i].is_some(),
                    mix(seed ^ ((i as u64 + 1) * 65_537)),
                )
            });
            for &p in order.iter().take(plan.seats) {
                game.warband_seats |= 1 << p;
            }
        }
        let mut episode = Self {
            game,
            seat,
            seed,
            next: 0,
            actions: 0,
            decisions: 0,
            budget_turns: 0,
            rounds: Vec::new(),
            teacher: PolicyState::new(TEACHER_POLICY),
            opponents,
            picks: Decisions::default(),
            warband,
            warband_tracks: [None; 8],
            warband_counts: WarbandCounts::default(),
        };
        episode.advance()?;
        Ok(episode)
    }

    fn fail(&self, reason: impl std::fmt::Display) -> String {
        format!(
            "seed {} seat {} turn {}: {reason}",
            self.seed, self.seat, self.game.turn
        )
    }

    fn advance(&mut self) -> Result<(), String> {
        for _ in 0..10_000 {
            if self.game.is_over() {
                return Ok(());
            }
            if self.game.stopped_reason.is_some() {
                return Err(self.fail("simulator guard stopped the game"));
            }
            if self.release_combat_warbands()? {
                continue;
            }
            let p = self.next;
            if !self.game.needs_action(p)
                || (self.game.phase == Phase::Recruit && self.game.warband_seat(p))
            {
                self.next = (p + 1) % 8;
            } else if p == self.seat {
                return Ok(());
            } else if let Some(policy) = &mut self.opponents[p] {
                let turn = self.game.turn;
                policy.update(&self.game, p);
                let action = policy
                    .choose(&mut self.game, p, Some(&mut self.picks))
                    .ok_or_else(|| {
                        self.fail(format!("teacher opponent at seat {p} has no action"))
                    })?;
                if !self.game.legal_mask(p)[action.encode()] {
                    return Err(self.fail(format!(
                        "teacher opponent seat {p} chose masked action {action:?}"
                    )));
                }
                self.game
                    .apply(p, action)
                    .map_err(|e| self.fail(format!("opponent seat {p}: {e}")))?;
                if action == Action::EndTurn || turn != self.game.turn || !self.game.needs_action(p)
                {
                    self.next = (p + 1) % 8;
                }
            } else {
                if !bot::play_turn(&mut self.game, p) {
                    return Err(self.fail(format!("original opponent at seat {p} did not advance")));
                }
                self.next = (p + 1) % 8;
            }
        }
        Err(self.fail("arena driver exceeded its guard"))
    }

    fn release_combat_warbands(&mut self) -> Result<bool, String> {
        let Some(plan) = &self.warband else {
            return Ok(false);
        };
        let game = &mut self.game;
        if game.phase != Phase::Recruit
            || game.turn < plan.start
            || game
                .players
                .iter()
                .any(|p| p.alive && !game.warband_seat(p.id) && !p.ended_turn)
        {
            return Ok(false);
        }
        let turn = game.turn;
        let mut drawn = Vec::with_capacity(plan.seats);
        for p in 0..8 {
            if !game.players[p].alive || !game.warband_seat(p) || game.players[p].ended_turn {
                continue;
            }
            let previous = self.warband_tracks[p];
            let key = self.seed ^ ((p as u64 + 1) << 32) ^ (u64::from(turn) << 16);
            let (id, frame, repeat) = plan.bank.choose(
                game.lobby_races.0,
                turn,
                plan.level,
                plan.mode,
                previous,
                &drawn,
                key,
            )?;
            game.install_warband(p, frame)
                .map_err(|e| format!("seed {} turn {turn} bank seat {p}: {e}", self.seed))?;
            drawn.push(id);
            self.warband_tracks[p] = Some(id);
            self.warband_counts.frames += 1;
            self.warband_counts.splices += usize::from(previous.is_some() && previous != Some(id));
            self.warband_counts.repeats += usize::from(repeat);
        }
        if drawn.is_empty() {
            return Ok(false);
        }
        game.fight_when_all_ended();
        if game.turn == turn && !game.is_over() {
            return Err(self.fail("external opponents did not release combat"));
        }
        Ok(true)
    }

    pub fn teacher_action(&mut self) -> Result<usize, String> {
        self.teacher.update(&self.game, self.seat);
        self.teacher
            .choose(&mut self.game, self.seat, Some(&mut self.picks))
            .map(Action::encode)
            .ok_or_else(|| self.fail("V2.0 teacher has no action"))
    }

    fn teacher_target(&mut self) -> Result<[u16; 3], String> {
        // Looking up the label must not alter subsequent game randomness.
        let rng = self.game.rng.clone();
        let action = self.teacher_action();
        self.game.rng = rng;
        let action = action?;
        let mask = self.game.legal_mask(self.seat);
        if !mask[action] {
            return Err(self.fail(format!("illegal teacher label {action}")));
        }
        let flags = u16::from(self.game.phase == Phase::HeroSelect)
            | (u16::from(!self.game.players[self.seat].pending.is_empty()) << 1)
            | (u16::from(mask.iter().filter(|&&bit| bit).take(2).count() > 1) << 2);
        Ok([action as u16, self.game.turn as u16, flags])
    }

    pub fn step(&mut self, action: usize, multi_choice: bool) -> Result<(), String> {
        let turn = self.game.turn;
        let p = &self.game.players[self.seat];
        let tier = p.tier;
        let budget_end =
            p.pending.is_empty() && p.actions_this_turn >= self.game.action_cap_of(self.seat);
        let end =
            self.game.phase == Phase::Recruit && Action::decode(action) == Some(Action::EndTurn);
        self.game
            .apply_index(self.seat, action)
            .map_err(|e| self.fail(e))?;
        self.actions += 1;
        self.decisions += u32::from(multi_choice);
        if end {
            let board = &self.game.players[self.seat].last_board;
            self.rounds.push(Round {
                turn,
                tier,
                attack: board.iter().map(|m| i64::from(m.atk)).sum(),
                health: board.iter().map(|m| i64::from(m.hp)).sum(),
            });
            self.budget_turns += u32::from(budget_end);
        }
        // As in strategy_experiment, the next seat acts once the subject's turn is over.
        if end || turn != self.game.turn || !self.game.needs_action(self.seat) {
            self.next = (self.seat + 1) % 8;
        }
        self.advance()
    }

    pub fn result(&mut self) -> Result<ResultRow, String> {
        let mut placements: Vec<_> = self.game.players.iter().map(|p| p.placement).collect();
        placements.sort_unstable();
        if !self.game.is_over()
            || self.game.turn_limit_end
            || self.game.stopped_reason.is_some()
            || self.game.truncated_combats != 0
            || self.game.truncated_recruit_contexts != 0
            || placements != [1, 2, 3, 4, 5, 6, 7, 8]
        {
            return Err(self.fail(format!("invalid terminal game; turn_limit={} combat_truncations={} recruit_truncations={} placements={placements:?}",
                self.game.turn_limit_end, self.game.truncated_combats, self.game.truncated_recruit_contexts)));
        }
        Ok(ResultRow {
            seed: self.seed,
            seat: self.seat,
            rank: self.game.players[self.seat].placement,
            lobby: self.game.lobby_races.0,
            actions: self.actions,
            decisions: self.decisions,
            budget_turns: self.budget_turns,
            rounds: std::mem::take(&mut self.rounds),
            warband: self
                .warband
                .as_ref()
                .map(|_| std::mem::take(&mut self.warband_counts)),
        })
    }
}

pub struct BatchObservation {
    pub indices: Vec<u32>,
    pub floats: Vec<f32>,
    pub cards: Vec<i32>,
    pub legal: Vec<u8>,
}

pub struct Arena {
    episodes: Vec<Episode>,
    active: Vec<usize>,
    threads: usize,
}

impl Arena {
    pub fn new(seeds: Vec<u64>, mixed: bool, threads: usize) -> Result<Self, String> {
        Self::with_plan(seeds, mixed, threads, None, false)
    }

    pub fn with_plan(
        seeds: Vec<u64>,
        mixed: bool,
        threads: usize,
        plan: Option<WarbandPlan>,
        capture: bool,
    ) -> Result<Self, String> {
        let episodes = seeds
            .into_iter()
            .map(|seed| Episode::with_plan(seed, mixed, plan.clone(), capture))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            active: (0..episodes.len()).collect(),
            episodes,
            threads: threads.max(1),
        })
    }

    pub fn observe(&self) -> BatchObservation {
        let n = self.active.len();
        let mut batch = BatchObservation {
            indices: Vec::with_capacity(n),
            floats: Vec::with_capacity(n * OBS_FLOATS),
            cards: Vec::with_capacity(n * OBS_CARDS),
            legal: Vec::with_capacity(n * N_ACTIONS),
        };
        for &i in &self.active {
            let e = &self.episodes[i];
            let seen = obs::observe(&e.game, e.seat);
            batch.indices.push(i as u32);
            batch.floats.extend(seen.floats);
            batch.cards.extend(seen.cards);
            batch
                .legal
                .extend(e.game.legal_mask(e.seat).into_iter().map(u8::from));
        }
        batch
    }

    pub fn step(&mut self, actions: &[usize]) -> Result<(), String> {
        if actions.len() != self.active.len() {
            return Err(format!(
                "expected {} actions, received {}",
                self.active.len(),
                actions.len()
            ));
        }
        let mut selected: Vec<_> = self
            .episodes
            .iter_mut()
            .filter(|e| !e.game.is_over())
            .zip(actions)
            .collect();
        let chunk = selected.len().div_ceil(self.threads).max(1);
        std::thread::scope(|scope| {
            let jobs: Vec<_> = selected
                .chunks_mut(chunk)
                .map(|pairs| {
                    workers::spawn(scope, || {
                        for (e, action) in pairs {
                            let action = **action;
                            let legal = e.game.legal_mask(e.seat);
                            if !legal.get(action).copied().unwrap_or(false) {
                                return Err(e.fail(format!("illegal action {action}")));
                            }
                            e.step(action, legal.iter().filter(|&&bit| bit).take(2).count() > 1)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for job in jobs {
                job.join().expect("arena worker panicked")?;
            }
            Ok::<_, String>(())
        })?;
        self.active.retain(|&i| !self.episodes[i].game.is_over());
        Ok(())
    }

    /// Advance each teacher's own policy history one decision along the student's game.
    /// Call once for each observation before stepping the arena.
    pub fn teacher_targets(&mut self) -> Result<Vec<[u16; 3]>, String> {
        let mut active: Vec<_> = self
            .episodes
            .iter_mut()
            .filter(|e| !e.game.is_over())
            .collect();
        let chunk = active.len().div_ceil(self.threads).max(1);
        std::thread::scope(|scope| {
            let jobs: Vec<_> = active
                .chunks_mut(chunk)
                .map(|episodes| {
                    workers::spawn(scope, || {
                        episodes
                            .iter_mut()
                            .map(|e| e.teacher_target())
                            .collect::<Result<Vec<_>, _>>()
                    })
                })
                .collect();
            let mut targets = Vec::with_capacity(self.active.len());
            for job in jobs {
                targets.extend(job.join().expect("teacher worker panicked")?);
            }
            Ok(targets)
        })
    }

    pub fn results(&mut self) -> Result<Vec<ResultRow>, String> {
        self.episodes.iter_mut().map(Episode::result).collect()
    }

    pub fn take_tracks(&mut self) -> Result<Vec<Track>, String> {
        self.episodes
            .iter_mut()
            .map(|e| {
                if !e.game.is_over() {
                    return Err("cannot export incomplete warband game".into());
                }
                let mut captures = e.game.warband_capture.take().ok_or("capture unavailable")?;
                Ok(Track {
                    seed: e.seed,
                    lobby: e.game.lobby_races.0,
                    source: "ppo".into(),
                    frames: std::mem::take(&mut captures[e.seat]),
                })
            })
            .collect()
    }

    pub fn formations(&self) -> Vec<Vec<&'static str>> {
        self.episodes
            .iter()
            .map(|e| crate::warband::formations(&e.game.players[e.seat], e.game.lobby_races))
            .collect()
    }

    pub fn snapshot(&self, index: usize) -> Option<&Game> {
        self.episodes.get(index).map(|e| &e.game)
    }
}
