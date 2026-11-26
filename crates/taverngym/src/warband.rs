//! Post-recruit combat opponents. Source state and live lobby state have separate lifetimes.

use crate::cards::db;
use crate::state::{Item, Player};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub turn: u32,
    pub player: Player,
}

impl Frame {
    pub(crate) fn capture(turn: u32, p: &Player) -> Self {
        let mut player = Player::new(p.id);
        player.board.clone_from(&p.board);
        player.hand.clone_from(&p.hand);
        player.hero = p.hero;
        player.tier = p.tier;
        player.gold = p.gold.max(0);
        player.powers.clone_from(&p.powers);
        player.stats.clone_from(&p.stats);
        player.deity.clone_from(&p.deity);
        player.effects.clone_from(&p.effects);
        player.trinkets.clone_from(&p.trinkets);
        player.secrets.clone_from(&p.secrets);
        player.quests.clone_from(&p.quests);
        player.rewards.clone_from(&p.rewards);
        player.levels.clone_from(&p.levels);
        player.counters.clone_from(&p.counters);
        player.turn_data.clone_from(&p.turn_data);
        for m in &mut player.board {
            m.killed_by = 0;
            m.mods.clear();
        }
        for h in &mut player.hand {
            h.link = 0;
            h.group = 0;
            h.src = 0;
            if let Item::Minion(m) = &mut h.item {
                m.killed_by = 0;
                m.mods.clear();
            }
        }
        Self { turn, player }
    }

    pub fn power(&self) -> f64 {
        let board = &self.player.board;
        let body: i64 = board
            .iter()
            .map(|m| i64::from(m.atk) + i64::from(m.hp))
            .sum();
        (body.max(0) as f64 + 1.).ln() + 0.035 * board.len() as f64
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Track {
    pub seed: u64,
    pub lobby: u16,
    pub source: String,
    pub frames: Vec<Frame>,
}

#[derive(Serialize, Deserialize)]
pub struct Header {
    pub schema: u32,
    pub patch: String,
    pub card_table: u64,
}

pub struct Bank {
    tracks: Vec<Track>,
    by_lobby: BTreeMap<u16, Vec<usize>>,
}

#[derive(Clone, Copy)]
pub enum Mode {
    Continuous,
    Rolling,
    ScriptOnly,
}

impl Mode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "continuous" => Ok(Self::Continuous),
            "rolling" => Ok(Self::Rolling),
            "script-bank" => Ok(Self::ScriptOnly),
            _ => Err(format!("unknown warband mode {value}")),
        }
    }
}

impl Bank {
    pub fn open(path: &Path) -> Result<Self, String> {
        let reader = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
        let mut rows = reader.lines();
        let header = rows
            .next()
            .ok_or("empty warband library")?
            .map_err(|e| e.to_string())?;
        let header: Header = serde_json::from_str(&header).map_err(|e| e.to_string())?;
        if header.schema != 1
            || header.patch != crate::SERVER_RULES_VERSION
            || header.card_table != db().fingerprint
        {
            return Err("warband patch or card table differs from engine".into());
        }
        let mut tracks = Vec::new();
        let mut by_lobby = BTreeMap::<u16, Vec<usize>>::new();
        for line in rows {
            let track: Track = serde_json::from_str(&line.map_err(|e| e.to_string())?)
                .map_err(|e| format!("warband record {}: {e}", tracks.len() + 1))?;
            if track.lobby.count_ones() != 5
                || track.frames.is_empty()
                || track
                    .frames
                    .iter()
                    .enumerate()
                    .any(|(i, f)| f.turn as usize != i + 1)
            {
                return Err(format!("invalid track from seed {}", track.seed));
            }
            by_lobby.entry(track.lobby).or_default().push(tracks.len());
            tracks.push(track);
        }
        for ids in by_lobby.values_mut() {
            ids.sort_by(|&a, &b| track_strength(&tracks[a]).total_cmp(&track_strength(&tracks[b])));
        }
        if tracks.is_empty() {
            return Err("warband library has no tracks".into());
        }
        Ok(Self { tracks, by_lobby })
    }

    pub fn last_turn(&self, lobby: u16, mode: Mode, seats: usize) -> Option<u32> {
        let mut lengths: Vec<_> = self
            .by_lobby
            .get(&lobby)?
            .iter()
            .filter_map(|&id| {
                let t = &self.tracks[id];
                (!matches!(mode, Mode::ScriptOnly) || t.source == "script")
                    .then(|| t.frames.last().unwrap().turn)
            })
            .collect();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        lengths.get(seats - 1).copied()
    }

    pub fn has_frame(&self, id: usize, turn: u32) -> bool {
        self.tracks[id].frames.len() >= turn as usize
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "selection inputs are supplied by the arena"
    )]
    pub fn choose(
        &self,
        lobby: u16,
        turn: u32,
        level: u8,
        mode: Mode,
        previous: Option<usize>,
        drawn: &[usize],
        key: u64,
    ) -> Result<(usize, &Frame, bool), String> {
        let candidates = self
            .by_lobby
            .get(&lobby)
            .ok_or_else(|| format!("missing bank lobby {lobby}"))?;
        let available =
            |id: usize| !matches!(mode, Mode::ScriptOnly) || self.tracks[id].source == "script";
        let frame_at = |id: usize| self.tracks[id].frames.get(turn as usize - 1);
        if let Some(id) = previous
            && !matches!(mode, Mode::Rolling)
            && let Some(frame) = frame_at(id)
        {
            return Ok((id, frame, false));
        }
        let mut eligible: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|&id| available(id) && frame_at(id).is_some())
            .collect();
        if eligible.is_empty() {
            let id = previous
                .or_else(|| candidates.iter().rev().copied().find(|&id| available(id)))
                .ok_or_else(|| format!("no warband for lobby {lobby}"))?;
            return Ok((id, self.tracks[id].frames.last().unwrap(), true));
        }
        if matches!(mode, Mode::Rolling) {
            eligible.sort_by(|&a, &b| {
                frame_at(a)
                    .unwrap()
                    .power()
                    .total_cmp(&frame_at(b).unwrap().power())
            });
        }
        let n = eligible.len();
        let (low, high) = match level {
            1 => (0, n.div_ceil(2)),
            2 => (n / 4, (3 * n).div_ceil(4)),
            3 => (n / 2, n),
            _ => (0, n),
        };
        let band = &eligible[low..high];
        let skip = |id: usize| {
            drawn.contains(&id)
                || drawn
                    .iter()
                    .any(|&other| self.tracks[other].seed == self.tracks[id].seed)
        };
        let start = mix(key) as usize;
        let id = band
            .iter()
            .cycle()
            .skip(start % band.len())
            .take(band.len())
            .copied()
            .find(|&id| !skip(id))
            .or_else(|| {
                eligible
                    .iter()
                    .cycle()
                    .skip(start % n)
                    .take(n)
                    .copied()
                    .find(|&id| !skip(id))
            })
            .or_else(|| {
                eligible
                    .iter()
                    .cycle()
                    .skip(start % n)
                    .take(n)
                    .copied()
                    .find(|id| !drawn.contains(id))
            })
            .ok_or_else(|| format!("no distinct warband for lobby {lobby} turn {turn}"))?;
        Ok((id, frame_at(id).unwrap(), false))
    }
}

fn track_strength(track: &Track) -> f64 {
    // Rank distinct recorded trajectories without scaling or inventing a card.
    let sample = |t: usize, center: f64| {
        track
            .frames
            .get(t - 1)
            .map_or(-1.5, |fr| fr.power() - center)
    };
    0.24 * sample(7, 4.6) + 0.43 * sample(10, 6.0) + 0.33 * sample(13, 7.2)
}

pub fn mix(mut key: u64) -> u64 {
    key = (key ^ (key >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    key = (key ^ (key >> 27)).wrapping_mul(0x94d049bb133111eb);
    key ^ (key >> 31)
}

pub fn formations(p: &Player, races: crate::types::Races) -> Vec<&'static str> {
    use crate::{
        strategy::{apm, azerite, catalog, demon, quilboar, tea},
        types::Races,
    };
    let b = &p.last_board;
    let mut result = Vec::new();
    for (key, ready) in [
        (
            "azerite",
            races.contains(Races::ELEMENTAL.union(Races::MURLOC))
                && azerite::pieces(b) == azerite::FULL,
        ),
        ("tea", tea::pieces(b) & tea::GUIDE_FOUR == tea::GUIDE_FOUR),
        (
            "quilboar",
            races.contains(Races::QUILBOAR) && quilboar::pieces(b) == quilboar::GUIDE_FIVE,
        ),
        (
            "demon",
            races.contains(Races::DEMON) && demon::pieces(b) & demon::SIX == demon::SIX,
        ),
    ] {
        if ready {
            result.push(key);
        }
    }
    result.extend(
        apm::Route::ALL
            .into_iter()
            .filter(|r| races.contains(r.required()) && r.full(b))
            .map(apm::Route::name),
    );
    result.extend(
        catalog::Route::ALL
            .into_iter()
            .filter(|r| r.full_pool(races) && r.full(p, b))
            .map(|r| r.spec().key),
    );
    result
}
