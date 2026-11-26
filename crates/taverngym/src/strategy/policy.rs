use super::{
    apm, catalog,
    guide_bot::{self, Build},
    hsreplay_picks, pivot, placement, undead_apm,
};
use crate::{Action, Game};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Original,
    Guide,
    Azerite,
    Tea,
    Combined,
    Cadence,
    Quilboar,
    AllBuilds,
    Demon,
    FourBuilds,
    EightBuilds,
    HsHeroes,
    HsTrinkets,
    HsBoth,
    V1,
    Pivot,
    UndeadApm,
    UndeadFinance,
    UndeadReborn,
    RushSix,
    V2,
    V2Placed,
    Focus(catalog::Route),
}

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Guide => "guide",
            Self::Azerite => "azerite",
            Self::Tea => "tea",
            Self::Combined => "combined",
            Self::Cadence => "cadence",
            Self::Quilboar => "quilboar",
            Self::AllBuilds => "all_builds",
            Self::Demon => "demon",
            Self::FourBuilds => "four_builds",
            Self::EightBuilds => "eight_builds",
            Self::HsHeroes => "hsreplay_heroes",
            Self::HsTrinkets => "hsreplay_trinkets",
            Self::HsBoth => "hsreplay_both",
            Self::V1 => "strategy_v1_0",
            Self::Pivot => "strategy_v1_2",
            Self::UndeadApm => "strategy_v1_3_undead_apm",
            Self::UndeadFinance => "strategy_v1_4_undead_apm",
            Self::UndeadReborn => "strategy_v1_5_undead_apm",
            Self::RushSix => "strategy_v1_6_rush_six",
            Self::V2 => "strategy_v2_0",
            Self::V2Placed => "strategy_v2_placed",
            Self::Focus(_) => "v1_focus",
        }
    }

    fn is_v2(self) -> bool {
        matches!(self, Self::V2 | Self::V2Placed)
    }

    pub fn build(self) -> Build {
        match self {
            Self::Azerite => Build::Azerite,
            Self::Tea => Build::Tea,
            Self::Quilboar => Build::Quilboar,
            Self::Demon => Build::Demon,
            _ => Build::General,
        }
    }

    pub fn uses_cadence(self) -> bool {
        matches!(
            self,
            Self::Cadence | Self::Quilboar | Self::AllBuilds | Self::Demon | Self::FourBuilds
        ) || self.uses_apm()
    }

    pub fn uses_apm(self) -> bool {
        matches!(
            self,
            Self::EightBuilds | Self::HsHeroes | Self::HsTrinkets | Self::HsBoth
        ) || self.uses_catalog()
    }

    pub fn uses_catalog(self) -> bool {
        matches!(
            self,
            Self::V1
                | Self::Pivot
                | Self::UndeadApm
                | Self::UndeadFinance
                | Self::UndeadReborn
                | Self::RushSix
                | Self::V2
                | Self::V2Placed
                | Self::Focus(_)
        )
    }

    pub fn pick_rules(self) -> hsreplay_picks::Rules {
        hsreplay_picks::Rules {
            hero: matches!(self, Self::HsHeroes | Self::HsBoth) || self.uses_catalog(),
            trinket: matches!(self, Self::HsTrinkets | Self::HsBoth) || self.uses_catalog(),
        }
    }
}

pub struct PolicyState {
    policy: Policy,
    build: Build,
    transition: Option<pivot::State>,
    specialize_undead: bool,
    with_apm: bool,
    with_demon: bool,
    focus: Option<catalog::Route>,
}

impl PolicyState {
    pub fn new(policy: Policy) -> Self {
        let build = policy.build();
        let transition = matches!(
            policy,
            Policy::Pivot
                | Policy::UndeadApm
                | Policy::UndeadFinance
                | Policy::UndeadReborn
                | Policy::RushSix
                | Policy::V2
                | Policy::V2Placed
        )
        .then(pivot::State::default);
        let specialize_undead = matches!(
            policy,
            Policy::UndeadApm
                | Policy::UndeadFinance
                | Policy::UndeadReborn
                | Policy::RushSix
                | Policy::V2
                | Policy::V2Placed
        );
        let with_apm = policy.uses_apm();
        let with_demon = policy == Policy::FourBuilds || with_apm;
        let focus = if let Policy::Focus(route) = policy {
            Some(route)
        } else {
            None
        };
        Self {
            policy,
            build,
            transition,
            specialize_undead,
            with_apm,
            with_demon,
            focus,
        }
    }

    pub fn update(&mut self, game: &Game, seat: usize) -> Build {
        let Self {
            policy,
            specialize_undead,
            with_apm,
            with_demon,
            focus,
            ..
        } = *self;
        let player = &game.players[seat];
        let turn = game.turn;
        let mut build = self.build;
        if (matches!(
            policy,
            Policy::Combined | Policy::Cadence | Policy::AllBuilds | Policy::FourBuilds
        ) || with_apm)
            && player.pending.is_empty()
        {
            let prior = build;
            let previous = if let Build::Apm(route) = build {
                Some(route)
            } else {
                None
            };
            build = build.adaptive(
                player,
                turn,
                matches!(policy, Policy::AllBuilds | Policy::FourBuilds) || with_apm,
                with_demon,
            );
            if with_apm
                && let Some((route, score)) = apm::candidate(game, seat, previous)
                && score >= 23
                && score >= build.evidence(player, turn) + 5
            {
                build = Build::Apm(route);
            }
            if policy.uses_catalog() {
                let previous = if let Build::Catalog(route) = prior {
                    Some(route)
                } else {
                    None
                };
                let incumbent = build.v1_incumbent(game, seat);
                if let Some((route, score)) =
                    catalog::candidate(game, seat, previous, focus, policy.is_v2())
                    && (focus.is_some() || score >= 20 && score >= incumbent + 4)
                {
                    build = Build::Catalog(route);
                }
            }
            if let Some(transition) = &mut self.transition {
                build = transition.select(game, seat, prior, build);
            }
            if specialize_undead {
                let specialized = undead_apm::select(game, seat, build);
                if specialized != build {
                    build = specialized;
                    self.transition = Some(pivot::State::default());
                }
            }
        }
        self.build = build;
        build
    }

    pub fn choose(
        &self,
        game: &mut Game,
        seat: usize,
        counters: Option<&mut hsreplay_picks::Decisions>,
    ) -> Option<Action> {
        let Self {
            policy,
            build,
            specialize_undead,
            with_apm,
            with_demon,
            ..
        } = *self;
        let action = guide_bot::choose(
            game,
            seat,
            build,
            guide_bot::Rules {
                cadence: policy.uses_cadence(),
                rush_six: policy == Policy::RushSix,
                demon: with_demon,
                apm: with_apm,
                undead_apm: specialize_undead,
                undead_finance: matches!(
                    policy,
                    Policy::UndeadFinance
                        | Policy::UndeadReborn
                        | Policy::RushSix
                        | Policy::V2
                        | Policy::V2Placed
                ),
                reborn_conversion: matches!(
                    policy,
                    Policy::UndeadReborn | Policy::RushSix | Policy::V2 | Policy::V2Placed
                ),
                v2: policy.is_v2(),
                transition: self.transition.as_ref(),
            },
        )?;
        let mut unrecorded = hsreplay_picks::Decisions::default();
        let action = hsreplay_picks::select(
            game,
            seat,
            build,
            action,
            &self.policy.pick_rules(),
            counters.unwrap_or(&mut unrecorded),
        );
        Some(if policy == Policy::V2Placed && game.cfg.hsrl_actions {
            placement::position(game, seat, action)
        } else {
            action
        })
    }
}
