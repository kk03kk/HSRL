use crate::Game;
use crate::cards::{CardId, db};
use crate::state::{Minion, Player};
use crate::types::{Kw, Races};
use std::sync::LazyLock;

pub const BRANN: &str = "BG_LOE_077";
pub const TITUS: &str = "BG25_354";
pub const DRAKKARI: &str = "BG26_ICC_901";
pub const BALINDA: &str = "BG35_883";
pub const KELP: &str = "BG36_701";
pub const GRIFFIN: &str = "BG36_204";
pub const BIRD: &str = "BG36_208";
pub const APE: &str = "BG26_802";
pub const RAT: &str = "BG35_604";
pub const DISTRACTOR: &str = "BG36_762";
pub const FELBOAR: &str = "BG28_633";
pub const EYE: &str = "BG22_403";
pub const APPRENTICE: &str = "BG33_890t";
pub const SAPPER: &str = "BG36_103";
pub const SALVAGER: &str = "BG36_700";
pub const POISON: &str = "BG33_318";
pub const POET: &str = "BG29_813";
pub const BROODMOTHER: &str = "BG36_849";
pub const SURVEYOR: &str = "BG30_121";
pub const GEOMANT: &str = "BG36_370";
pub const REFINDER: &str = "BG33_885";
pub const DRONE: &str = "BG26_152";
pub const CZAR: &str = "BG28_741";
pub const CYCLONE: &str = "BGS_123";
pub const CHRONO: &str = "BG36_242";
pub const FUNGUS: &str = "BG33_891";
pub const HUNGRY: &str = "BG27_556";
pub const LIONFISH: &str = "BG36_201";
pub const SHARK: &str = "BG36_206";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Route {
    Keywords,
    Lobster,
    Behemoth,
    UndeadAttack,
    DemonShop,
    Tempest,
    Beetle,
    DemonDeath,
    DeityDeath,
    Magicfin,
    MechSpell,
    Ultraviolet,
    Poison,
    PirateGold,
    Barrage,
    Hope,
    Knight,
    HandGems,
    Sludge,
    DragonSpell,
    Volumizer,
    Tidecaller,
    DragonCry,
    BeastSummon,
    Shield,
}

use Route::*;

impl Route {
    pub fn is_bridge(self) -> bool {
        matches!(
            self,
            Self::Ultraviolet
                | Self::Hope
                | Self::Knight
                | Self::HandGems
                | Self::Sludge
                | Self::DragonSpell
                | Self::Volumizer
                | Self::Tidecaller
                | Self::DragonCry
                | Self::BeastSummon
                | Self::Shield
        )
    }

    pub const ALL: [Self; 25] = [
        Keywords,
        Lobster,
        Behemoth,
        UndeadAttack,
        DemonShop,
        Tempest,
        Beetle,
        DemonDeath,
        DeityDeath,
        Magicfin,
        MechSpell,
        Ultraviolet,
        Poison,
        PirateGold,
        Barrage,
        Hope,
        Knight,
        HandGems,
        Sludge,
        DragonSpell,
        Volumizer,
        Tidecaller,
        DragonCry,
        BeastSummon,
        Shield,
    ];

    pub fn index(self) -> usize {
        self as usize
    }
    pub fn spec(self) -> &'static Spec {
        &CATALOG[self as usize]
    }
    pub fn race(self) -> Races {
        self.spec().race
    }
    pub fn has_eligible_pool(self, lobby: Races) -> bool {
        lobby.contains(self.race())
    }

    pub fn full_pool(self, lobby: Races) -> bool {
        let extra = match self {
            Tempest | Ultraviolet => Races::MURLOC,
            _ => Races::NONE,
        };
        lobby.contains(self.race().union(extra))
    }

    pub fn core(self, p: &Player, board: &[Minion]) -> bool {
        self.spec()
            .core
            .iter()
            .all(|group| group.iter().any(|&card| has(board, card)))
            && match self {
                Poison => board.iter().filter(|m| tech(m)).count() >= 2,
                Hope | Shield => board.iter().filter(|m| m.has(Kw::DIVINE_SHIELD)).count() >= 2,
                DeityDeath => p.deity.is_some(),
                _ => true,
            }
    }

    pub fn full(self, p: &Player, board: &[Minion]) -> bool {
        self.core(p, board)
            && self.spec().full.iter().all(|&card| has(board, card))
            && (self != Shield
                || board
                    .iter()
                    .filter(|m| is(m, CZAR))
                    .map(|m| if m.golden { 2 } else { 1 })
                    .sum::<u32>()
                    >= 2)
    }
}

pub struct Spec {
    pub name: &'static str,
    pub key: &'static str,
    pub race: Races,
    pub anchor: CardId,
    pub core: Vec<Vec<CardId>>,
    pub full: Vec<CardId>,
    pub support: Vec<CardId>,
    pub early: u32,
}

impl Spec {
    pub fn essential(&self, id: CardId) -> bool {
        id == self.anchor
            || self.full.contains(&id)
            || self
                .core
                .iter()
                .any(|group| group.len() == 1 && group.contains(&id))
    }
}

macro_rules! spec {
    ($name:literal, $key:literal, $race:ident, $early:literal, $anchor:expr,
        [$( [$($core:expr),+ $(,)?] ),+ $(,)?], [$($full:expr),* $(,)?], [$($support:expr),* $(,)?]) => {
        Spec { name: $name, key:$key, race: Races::$race, anchor: db().id($anchor), early:$early,
            core: vec![$(vec![$(db().id($core)),+]),+], full: vec![$(db().id($full)),*],
            support: vec![$(db().id($support)),*] }
    };
}

static CATALOG: LazyLock<[Spec; 25]> = LazyLock::new(|| {
    [
        spec!(
            "鱼人 - 关键词",
            "keywords",
            MURLOC,
            7,
            "BG31_148",
            [["BG31_148"], [SALVAGER, KELP], [BRANN, POISON]],
            [SALVAGER, POISON, KELP, BRANN],
            [EYE, "BG31_149", "BG32_336", "BGS_020"]
        ),
        spec!(
            "野兽 - 美味龙虾",
            "lobster",
            BEAST,
            5,
            "BG36_202",
            [["BG36_202"], [GRIFFIN, BIRD, "BG36_210"]],
            [TITUS, GRIFFIN, BIRD],
            [LIONFISH, SHARK, "BG36_210", "BG25_016"]
        ),
        spec!(
            "野兽 - 深潜巨兽",
            "behemoth",
            BEAST,
            7,
            "BG35_602",
            [["BG35_602"], [RAT, "BG36_210", "BG36_209"]],
            [RAT, APE, GRIFFIN],
            ["BG36_209", "BG36_210", TITUS, "BG31_801", "BG31_803"]
        ),
        spec!(
            "亡灵 - 攻击力成长",
            "undead_attack",
            UNDEAD,
            7,
            "BG32_324",
            [["BG32_324"], ["BG25_010", "BG28_309", "BG25_009"]],
            ["BG25_010", "BG28_309", "BG36_511", "BG36_515"],
            [
                TITUS, "BG25_009", "BG31_835", "BG32_880", "BG36_514", "BG25_001", "BG25_022"
            ]
        ),
        spec!(
            "恶魔 - 酒馆强化",
            "demon_shop",
            DEMON,
            7,
            DISTRACTOR,
            [[DISTRACTOR], [FELBOAR, "BG35_155", "BG36_731", "BG21_004"]],
            [FELBOAR, BALINDA, "BG36_731"],
            [
                "BG21_004", "BG26_174", SURVEYOR, "BG32_873", "BG35_155", BRANN, "BG36_621"
            ]
        ),
        spec!(
            "元素 - 无羁雷暴",
            "tempest",
            ELEMENTAL,
            8,
            "BG36_352",
            [
                ["BG36_352"],
                ["BG34_858", "BG28_707", "BG34_500"],
                [CYCLONE, BRANN]
            ],
            ["BG34_858", KELP, BRANN, CYCLONE],
            ["BG28_707", "BG34_500", "BGS_115", "BG31_815", "BG36_181"]
        ),
        spec!(
            "野兽 - 甲虫",
            "beetle",
            BEAST,
            6,
            "BG31_809",
            [["BG31_809", "BG36_209"]],
            ["BG31_809", "BG36_209", APE, GRIFFIN],
            [
                "BG31_801", "BG31_803", "BG35_602", RAT, BIRD, LIONFISH, SHARK, TITUS
            ]
        ),
        spec!(
            "恶魔 - 亡语/酒馆强化",
            "demon_death",
            DEMON,
            6,
            "BG36_362",
            [["BG36_362"], ["BG36_731", "BG36_503", FELBOAR]],
            [TITUS, "BG36_731"],
            ["BG35_155", BALINDA, "BG36_503", FELBOAR, "BG36_621"]
        ),
        spec!(
            "畸变怪 - 亡语/法术",
            "deity_death",
            ABERRATION,
            7,
            "BG36_318",
            [["BG36_318"], [TITUS, BIRD]],
            [TITUS, "BG36_101", SAPPER],
            [
                GRIFFIN, BIRD, "BG25_016", "BG33_823", "BG33_825", "BG36_109", "BG36_115",
                "BG36_104"
            ]
        ),
        spec!(
            "鱼人 - 魔鳍",
            "magicfin",
            MURLOC,
            8,
            FUNGUS,
            [[FUNGUS], [EYE, KELP], [BRANN, EYE]],
            [EYE, BRANN, DRAKKARI],
            [APPRENTICE, KELP, "BG32_821", "BG33_830", "BG34_633"]
        ),
        spec!(
            "机械 - 磁力/法术",
            "mech_spell",
            MECH,
            6,
            "BG36_853",
            [["BG36_853"], [BALINDA, DRONE, "BG36_854"]],
            [BALINDA, DRONE],
            [
                DRAKKARI, SURVEYOR, GEOMANT, "BG36_854", "BG36_764", "BG36_506"
            ]
        ),
        spec!(
            "元素 - 紫外线",
            "ultraviolet",
            ELEMENTAL,
            8,
            "BG31_810",
            [["BG31_810"], [BRANN, CYCLONE]],
            [BRANN, KELP, CYCLONE],
            ["BG34_858", "BG36_352", "BGS_126", "BGS_115", "BG31_815"]
        ),
        spec!(
            "鱼人 - 烈毒搅局",
            "poison",
            MURLOC,
            9,
            POISON,
            [[POISON, "BG23_318", "BGS_131"]],
            ["BG23_318", POISON, SALVAGER],
            [HUNGRY, "BG34_604", "BG36_243", "BGS_131"]
        ),
        spec!(
            "海盗 - 手速金色",
            "pirate_gold",
            PIRATE,
            8,
            "BG36_344",
            [
                ["BG33_823"],
                ["BG33_825", "BG36_523"],
                ["BG36_344", "BG36_523"]
            ],
            ["BG36_344", "BG36_523", "BG33_825"],
            [
                BRANN, "BG33_821", "BG36_520", "BG32_231", "BG26_817", "BG36_369", "BG36_524"
            ]
        ),
        spec!(
            "野猪人 - 宝石弹幕",
            "barrage",
            QUILBOAR,
            6,
            "BG34_682",
            [["BG34_682"], ["BG34_684", FELBOAR]],
            ["BG34_684", FELBOAR],
            [TITUS, DRAKKARI, REFINDER, "BG23_017", "BG31_326", SURVEYOR]
        ),
        spec!(
            "龙 - 希望信使",
            "hope",
            DRAGON,
            7,
            "BG36_364",
            [["BG36_364"]],
            [POET, BROODMOTHER, "BGS_040"],
            ["BG36_241", "BGS_034", "BG36_245", "BG31_149"]
        ),
        spec!(
            "亡灵 - 永恒骑士",
            "knight",
            UNDEAD,
            3,
            "BG25_008",
            [["BG25_008"], ["BG25_009", "BG34_403"]],
            ["BG25_009", "BG34_403"],
            [
                "BG28_309", "BG36_511", "BG36_515", "BG25_010", "BG25_001", TITUS
            ]
        ),
        spec!(
            "野猪人 - 手牌宝石",
            "hand_gems",
            QUILBOAR,
            7,
            "BG28_583",
            [
                ["BG28_583"],
                [SURVEYOR, GEOMANT],
                [REFINDER, "BG23_017", "BG31_326"]
            ],
            [SURVEYOR, REFINDER, GEOMANT],
            [
                "BG23_017", "BG26_817", "BG28_582", BALINDA, "BG31_326", "BG34_684"
            ]
        ),
        spec!(
            "畸变怪 - 腐泥",
            "sludge",
            ABERRATION,
            6,
            "BG36_115",
            [["BG36_115"], ["BG36_320", "BG36_300", "BG36_312"]],
            [DRAKKARI, "BG36_320", "BG36_300"],
            ["BG32_821", "BG36_104", SAPPER, "BG36_114", "BG36_312"]
        ),
        spec!(
            "龙 - 法术强度",
            "dragon_spell",
            DRAGON,
            6,
            "BG36_241",
            [["BG36_241"], [CHRONO, "BG32_821", "BG36_245"]],
            [POET, CHRONO, "BG36_243", "BG36_245", BRANN],
            ["BG32_821", "BG36_240", BROODMOTHER, "BG34_633"]
        ),
        spec!(
            "机械 - 扩音机",
            "volumizer",
            MECH,
            6,
            DRONE,
            [[DRONE], ["BG34_171", "BG36_366", "BG36_367", "BG34_170"]],
            [DRAKKARI, "BG34_171", "BG36_366"],
            ["BG34_170", "BG36_367", KELP, "BG36_851", BRANN, "BG36_506"]
        ),
        spec!(
            "鱼人 - 招潮者",
            "tidecaller",
            MURLOC,
            7,
            "BG36_704",
            [["BG36_704"], [BRANN, BALINDA, FUNGUS]],
            [BRANN, BALINDA, FUNGUS, HUNGRY, "BG26_354"],
            [POISON, "BGS_020", "BG32_336", "BG31_148"]
        ),
        spec!(
            "龙 - 战吼",
            "dragon_cry",
            DRAGON,
            7,
            "BGS_041",
            [["BGS_041"], [BRANN]],
            ["BG34_633", POET, BROODMOTHER],
            [KELP, CHRONO, "BGS_040", "BG36_245", "BG32_821"]
        ),
        spec!(
            "野兽 - 召唤",
            "beast_summon",
            BEAST,
            7,
            "BGS_018",
            [["BGS_018"], [GRIFFIN, BIRD], [RAT, APE, BIRD]],
            [GRIFFIN, BIRD, TITUS, RAT, APE],
            [LIONFISH, SHARK, "BG36_210", "BG36_209", "BG31_801"]
        ),
        spec!(
            "机械 - 圣盾",
            "shield",
            MECH,
            7,
            CZAR,
            [[CZAR]],
            [],
            ["BG35_342", "BG36_364", "BG36_764", "BG36_506", "BG36_853"]
        ),
    ]
});

pub fn id(m: &Minion) -> CardId {
    db().normal_of(m.card)
}
pub fn key(card: CardId) -> &'static str {
    &db().get(db().normal_of(card)).id
}
pub fn is(m: &Minion, key: &str) -> bool {
    id(m) == db().id(key)
}
pub fn has(board: &[Minion], card: CardId) -> bool {
    board.iter().any(|m| id(m) == card)
}
pub fn has_key(board: &[Minion], key: &str) -> bool {
    has(board, db().id(key))
}
pub fn tech(m: &Minion) -> bool {
    m.has(Kw::VENOMOUS) || m.has(Kw::POISONOUS) || is(m, "BG23_318")
}

pub fn candidate(
    game: &Game,
    seat: usize,
    previous: Option<Route>,
    focus: Option<Route>,
    advanced: bool,
) -> Option<(Route, i32)> {
    let p = &game.players[seat];
    if game.turn < 3 || p.tier < 2 {
        return None;
    }
    let mut locations = [0_u8; 4096];
    for m in p.shop.iter().filter_map(|s| s.item.as_minion()) {
        locations[id(m).idx()] = 1;
    }
    for m in p.hand.iter().filter_map(|h| h.item.as_minion()) {
        locations[id(m).idx()] = 2;
    }
    for m in &p.board {
        locations[id(m).idx()] = 3;
    }
    let life = p.health + p.armor;
    let pressured = life <= 14 || life <= 22 && p.stats.last_damage <= -9;
    Route::ALL
        .into_iter()
        .filter(|&route| {
            focus.is_none_or(|wanted| route == wanted) && route.has_eligible_pool(game.lobby_races)
        })
        .filter_map(|route| {
            let s = route.spec();
            let present = |card: CardId| i32::from(locations[card.idx()]);
            let primary = present(s.anchor);
            if advanced
                && route.is_bridge()
                && previous != Some(route)
                && game.turn >= 11
                && primary == 1
                && !route.core(p, &p.board)
                && !pressured
            {
                return None;
            }
            let support = s
                .core
                .iter()
                .map(|group| group.iter().map(|&c| present(c)).max().unwrap_or(0))
                .sum::<i32>();
            if game.turn < s.early || route == Poison && !pressured && focus.is_none() {
                return None;
            }
            let continuity = if previous == Some(route) { 6 } else { 0 };
            if route == PirateGold {
                return super::apm::golden_evidence(p).map(|score| (route, score + continuity));
            }
            if primary == 0 && (route != Poison && route != Beetle || support == 0)
                || primary == 1
                    && !pressured
                    && focus.is_none()
                    && support <= 2
                    && game.turn < 9
                    && route != Knight
            {
                return None;
            }
            let extra = s.full.iter().map(|&card| present(card)).sum::<i32>();
            let typed = p.board.iter().filter(|m| m.is(s.race)).count().min(5) as i32;
            let active = 10 * i32::from(route.core(p, &p.board));
            let payoff = match route {
                Knight => p.stats.eternal_knight_deaths.min(5) * 2,
                Lobster => p.stats.lobster_bonus.clamp(0, 8),
                Beetle => (p.stats.beetle_atk / 3).clamp(0, 8),
                DemonShop | DemonDeath => (p.stats.tavern_hp / 5).clamp(0, 8),
                Poison if pressured => 9,
                _ => 0,
            };
            Some((
                route,
                7 + primary * 5
                    + support * 2
                    + extra
                    + typed
                    + active
                    + payoff
                    + continuity
                    + if advanced && !route.is_bridge() && primary >= 2 {
                        3
                    } else {
                        0
                    },
            ))
        })
        .max_by_key(|(route, score)| (*score, std::cmp::Reverse(route.index())))
}
