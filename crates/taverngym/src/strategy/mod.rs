//! Scripted research policies shared by the evaluation and imitation-learning tools.
pub mod apm;
pub mod azerite;
pub mod cadence;
pub mod catalog;
pub mod catalog_plan;
pub mod demon;
pub mod guide_bot;
pub mod hsreplay_picks;
pub mod pivot;
mod placement;
mod policy;
pub mod quilboar;
pub mod tea;
pub mod undead_apm;
pub mod v2;
pub use policy::{Policy, PolicyState};
