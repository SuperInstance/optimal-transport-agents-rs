//! # optimal-transport-agents
//!
//! Wasserstein distance and optimal transport between agent distributions.
//!
//! This crate provides tools for computing optimal transport plans and
//! Wasserstein distances between probability distributions that represent
//! populations of agents. It implements the Sinkhorn algorithm for entropic
//! regularized transport, Jordan-Kinderlehrer-Otto (JKO) gradient flows for
//! distribution evolution, and support-point barycenters for distributional
//! averaging.
//!
//! ## Core Concepts
//!
//! **Optimal transport** answers: "How much work is required to reshape one
//! agent distribution into another?" The "work" is mass × distance, summed
//! over the optimal coupling between source and target.
//!
//! ## Modules
//!
//! - [`distribution`] — Agent distributions with mean, covariance, support points
//! - [`sinkhorn`] — Sinkhorn algorithm and Wasserstein distance computation
//! - [`jko`] — JKO gradient flow for distribution evolution
//! - [`barycenter`] — Distribution barycenters (Fréchet means)
//! - [`earth_mover`] — Earth Mover's Distance (unregularized)

pub mod barycenter;
pub mod distribution;
pub mod earth_mover;
pub mod jko;
pub mod sinkhorn;

pub use barycenter::{barycenter, free_support_barycenter};
pub use distribution::AgentDistribution;
pub use earth_mover::{emd, emd_1d};
pub use jko::{jko_flow, jko_step};
pub use sinkhorn::{sinkhorn, wasserstein_1, wasserstein_2};
