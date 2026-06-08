//! Jordan-Kinderlehrer-Otto (JKO) gradient flow for distribution evolution.
//!
//! The JKO scheme is the Wasserstein analog of implicit Euler time-stepping
//! for gradient flows. At each step, the distribution evolves by solving:
//!
//! ```text
//! μ^{k+1} = argmin_μ  (1/(2τ)) W₂²(μ, μᵏ) + F(μ)
//! ```
//!
//! where τ is the time step and F is the driving functional. In our
//! simplified setting, we use the Sinkhorn transport plan to push mass
//! from the current distribution toward a lower-energy configuration.

use crate::distribution::AgentDistribution;
use crate::sinkhorn::sinkhorn;

/// Perform a single JKO gradient flow step.
///
/// Computes the optimal transport plan from the current distribution to
/// itself (using the cost matrix), then shifts support points toward
/// lower-cost configurations weighted by the transport plan.
///
/// # Arguments
///
/// * `dist` - Current distribution
/// * `cost` - Cost matrix (squared Euclidean distances)
/// * `step_size` - How aggressively to move points (0 = no movement, 1 = full)
/// * `tau` - Time step parameter for the JKO scheme
///
/// # Returns
///
/// The updated distribution after one JKO step.
pub fn jko_step(
    dist: &AgentDistribution,
    cost: &[Vec<f64>],
    step_size: f64,
    tau: f64,
) -> AgentDistribution {
    let n = dist.n();
    let d = dist.dim();
    let weights = dist.weights().to_vec();
    let points = dist.points().to_vec();

    // Compute transport plan
    let plan = sinkhorn(&weights, &weights, cost, 0.01, 500);

    // Move each support point toward its transported position
    // New position of point i = weighted average of all j's, weighted by plan[i][j]
    let mut new_points = vec![vec![0.0; d]; n];
    for i in 0..n {
        for j in 0..n {
            let w = plan[i][j];
            for k in 0..d {
                // Move toward point j, scaled by step_size and tau
                let shift = (points[j][k] - points[i][k]) * step_size * tau;
                new_points[i][k] += points[i][k] * w + shift * w;
            }
        }
    }

    AgentDistribution::with_weights(new_points, weights)
}

/// Run a JKO gradient flow for multiple steps.
///
/// Starting from an initial distribution, iteratively applies `jko_step`
/// to produce a sequence of distributions evolving along the Wasserstein
/// gradient flow.
///
/// # Arguments
///
/// * `initial` - Starting distribution
/// * `cost` - Cost matrix between support points
/// * `steps` - Number of JKO steps to take
/// * `tau` - Time step parameter
///
/// # Returns
///
/// Vector of distributions, including the initial one (length = steps + 1).
pub fn jko_flow(
    initial: AgentDistribution,
    cost: &[Vec<f64>],
    steps: usize,
    tau: f64,
) -> Vec<AgentDistribution> {
    let mut trajectory = Vec::with_capacity(steps + 1);
    trajectory.push(initial.clone());

    let mut current = initial;
    for _ in 0..steps {
        current = jko_step(&current, cost, 0.5, tau);
        trajectory.push(current.clone());
    }

    trajectory
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jko_step_preserves_count() {
        let dist = AgentDistribution::new(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ]);
        let cost = dist.cost_matrix_to(&dist);
        let result = jko_step(&dist, &cost, 0.5, 0.1);
        assert_eq!(result.n(), 4);
        assert_eq!(result.dim(), 2);
    }

    #[test]
    fn test_jko_flow_length() {
        let dist = AgentDistribution::new(vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ]);
        let cost = dist.cost_matrix_to(&dist);
        let trajectory = jko_flow(dist, &cost, 5, 0.1);
        assert_eq!(trajectory.len(), 6); // initial + 5 steps
    }
}
