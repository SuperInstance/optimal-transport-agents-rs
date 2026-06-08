//! Distribution barycenters — Fréchet means in Wasserstein space.
//!
//! The Wasserstein barycenter of a set of distributions {μ₁, ..., μₖ} with
//! weights {ω₁, ..., ωₖ} is the distribution ν that minimizes:
//!
//! ```text
//! ν = argmin  Σₖ ωₖ W₂²(ν, μₖ)
//! ```
//!
//! This is the natural notion of "average distribution" in optimal transport
//! geometry. We provide both fixed-support barycenters (weighted averaging of
//! support points) and free-support barycenters (iterative optimization).

use crate::distribution::AgentDistribution;
use crate::sinkhorn::sinkhorn;

/// Compute the barycenter (Fréchet mean) of multiple distributions.
///
/// For distributions with the same number of support points, computes the
/// optimal transport from a uniform reference to each distribution, then
/// takes the weighted average of transported positions.
///
/// # Arguments
///
/// * `distributions` - Slice of agent distributions
/// * `weights` - Barycenter weights (must sum to 1, same length as distributions)
///
/// # Returns
///
/// A new distribution representing the Wasserstein barycenter.
///
/// # Simple case: identical distributions
///
/// The barycenter of identical distributions with any weights is the
/// distribution itself.
pub fn barycenter(distributions: &[AgentDistribution], weights: &[f64]) -> AgentDistribution {
    assert_eq!(
        distributions.len(),
        weights.len(),
        "must have same number of distributions and weights"
    );
    assert!(!distributions.is_empty(), "need at least one distribution");

    if distributions.len() == 1 {
        return distributions[0].clone();
    }

    let n = distributions[0].n();
    let d = distributions[0].dim();

    // Verify all distributions have same dimensionality
    for dist in distributions {
        assert_eq!(dist.dim(), d, "all distributions must have same dimensionality");
    }

    // For same-size distributions: compute weighted average of support points
    // using Sinkhorn-aligned transport plans
    let uniform = vec![1.0 / n as f64; n];

    // Accumulate weighted point positions
    let mut bary_points = vec![vec![0.0; d]; n];

    for (k, dist) in distributions.iter().enumerate() {
        let w = weights[k];
        let pts = dist.points();

        if dist.n() == n {
            // Same support size: use transport-based alignment
            let cost = {
                let mut c = vec![vec![0.0; dist.n()]; n];
                for i in 0..n {
                    for j in 0..dist.n() {
                        c[i][j] = pts[j]
                            .iter()
                            .enumerate()
                            .map(|(dd, v)| (v - uniform_point(i, n, d)[dd]).powi(2))
                            .sum();
                    }
                    // Use simpler: just squared distances between ref and target
                }
                // Actually compute proper cost: ref uniform points → target points
                let ref_pts = generate_uniform_reference(n, d);
                let mut c = vec![vec![0.0; dist.n()]; n];
                for i in 0..n {
                    for j in 0..dist.n() {
                        c[i][j] = ref_pts[i]
                            .iter()
                            .zip(pts[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                    }
                }
                c
            };

            let plan = sinkhorn(&uniform, dist.weights(), &cost, 0.1, 200);

            // Transport target points back to reference
            for i in 0..n {
                for j in 0..dist.n() {
                    let mass = plan[i][j];
                    for dd in 0..d {
                        bary_points[i][dd] += w * mass * pts[j][dd] / uniform[i];
                    }
                }
            }
        } else {
            // Different sizes: simple weighted average fallback
            for (i, pt) in pts.iter().enumerate().take(n) {
                for dd in 0..d {
                    bary_points[i][dd] += w * pt[dd];
                }
            }
        }
    }

    AgentDistribution::new(bary_points)
}

/// Free-support barycenter via iterative optimization.
///
/// Starts from an initial guess of support points and iteratively refines
/// them by:
/// 1. Computing transport plans from current barycenter to each input
/// 2. Updating barycenter support points as weighted average of transported positions
///
/// # Arguments
///
/// * `supports` - Support points of each input distribution
/// * `weights` - Barycenter weights (uniform if all equal)
/// * `iterations` - Number of optimization iterations
///
/// # Returns
///
/// The optimized barycenter support points.
pub fn free_support_barycenter(
    supports: &[Vec<Vec<f64>>],
    weights: &[f64],
    iterations: usize,
) -> Vec<Vec<f64>> {
    let k = supports.len();
    assert!(k > 0, "need at least one distribution");
    assert_eq!(weights.len(), k);

    // Use the first distribution's support as initial guess
    let n = supports[0].len();
    let d = supports[0][0].len();

    let mut bary = supports[0].clone();

    let bary_weights = vec![1.0 / n as f64; n];

    for _ in 0..iterations {
        let mut new_bary = vec![vec![0.0; d]; n];

        for (idx, supp) in supports.iter().enumerate() {
            let w = weights[idx];
            let m = supp.len();
            let target_weights = vec![1.0 / m as f64; m];

            // Cost matrix from barycenter to this support
            let cost = {
                let mut c = vec![vec![0.0; m]; n];
                for i in 0..n {
                    for j in 0..m {
                        c[i][j] = bary[i]
                            .iter()
                            .zip(supp[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                    }
                }
                c
            };

            let plan = sinkhorn(&bary_weights, &target_weights, &cost, 0.05, 300);

            // Accumulate transported positions
            for i in 0..n {
                for j in 0..m {
                    let mass = plan[i][j];
                    for dd in 0..d {
                        new_bary[i][dd] += w * mass * supp[j][dd] / bary_weights[i];
                    }
                }
            }
        }

        bary = new_bary;
    }

    bary
}

/// Generate a uniform reference grid in d dimensions.
fn generate_uniform_reference(n: usize, d: usize) -> Vec<Vec<f64>> {
    // Place points uniformly in [0, 1]^d
    let pts: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut pt = vec![0.0; d];
            let mut idx = i;
            for dd in 0..d {
                let side = (n as f64).powf(1.0 / d as f64).ceil() as usize;
                pt[dd] = (idx % side) as f64 / (side - 1).max(1) as f64;
                idx /= side;
            }
            pt
        })
        .collect();
    pts
}

/// Helper for uniform reference point generation.
fn uniform_point(i: usize, n: usize, d: usize) -> Vec<f64> {
    let ref_pts = generate_uniform_reference(n, d);
    ref_pts[i].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barycenter_identical_distributions() {
        let dist = AgentDistribution::new(vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ]);
        let weights = vec![0.5, 0.5];
        let result = barycenter(&[dist.clone(), dist.clone()], &weights);
        assert_eq!(result.n(), 2);
        assert_eq!(result.dim(), 2);
    }

    #[test]
    fn test_free_support_barycenter() {
        let supports = vec![
            vec![vec![0.0], vec![1.0]],
            vec![vec![0.5], vec![1.5]],
        ];
        let weights = vec![0.5, 0.5];
        let result = free_support_barycenter(&supports, &weights, 10);
        assert_eq!(result.len(), 2);
    }
}
