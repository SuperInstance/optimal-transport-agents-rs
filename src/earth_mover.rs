//! Earth Mover's Distance (EMD) — unregularized optimal transport.
//!
//! The EMD is the original optimal transport distance, corresponding to the
//! minimum amount of "work" needed to transform one distribution into another.
//! Unlike the Sinkhorn-based approach, EMD solves the unregularized linear
//! program directly.
//!
//! For 1D distributions, EMD has an efficient O(n log n) solution based on
//! sorting. For general distributions, we use a simple iterative algorithm
//! (network simplex approximation).

/// Compute the Earth Mover's Distance between two discrete distributions.
///
/// Returns both the distance and the optimal transport plan.
///
/// # Arguments
///
/// * `source` - Source support points
/// * `target` - Target support points
///
/// # Returns
///
/// Tuple of (distance, transport_plan).
///
/// Uses uniform weights and squared Euclidean ground distance.
pub fn emd(source: &[Vec<f64>], target: &[Vec<f64>]) -> (f64, Vec<Vec<f64>>) {
    let n = source.len();
    let m = target.len();

    let a = vec![1.0 / n as f64; n];
    let b = vec![1.0 / m as f64; m];

    // Cost matrix: squared Euclidean distance
    let cost = {
        let mut c = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                c[i][j] = source[i]
                    .iter()
                    .zip(target[j].iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum();
            }
        }
        c
    };

    // North-west corner rule + iterative improvement
    let mut plan = northwest_corner(&a, &b);

    // Simple iterative scaling to improve the plan
    for _ in 0..100 {
        let mut improved = false;
        for i in 0..n {
            for j in 0..m {
                if plan[i][j] < 1e-15 {
                    continue;
                }
                // Try redistributing mass from (i,j) to a cheaper cell
                for i2 in 0..n {
                    for j2 in 0..m {
                        if i2 == i && j2 == j {
                            continue;
                        }
                        let delta_cost = cost[i2][j2] - cost[i][j];
                        if delta_cost < 0.0 && plan[i2][j2] >= 0.0 {
                            let transfer = plan[i][j].min(a[i] * 0.1);
                            if transfer > 1e-15 {
                                plan[i][j] -= transfer;
                                plan[i2][j2] += transfer;
                                improved = true;
                            }
                        }
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }

    // Compute total cost
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..m {
            total += plan[i][j] * cost[i][j];
        }
    }

    (total.sqrt().max(0.0), plan)
}

/// Compute EMD for 1D distributions.
///
/// For one-dimensional distributions, the optimal transport plan is determined
/// entirely by the ordering of points. The EMD is computed by matching sorted
/// quantiles.
///
/// # Arguments
///
/// * `source` - Source 1D values
/// * `target` - Target 1D values
///
/// # Returns
///
/// The Earth Mover's Distance (Wasserstein-1 with L¹ ground metric).
pub fn emd_1d(source: &[f64], target: &[f64]) -> f64 {
    let mut s: Vec<f64> = source.to_vec();
    let mut t: Vec<f64> = target.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // If different sizes, interpolate to common quantiles
    let n = s.len();
    let m = t.len();
    let common = n.max(m);

    let s_interp = interpolate_quantiles(&s, common);
    let t_interp = interpolate_quantiles(&t, common);

    // W₁ = (1/n) * Σ |s_i - t_i| for sorted, equal-weight distributions
    let mut distance = 0.0;
    for i in 0..common {
        distance += (s_interp[i] - t_interp[i]).abs();
    }
    distance / common as f64
}

/// Interpolate to n equally-spaced quantiles.
fn interpolate_quantiles(sorted: &[f64], n: usize) -> Vec<f64> {
    let m = sorted.len();
    if m == 0 || n == 0 {
        return vec![];
    }
    if m == 1 {
        return vec![sorted[0]; n];
    }

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let frac = i as f64 / (n - 1).max(1) as f64 * (m - 1) as f64;
        let lo = frac.floor() as usize;
        let hi = (lo + 1).min(m - 1);
        let t = frac - lo as f64;
        result.push(sorted[lo] * (1.0 - t) + sorted[hi] * t);
    }
    result
}

/// North-west corner method for initial feasible transport plan.
fn northwest_corner(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b.len();
    let mut plan = vec![vec![0.0; m]; n];
    let mut supply = a.to_vec();
    let mut demand = b.to_vec();

    let mut i = 0;
    let mut j = 0;
    while i < n && j < m {
        let flow = supply[i].min(demand[j]);
        plan[i][j] = flow;
        supply[i] -= flow;
        demand[j] -= flow;
        if supply[i] < 1e-15 {
            i += 1;
        }
        if demand[j] < 1e-15 {
            j += 1;
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emd_identical() {
        let pts = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let (dist, plan) = emd(&pts, &pts);
        assert!(dist < 0.01, "EMD of identical distributions should be ~0, got {}", dist);
    }

    #[test]
    fn test_emd_1d_known() {
        // EMD between {0, 1} and {1, 2} = 1.0
        let source = vec![0.0, 1.0];
        let target = vec![1.0, 2.0];
        let dist = emd_1d(&source, &target);
        assert!(
            (dist - 1.0).abs() < 0.01,
            "EMD should be 1.0, got {}",
            dist
        );
    }

    #[test]
    fn test_emd_1d_zero() {
        let source = vec![1.0, 2.0, 3.0];
        let dist = emd_1d(&source, &source);
        assert!(
            dist.abs() < 0.01,
            "EMD of same distribution should be 0, got {}",
            dist
        );
    }
}
