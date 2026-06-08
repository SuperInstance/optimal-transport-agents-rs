//! Sinkhorn algorithm for entropic regularized optimal transport.
//!
//! The Sinkhorn algorithm computes an approximate optimal transport plan by
//! adding an entropy regularization term. This makes the problem strictly
//! convex and yields a unique solution that can be computed efficiently via
//! iterative row/column normalization.
//!
//! ## Mathematical Background
//!
//! Given source weights **a** ∈ Σₙ, target weights **b** ∈ Σₘ, and cost
//! matrix C ∈ ℝⁿˣᵐ, the entropic OT problem is:
//!
//! ```text
//! min_{P ∈ U(a,b)}  Σᵢⱼ Cᵢⱼ Pᵢⱼ - ε H(P)
//! ```
//!
//! where H(P) = -Σᵢⱼ Pᵢⱼ log(Pᵢⱼ) is the entropy and ε > 0 is the
//! regularization strength. The solution has the form P = diag(u) K diag(v)
//! where K = exp(-C/ε).

/// Sinkhorn algorithm for entropic regularized optimal transport.
///
/// Computes the optimal transport plan P between distributions with weights
/// `a` and `b`, given cost matrix `cost` and regularization parameter `reg`.
///
/// # Arguments
///
/// * `a` - Source distribution weights (must sum to 1)
/// * `b` - Target distribution weights (must sum to 1)
/// * `cost` - Cost matrix C[i][j] = cost of moving from i to j
/// * `reg` - Entropic regularization strength (ε > 0). Smaller = more accurate but slower.
/// * `max_iter` - Maximum number of Sinkhorn iterations
///
/// # Returns
///
/// The optimal transport plan P[i][j] ≈ mass moved from i to j.
///
/// # Convergence
///
/// The algorithm converges when successive iterates of u change by less
/// than ε = 1e-9, or when `max_iter` is reached.
pub fn sinkhorn(
    a: &[f64],
    b: &[f64],
    cost: &[Vec<f64>],
    reg: f64,
    max_iter: usize,
) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b.len();

    // Gibbs kernel: K[i][j] = exp(-C[i][j] / reg)
    let mut k = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            k[i][j] = (-cost[i][j] / reg).exp();
        }
    }

    // Dual variables
    let mut u = vec![1.0; n];
    let mut v = vec![1.0; m];

    for _ in 0..max_iter {
        // u = a ./ (K @ v)
        let old_u = u.clone();
        for i in 0..n {
            let sum: f64 = (0..m).map(|j| k[i][j] * v[j]).sum();
            u[i] = if sum > 1e-300 { a[i] / sum } else { 1e-300 };
        }

        // v = b ./ (Kᵀ @ u)
        for j in 0..m {
            let sum: f64 = (0..n).map(|i| k[i][j] * u[i]).sum();
            v[j] = if sum > 1e-300 { b[j] / sum } else { 1e-300 };
        }

        // Check convergence
        let delta: f64 = u
            .iter()
            .zip(old_u.iter())
            .map(|(new, old)| (new - old).abs())
            .fold(0.0, f64::max);

        if delta < 1e-9 {
            break;
        }
    }

    // P = diag(u) @ K @ diag(v)
    let mut p = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            p[i][j] = u[i] * k[i][j] * v[j];
        }
    }
    p
}

/// Compute the Wasserstein-1 distance between two 1D distributions.
///
/// For 1D distributions, the optimal transport cost with the L¹ ground metric
/// can be computed analytically via the cumulative distribution functions.
///
/// W₁(a, b) = Σᵢ |CDF_a(i) - CDF_b(i)| × cost[i]
///
/// # Arguments
///
/// * `a` - Source weights
/// * `b` - Target weights
/// * `cost` - Cost values (typically position differences or distances)
///
/// # Returns
///
/// The Wasserstein-1 distance.
pub fn wasserstein_1(a: &[f64], b: &[f64], cost: &[f64]) -> f64 {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert_eq!(cost.len(), n);

    let mut cum_a = 0.0;
    let mut cum_b = 0.0;
    let mut distance = 0.0;

    for i in 0..n {
        cum_a += a[i];
        cum_b += b[i];
        distance += (cum_a - cum_b).abs() * cost[i];
    }
    distance
}

/// Compute the Wasserstein-2 distance between two distributions.
///
/// Uses the Sinkhorn algorithm to compute the regularized transport plan,
/// then evaluates <P, C> (the Frobenius inner product of plan and cost).
///
/// W₂(a, b) = sqrt(min_P Σᵢⱼ Cᵢⱼ Pᵢⱼ)
///
/// # Arguments
///
/// * `a` - Source weights
/// * `b` - Target weights
/// * `cost` - Cost matrix C[i][j] (should be squared distances for W₂)
///
/// # Returns
///
/// The Wasserstein-2 distance.
pub fn wasserstein_2(a: &[f64], b: &[f64], cost: &[Vec<f64>]) -> f64 {
    let plan = sinkhorn(a, b, cost, 0.01, 1000);
    let mut total_cost = 0.0;
    for i in 0..a.len() {
        for j in 0..b.len() {
            total_cost += plan[i][j] * cost[i][j];
        }
    }
    total_cost.sqrt().max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinkhorn_row_sums() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![0.25, 0.25, 0.25, 0.25];
        let cost = vec![
            vec![0.0, 1.0, 4.0, 9.0],
            vec![1.0, 0.0, 1.0, 4.0],
            vec![4.0, 1.0, 0.0, 1.0],
            vec![9.0, 4.0, 1.0, 0.0],
        ];
        let plan = sinkhorn(&a, &b, &cost, 0.1, 1000);

        // Row sums should approximate a
        for i in 0..4 {
            let row_sum: f64 = plan[i].iter().sum();
            assert!((row_sum - a[i]).abs() < 0.01, "row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn test_sinkhorn_col_sums() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![0.25, 0.25, 0.25, 0.25];
        let cost = vec![
            vec![0.0, 1.0, 4.0, 9.0],
            vec![1.0, 0.0, 1.0, 4.0],
            vec![4.0, 1.0, 0.0, 1.0],
            vec![9.0, 4.0, 1.0, 0.0],
        ];
        let plan = sinkhorn(&a, &b, &cost, 0.1, 1000);

        // Column sums should approximate b
        for j in 0..4 {
            let col_sum: f64 = plan.iter().map(|row| row[j]).sum();
            assert!((col_sum - b[j]).abs() < 0.01, "col {} sum = {}", j, col_sum);
        }
    }
}
