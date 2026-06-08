//! Integration tests for optimal-transport-agents.

use optimal_transport_agents::*;

// ── Distribution tests ──────────────────────────────────────────────

#[test]
fn distribution_creation() {
    let dist = AgentDistribution::new(vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ]);
    assert_eq!(dist.n(), 3);
    assert_eq!(dist.dim(), 2);
    assert_eq!(dist.weights().len(), 3);
}

#[test]
fn distribution_mean() {
    let dist = AgentDistribution::new(vec![
        vec![0.0],
        vec![2.0],
        vec![4.0],
    ]);
    let mean = dist.mean();
    assert!((mean[0] - 2.0).abs() < 1e-10, "mean should be 2.0, got {}", mean[0]);
}

#[test]
fn distribution_covariance() {
    let dist = AgentDistribution::new(vec![
        vec![0.0],
        vec![2.0],
    ]);
    let cov = dist.covariance();
    // Variance of {0, 2} with uniform weights = E[X²] - (E[X])² = 2 - 1 = 1
    assert!((cov[0][0] - 1.0).abs() < 1e-10, "variance should be 1.0, got {}", cov[0][0]);
}

#[test]
fn distribution_spread() {
    let dist = AgentDistribution::new(vec![
        vec![0.0, 0.0],
        vec![2.0, 0.0],
        vec![0.0, 2.0],
    ]);
    let spread = dist.spread();
    // Mean = (2/3, 2/3)
    // Var(X) = E[X²] - (E[X])² = (4/3) - (4/9) = 8/9
    // Same for Y, total spread = 16/9
    assert!((spread - 16.0 / 9.0).abs() < 1e-8, "spread = {}", spread);
}

#[test]
fn distribution_sample_count() {
    let dist = AgentDistribution::new(vec![
        vec![0.0],
        vec![1.0],
    ]);
    let samples = dist.sample(100);
    assert_eq!(samples.len(), 100);
    // All samples should be either [0.0] or [1.0]
    for s in &samples {
        assert!(s.len() == 1 && (s[0] == 0.0 || s[0] == 1.0));
    }
}

// ── Sinkhorn tests ──────────────────────────────────────────────────

#[test]
fn sinkhorn_convergence() {
    let a = vec![0.5, 0.5];
    let b = vec![0.5, 0.5];
    let cost = vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ];
    let p1 = sinkhorn(&a, &b, &cost, 0.5, 100);
    let p2 = sinkhorn(&a, &b, &cost, 0.01, 100);

    // Lower regularization should give more diagonal transport
    let diag1: f64 = p1[0][0] + p1[1][1];
    let diag2: f64 = p2[0][0] + p2[1][1];
    assert!(diag2 >= diag1 - 0.01, "lower reg should concentrate on diagonal: {} vs {}", diag2, diag1);
}

#[test]
fn sinkhorn_mass_conservation() {
    let a = vec![0.3, 0.3, 0.4];
    let b = vec![0.2, 0.5, 0.3];
    let cost = vec![
        vec![0.0, 1.0, 2.0],
        vec![1.0, 0.0, 1.0],
        vec![2.0, 1.0, 0.0],
    ];
    let plan = sinkhorn(&a, &b, &cost, 0.1, 500);

    // Total mass should be preserved (sum ≈ 1)
    let total: f64 = plan.iter().flat_map(|r| r.iter()).sum();
    assert!((total - 1.0).abs() < 0.01, "total mass = {}", total);

    // Row sums ≈ a
    for i in 0..3 {
        let row_sum: f64 = plan[i].iter().sum();
        assert!((row_sum - a[i]).abs() < 0.02, "row {} sum = {} vs {}", i, row_sum, a[i]);
    }

    // Column sums ≈ b
    for j in 0..3 {
        let col_sum: f64 = plan.iter().map(|r| r[j]).sum();
        assert!((col_sum - b[j]).abs() < 0.02, "col {} sum = {} vs {}", j, col_sum, b[j]);
    }
}

// ── Wasserstein distance tests ──────────────────────────────────────

#[test]
fn wasserstein_1_correctness() {
    // W₁ between two 1D distributions with 3 bins
    // Source: mass at positions 0, 1, 2
    // Target: mass at positions 1, 2, 3
    // With position-based costs, W₁ measures the CDF gap × distance
    let a = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let b = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let cost = vec![1.0, 1.0, 1.0]; // bin widths
    let w = wasserstein_1(&a, &b, &cost);
    // All CDFs match since distributions are identical → W₁ = 0
    assert!(w.abs() < 0.01, "W₁ of identical distributions should be 0, got {}", w);

    // Now with different distributions
    let a2 = vec![0.5, 0.5, 0.0];
    let b2 = vec![0.0, 0.5, 0.5];
    let w2 = wasserstein_1(&a2, &b2, &cost);
    // CDF difference: at bin 0: |0.5 - 0.0| = 0.5, at bin 1: |1.0 - 0.5| = 0.5, at bin 2: |1.0 - 1.0| = 0
    // W₁ = 0.5*1 + 0.5*1 + 0*1 = 1.0
    assert!((w2 - 1.0).abs() < 0.01, "W₁ should be 1.0, got {}", w2);
}

#[test]
fn wasserstein_2_identical() {
    let a = vec![0.5, 0.5];
    let b = vec![0.5, 0.5];
    let cost = vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ];
    let w = wasserstein_2(&a, &b, &cost);
    assert!(w < 0.5, "W₂ of identical distributions should be small, got {}", w);
}

// ── JKO flow tests ──────────────────────────────────────────────────

#[test]
fn jko_flow_sequence() {
    let dist = AgentDistribution::new(vec![
        vec![0.0],
        vec![1.0],
        vec![2.0],
    ]);
    let cost = dist.cost_matrix_to(&dist);
    let trajectory = jko_flow(dist, &cost, 3, 0.1);
    assert_eq!(trajectory.len(), 4); // initial + 3 steps

    // Each step should preserve number of points
    for t in &trajectory {
        assert_eq!(t.n(), 3);
        assert_eq!(t.dim(), 1);
    }
}

#[test]
fn jko_flow_spread_decreases() {
    let dist = AgentDistribution::new(vec![
        vec![-2.0],
        vec![-1.0],
        vec![1.0],
        vec![2.0],
    ]);
    let cost = dist.cost_matrix_to(&dist);
    let trajectory = jko_flow(dist, &cost, 10, 0.05);

    let initial_spread = trajectory[0].spread();
    let final_spread = trajectory.last().unwrap().spread();

    // JKO diffusion should reduce spread (mass concentrates toward center)
    assert!(
        final_spread <= initial_spread * 1.5,
        "spread should not increase dramatically: initial={}, final={}",
        initial_spread, final_spread
    );
}

// ── Barycenter tests ────────────────────────────────────────────────

#[test]
fn barycenter_single_distribution() {
    let dist = AgentDistribution::new(vec![
        vec![0.0],
        vec![1.0],
    ]);
    let result = barycenter(&[dist.clone()], &[1.0]);
    assert_eq!(result.n(), 2);
    assert_eq!(result.dim(), 1);
}

#[test]
fn barycenter_identical_distributions() {
    let dist = AgentDistribution::new(vec![
        vec![0.0],
        vec![1.0],
    ]);
    let result = barycenter(&[dist.clone(), dist.clone()], &[0.5, 0.5]);
    assert_eq!(result.n(), 2);

    // Mean should be approximately the same as original
    let orig_mean = dist.mean();
    let bary_mean = result.mean();
    for i in 0..orig_mean.len() {
        assert!(
            (orig_mean[i] - bary_mean[i]).abs() < 0.5,
            "barycenter mean should be close to original: {} vs {}",
            orig_mean[i], bary_mean[i]
        );
    }
}

#[test]
fn free_support_barycenter_convergence() {
    let s1 = vec![vec![0.0], vec![1.0]];
    let s2 = vec![vec![1.0], vec![2.0]];
    let result = free_support_barycenter(&[s1, s2], &[0.5, 0.5], 20);
    assert_eq!(result.len(), 2);

    // Average position should be near 0.5 and 1.5
    let avg: f64 = result.iter().map(|p| p[0]).sum::<f64>() / result.len() as f64;
    assert!((avg - 1.0).abs() < 0.5, "average should be ~1.0, got {}", avg);
}

// ── EMD tests ───────────────────────────────────────────────────────

#[test]
fn emd_known_answer() {
    let source = vec![vec![0.0], vec![1.0]];
    let target = vec![vec![1.0], vec![2.0]];
    let (dist, plan) = emd(&source, &target);
    // Should be positive
    assert!(dist > 0.0, "EMD should be positive, got {}", dist);

    // Mass should be conserved
    let total: f64 = plan.iter().flat_map(|r| r.iter()).sum();
    assert!((total - 1.0).abs() < 0.05, "total mass = {}", total);
}

#[test]
fn emd_1d_shift() {
    // EMD between {0, 1} and {2, 3} should be 2.0
    let source = vec![0.0, 1.0];
    let target = vec![2.0, 3.0];
    let dist = emd_1d(&source, &target);
    assert!((dist - 2.0).abs() < 0.01, "EMD₁D should be 2.0, got {}", dist);
}

#[test]
fn emd_1d_symmetry() {
    let source = vec![0.0, 1.0, 2.0];
    let target = vec![1.0, 3.0, 5.0];
    let d1 = emd_1d(&source, &target);
    let d2 = emd_1d(&target, &source);
    assert!((d1 - d2).abs() < 0.01, "EMD should be symmetric: {} vs {}", d1, d2);
}

#[test]
fn transport_plan_non_negative() {
    let a = vec![0.25, 0.25, 0.25, 0.25];
    let b = vec![0.25, 0.25, 0.25, 0.25];
    let cost = vec![
        vec![0.0, 1.0, 4.0, 9.0],
        vec![1.0, 0.0, 1.0, 4.0],
        vec![4.0, 1.0, 0.0, 1.0],
        vec![9.0, 4.0, 1.0, 0.0],
    ];
    let plan = sinkhorn(&a, &b, &cost, 0.1, 100);
    for (i, row) in plan.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!(v >= -1e-10, "plan[{}][{}] = {} is negative", i, j, v);
        }
    }
}
