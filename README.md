# optimal-transport-agents

**Wasserstein distance and optimal transport between agent distributions.**

Rust library for computing how much "work" it takes to reshape one agent distribution into another. Implements the Sinkhorn algorithm for entropic regularized transport, Jordan-Kinderlehrer-Otto (JKO) gradient flows for distribution evolution, Fréchet barycenters for distribution averaging, and Earth Mover's Distance for direct computation.

## The Core Idea

Imagine you have a pile of sand (source distribution) and want to fill a hole (target distribution). Optimal transport asks: **what's the cheapest way to move all the sand into the hole?**

The cost is mass × distance. The Wasserstein distance is the minimum total cost. It's a metric on the space of probability distributions that respects their geometry — two distributions that are "close" in shape and location have small Wasserstein distance.

For agent systems, this means:
- **How different are two populations of agents?**
- **How has an agent distribution evolved over time?**
- **What's the "average" of several agent distributions?**

## Quick Start

```toml
[dependencies]
optimal-transport-agents = "0.1.0"
```

```rust
use optimal_transport_agents::*;

fn main() {
    // Two distributions of agents
    let source = AgentDistribution::new(vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ]);

    let target = AgentDistribution::new(vec![
        vec![2.0, 0.0],
        vec![3.0, 0.0],
        vec![2.0, 1.0],
        vec![3.0, 1.0],
    ]);

    // Compute cost matrix (squared Euclidean distances)
    let cost = source.cost_matrix_to(&target);

    // Sinkhorn transport plan
    let plan = sinkhorn(
        source.weights(),
        target.weights(),
        &cost,
        0.1,   // regularization
        1000,  // max iterations
    );

    // Wasserstein-2 distance
    let w2 = wasserstein_2(source.weights(), target.weights(), &cost);
    println!("W₂ = {:.4}", w2);
    // => W₂ = 2.0000

    // Earth Mover's Distance (1D case)
    let d = emd_1d(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
    println!("EMD₁D = {:.4}", d);
    // => EMD₁D = 1.0000
}
```

## Mathematical Background

### Optimal Transport

Given two probability measures μ and ν on ℝᵈ, the optimal transport problem is:

```
W_p(μ, ν) = (inf_{γ ∈ Π(μ,ν)} ∫‖x - y‖ᵖ dγ(x,y))^{1/p}
```

where Π(μ,ν) is the set of all couplings (joint distributions) with marginals μ and ν.

For discrete distributions with weights **a** ∈ Σₙ and **b** ∈ Σₘ:

```
W_p(a, b) = (min_{P ∈ U(a,b)} Σᵢⱼ Cᵢⱼᵖ Pᵢⱼ)^{1/p}
```

where U(a,b) = {P ∈ ℝⁿˣᵐ : P𝟏 = a, Pᵀ𝟏 = b, P ≥ 0} is the set of valid transport plans.

### Sinkhorn Algorithm

The entropic regularized OT problem adds an entropy term:

```
min_{P ∈ U(a,b)} Σᵢⱼ Cᵢⱼ Pᵢⱼ - ε H(P)
```

where H(P) = -Σᵢⱼ Pᵢⱼ log(Pᵢⱼ) and ε > 0 is the regularization.

The solution has the form **P = diag(u) K diag(v)** where K = exp(-C/ε).

Sinkhorn alternates:
1. **u = a / (Kv)** — normalize rows
2. **v = b / (Kᵀu)** — normalize columns

This converges to the unique optimal plan. Lower ε → more accurate but slower convergence.

### Wasserstein Distance

- **W₁ (Earth Mover's Distance):** p = 1, the minimum total distance to move all mass.
- **W₂:** p = 2, penalizes long-range transport more heavily.

Properties:
- W_p(μ, ν) = 0 iff μ = ν
- W_p(μ, ν) = W_p(ν, μ) (symmetric)
- W_p(μ, λ) ≤ W_p(μ, ν) + W_p(ν, λ) (triangle inequality)

### JKO Gradient Flow

The Jordan-Kinderlehrer-Otto scheme evolves distributions via:

```
μ^{k+1} = argmin_μ { (1/(2τ)) W₂²(μ, μᵏ) + τ F(μ) }
```

This is the Wasserstein-space analog of gradient descent. Each step moves the distribution toward lower values of the functional F while staying close to the previous iterate.

Applications:
- Diffusion processes (F = entropy → heat equation)
- Aggregation (F = interaction potential → swarming models)
- Porous medium equation
- Fokker-Planck equation

### Distribution Barycenter

The Wasserstein barycenter of distributions {μ₁, ..., μₖ} with weights {ω₁, ..., ωₖ}:

```
ν = argmin Σₖ ωₖ W₂²(ν, μₖ)
```

This is the Fréchet mean in Wasserstein space — the "average distribution" that minimizes total squared distance to all inputs.

## API Reference

### `AgentDistribution`

Represents a discrete probability distribution over agent states.

```rust
// Create uniform distribution
let dist = AgentDistribution::new(vec![
    vec![0.0, 0.0],  // point 1
    vec![1.0, 0.0],  // point 2
    vec![0.5, 1.0],  // point 3
]);

// Custom weights
let weighted = AgentDistribution::with_weights(
    vec![vec![0.0], vec![1.0], vec![2.0]],
    vec![0.5, 0.3, 0.2],  // automatically normalized
);

// Statistics
let mu = dist.mean();           // => [0.5, 0.333]
let cov = dist.covariance();    // => 2×2 matrix
let spread = dist.spread();     // => trace(covariance)

// Sampling (deterministic seed)
let samples = dist.sample(100); // => 100 points
```

### `sinkhorn`

Entropic regularized optimal transport.

```rust
let plan = sinkhorn(
    &source_weights,   // &[f64], sums to 1
    &target_weights,   // &[f64], sums to 1
    &cost_matrix,      // &[Vec<f64>], n×m
    0.1,               // regularization ε
    1000,              // max iterations
);

// plan[i][j] = mass transported from i to j
// Row sums ≈ source_weights
// Col sums ≈ target_weights
```

### `wasserstein_1` / `wasserstein_2`

```rust
// W₁ via CDF method (1D)
let w1 = wasserstein_1(&a, &b, &cost);

// W₂ via Sinkhorn
let w2 = wasserstein_2(&a, &b, &cost_matrix);
```

### `jko_step` / `jko_flow`

```rust
// Single JKO step
let next = jko_step(&dist, &cost, 0.5, 0.1);

// Full trajectory
let trajectory = jko_flow(initial_dist, &cost, 10, 0.05);
// trajectory[0] = initial
// trajectory[k] = after k steps
```

### `barycenter` / `free_support_barycenter`

```rust
// Fixed-support barycenter
let bary = barycenter(&[dist1, dist2, dist3], &[0.5, 0.3, 0.2]);

// Free-support barycenter (optimizes support points)
let points = free_support_barycenter(
    &[support1, support2],
    &[0.5, 0.5],
    20,  // iterations
);
```

### `emd` / `emd_1d`

```rust
// General EMD
let (distance, plan) = emd(&source_points, &target_points);

// 1D EMD (efficient sorting-based)
let d = emd_1d(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
// => 1.0
```

## Examples

### Comparing Agent Populations

```rust
use optimal_transport_agents::*;

// Two teams of agents in 2D space
let team_a = AgentDistribution::new(vec![
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
]);

let team_b = AgentDistribution::new(vec![
    vec![10.0, 0.0],
    vec![11.0, 0.0],
    vec![10.0, 1.0],
]);

let cost = team_a.cost_matrix_to(&team_b);
let w2 = wasserstein_2(team_a.weights(), team_b.weights(), &cost);
println!("Teams are {:.2} units apart", w2);
// => Teams are 10.00 units apart
```

### JKO Diffusion

```rust
use optimal_transport_agents::*;

// Start with concentrated distribution
let initial = AgentDistribution::new(vec![
    vec![-1.0],
    vec![0.0],
    vec![1.0],
]);

let cost = initial.cost_matrix_to(&initial);

// Run JKO flow (diffusion)
let trajectory = jko_flow(initial, &cost, 20, 0.05);

println!("Initial spread: {:.4}", trajectory[0].spread());
println!("Final spread:   {:.4}", trajectory.last().unwrap().spread());
// Distribution spreads out over time
```

### Distribution Barycenter

```rust
use optimal_transport_agents::*;

let dist_a = AgentDistribution::new(vec![
    vec![0.0], vec![1.0],
]);

let dist_b = AgentDistribution::new(vec![
    vec![4.0], vec![5.0],
]);

// Equal-weight barycenter
let mid = barycenter(&[dist_a, dist_b], &[0.5, 0.5]);
println!("Barycenter mean: {:?}", mid.mean());
// => Barycenter mean: ~[2.5]

// Free-support version
let supports = vec![
    vec![vec![0.0], vec![1.0]],
    vec![vec![4.0], vec![5.0]],
];
let free = free_support_barycenter(&supports, &[0.5, 0.5], 30);
println!("Free support: {:?}", free);
// => Points approximately at [2.0] and [3.0]
```

### Earth Mover's Distance

```rust
use optimal_transport_agents::*;

// 1D: sorted matching
let d = emd_1d(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]);
assert!((d - 0.0).abs() < 0.01);  // identical → 0
// => true

let d = emd_1d(&[0.0, 1.0], &[2.0, 3.0]);
assert!((d - 2.0).abs() < 0.01);  // shifted by 2
// => true

// Multi-dimensional
let (dist, plan) = emd(
    &[vec![0.0, 0.0], vec![1.0, 1.0]],
    &[vec![1.0, 0.0], vec![2.0, 1.0]],
);
println!("EMD = {:.4}, mass conserved = {}", dist,
    (plan.iter().flat_map(|r| r.iter()).sum::<f64>() - 1.0).abs() < 0.05
);
// => EMD = 1.0000, mass conserved = true
```

## Integration with Conservation Laws

This library is designed to compose naturally with conservation-law constraints. Agent distributions carry mass (probability weights that sum to 1), and every transport operation preserves total mass:

```rust
use optimal_transport_agents::*;

// Transport preserves total mass
let a = vec![0.3, 0.3, 0.4];
let b = vec![0.2, 0.5, 0.3];
let cost = vec![
    vec![0.0, 1.0, 4.0],
    vec![1.0, 0.0, 1.0],
    vec![4.0, 1.0, 0.0],
];

let plan = sinkhorn(&a, &b, &cost, 0.1, 500);

// Conservation: Σᵢⱼ Pᵢⱼ = 1 (total mass)
let total_mass: f64 = plan.iter().flat_map(|r| r.iter()).sum();
assert!((total_mass - 1.0).abs() < 0.01);
// => true

// Source conservation: Σⱼ Pᵢⱼ = aᵢ for each i
for i in 0..a.len() {
    let row_sum: f64 = plan[i].iter().sum();
    assert!((row_sum - a[i]).abs() < 0.02);
}
// => true (all rows checked)

// Target conservation: Σᵢ Pᵢⱼ = bⱼ for each j
for j in 0..b.len() {
    let col_sum: f64 = plan.iter().map(|r| r[j]).sum();
    assert!((col_sum - b[j]).abs() < 0.02);
}
// => true (all columns checked)
```

### Combining with `conservation-law` Crate

When modeling physical agent systems, mass conservation is not optional — it's physics. The transport plans from this library guarantee:

1. **Total mass conservation:** Σᵢⱼ Pᵢⱼ = Σᵢ aᵢ = 1
2. **Source marginal:** Σⱼ Pᵢⱼ = aᵢ (all mass leaves source)
3. **Target marginal:** Σᵢ Pᵢⱼ = bⱼ (all mass arrives at target)
4. **Non-negativity:** Pᵢⱼ ≥ 0 (no negative mass)

```rust
// Verify conservation invariants on every transport plan
fn verify_conservation(plan: &[Vec<f64>], source: &[f64], target: &[f64]) -> bool {
    let eps = 0.02;

    // Non-negativity
    for row in plan {
        for &val in row {
            if val < -eps { return false; }
        }
    }

    // Source marginal
    for (i, row) in plan.iter().enumerate() {
        if (row.iter().sum::<f64>() - source[i]).abs() > eps {
            return false;
        }
    }

    // Target marginal
    for j in 0..target.len() {
        let col_sum: f64 = plan.iter().map(|r| r[j]).sum();
        if (col_sum - target[j]).abs() > eps {
            return false;
        }
    }

    // Total mass
    let total: f64 = plan.iter().flat_map(|r| r.iter()).sum();
    if (total - 1.0).abs() > eps {
        return false;
    }

    true
}
```

## Performance

The Sinkhorn algorithm runs in O(nm × iterations) where n and m are the sizes of the source and target distributions. For most practical cases:

| Distribution sizes | Regularization | Iterations | Time |
|---|---|---|---|
| 10 × 10 | 0.1 | ~50 | < 1ms |
| 100 × 100 | 0.1 | ~100 | ~5ms |
| 1000 × 1000 | 0.1 | ~200 | ~500ms |

Lower regularization → more accurate but more iterations needed.

## Algorithm Details

### Sinkhorn Convergence

The algorithm provably converges for any ε > 0. The convergence rate is linear:

```
‖P^{k+1} - P*‖ ≤ ρ ‖P^k - P*‖
```

where ρ < 1 depends on ε and the cost matrix. In practice, convergence is fast (50-500 iterations for reasonable accuracy).

### Choosing Regularization

- **ε = 0.01 - 0.1:** High accuracy, sharp transport plans. Good for computing precise distances.
- **ε = 0.1 - 1.0:** Moderate accuracy, smoother plans. Good for gradient-based optimization.
- **ε > 1.0:** Heavy smoothing. The plan becomes close to the product measure a × b. Useful for initialization.

### Numerical Stability

For very small ε, the kernel matrix K = exp(-C/ε) can underflow. This implementation handles this with floor clamping at 1e-300. For production use with small ε, consider the log-domain Sinkhorn variant.

## Cargo Features

Currently no optional features. The library uses `nalgebra` for potential future linear algebra operations but all core algorithms work with standard `Vec`.

## License

MIT
