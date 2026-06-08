# INTEGRATION.md — optimal-transport-agents-rs

> Wasserstein distance, Sinkhorn optimal transport, JKO gradient flows,
> distribution barycenters, and Earth Mover's Distance for agent
> populations. Written in Rust with `nalgebra`.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Concepts](#core-concepts)
3. [Cross-Repo Integration Map](#cross-repo-integration-map)
4. [Integration with wasserstein-agents](#integration-with-wasserstein-agents)
5. [Integration with categorical-agents](#integration-with-categorical-agents)
6. [Integration with sunset-ecosystem](#integration-with-sunset-ecosystem)
7. [Integration with conservation-law-rs](#integration-with-conservation-law-rs)
8. [Integration with si-cli](#integration-with-si-cli)
9. [Integration with si-fleet-api](#integration-with-si-fleet-api)
10. [Integration with ecosystem-dashboard](#integration-with-ecosystem-dashboard)
11. [Module Reference](#module-reference)
12. [Usage Examples](#usage-examples)
13. [Conservation Law Connection](#conservation-law-connection)
14. [Fleet.toml Integration](#fleettoml-integration)

---

## Architecture Overview

```
src/
├── lib.rs           — Crate root, re-exports public API
├── distribution.rs  — AgentDistribution: support points + weights
├── sinkhorn.rs      — Sinkhorn algorithm, W₁ and W₂ distances
├── jko.rs           — JKO gradient flow (Wasserstein time-stepping)
├── barycenter.rs    — Wasserstein barycenters (Fréchet means)
└── earth_mover.rs   — Earth Mover's Distance (unregularized OT)
```

Public API surface:

```rust
pub use barycenter::{barycenter, free_support_barycenter};
pub use distribution::AgentDistribution;
pub use earth_mover::{emd, emd_1d};
pub use jko::{jko_flow, jko_step};
pub use sinkhorn::{sinkhorn, wasserstein_1, wasserstein_2};
```

---

## Core Concepts

### Optimal Transport

Given two agent distributions μ and ν, optimal transport answers:
"What is the minimum cost to reshape μ into ν?"

### Wasserstein Distance

The Wasserstein-p distance between distributions:

```
Wₚ(μ, ν) = (min_γ ∫‖x-y‖ᵖ dγ(x,y))^(1/p)
```

We implement:
- **W₁** (Wasserstein-1) — via CDF comparison in 1D
- **W₂** (Wasserstein-2) — via Sinkhorn-regularized transport

### JKO Gradient Flow

The Jordan-Kinderlehrer-Otto scheme evolves distributions over time:

```
μ^{k+1} = argmin_μ  (1/(2τ)) W₂²(μ, μᵏ) + F(μ)
```

Each step moves the distribution toward lower-energy configurations
while respecting optimal transport geometry.

### Distribution Barycenter

The Fréchet mean in Wasserstein space:

```
ν = argmin  Σₖ ωₖ W₂²(ν, μₖ)
```

This is the natural "average" of distributions.

---

## Cross-Repo Integration Map

```
                    ┌──────────────────────────────┐
                    │   conservation-law-rs         │
                    │   γ + η = const (mathematical  │
                    │   framework)                  │
                    └──────────┬───────────────────┘
                               │ defines invariant
                               ▼
┌──────────────────┐    ┌──────────────────────┐    ┌───────────────────┐
│ wasserstein-     │    │ optimal-transport-   │    │ categorical-      │
│ agents           │───►│ agents-rs            │◄───│ agents            │
│ (uses W₂ for     │    │ (this crate)          │    │ (uses OT for      │
│  agent movement) │    │ Sinkhorn + JKO + EMD │    │  category transfer│
└──────────────────┘    └──────────┬───────────┘    └───────────────────┘
                                   │
                    ┌──────────────┼──────────────────┐
                    │              │                  │
                    ▼              ▼                  ▼
             ┌────────────┐ ┌───────────┐  ┌──────────────────┐
             │  si-cli    │ │ sunset-   │  │ ecosystem-       │
             │  check/rank│ │ ecosystem │  │ dashboard        │
             └────────────┘ └───────────┘  └──────────────────┘
```

---

## Integration with wasserstein-agents

`wasserstein-agents` uses this crate's `AgentDistribution` and `sinkhorn`
modules to compute transport plans for agent movement in Wasserstein space.

### How wasserstein-agents Uses This Crate

```rust
use optimal_transport_agents::{
    AgentDistribution,
    sinkhorn,
    wasserstein_2,
    jko_step,
};

// Define agent states as distributions
let source = AgentDistribution::new(vec![
    vec![0.0, 0.0],   // agent at origin
    vec![1.0, 0.0],   // agent at (1,0)
    vec![0.0, 1.0],   // agent at (0,1)
]);

let target = AgentDistribution::new(vec![
    vec![0.5, 0.5],
    vec![1.5, 0.5],
    vec![0.5, 1.5],
]);

// Compute optimal transport plan
let cost = source.cost_matrix_to(&target);
let plan = sinkhorn(source.weights(), target.weights(), &cost, 0.01, 500);

// Compute Wasserstein-2 distance
let w2 = wasserstein_2(source.weights(), target.weights(), &cost);
```

### JKO Evolution for Agent Dynamics

```rust
use optimal_transport_agents::{jko_flow, AgentDistribution};

let initial = AgentDistribution::new(vec![
    vec![0.0], vec![1.0], vec![2.0], vec![3.0],
]);

let cost = initial.cost_matrix_to(&initial);

// Run 10 JKO steps with τ=0.1
let trajectory = jko_flow(initial, &cost, 10, 0.1);
// trajectory[0] = initial distribution
// trajectory[10] = distribution after 10 steps of gradient flow
```

The wasserstein-agents crate wraps these primitives into higher-level
agent behaviors: spawn, halt, relocate, merge.

---

## Integration with categorical-agents

`categorical-agents` operates on discrete label distributions rather
than continuous state vectors. This crate provides the transport
infrastructure that categorical-agents uses for label transfer.

### Categorical Transport

```rust
use optimal_transport_agents::{AgentDistribution, emd_1d};

// Agent's category distribution: probability over labels
let current_labels = vec![0.3, 0.5, 0.2];  // 3 categories
let target_labels = vec![0.1, 0.6, 0.3];

// Compute Earth Mover's Distance between category distributions
let distance = emd_1d(&current_labels, &target_labels);
```

### Cost Matrix for Categories

When categories have semantic distances (e.g., "cold" is closer to "cool"
than to "hot"), categorical-agents defines a custom cost matrix:

```rust
use optimal_transport_agents::sinkhorn;

// Semantic cost between categories
let cost = vec![
    vec![0.0, 1.0, 3.0],  // cold → cold, cool, hot
    vec![1.0, 0.0, 2.0],  // cool → cold, cool, hot
    vec![3.0, 2.0, 0.0],  // hot  → cold, cool, hot
];

let source = vec![0.3, 0.5, 0.2];
let target = vec![0.1, 0.6, 0.3];

let plan = sinkhorn(&source, &target, &cost, 0.01, 500);
// plan[i][j] = amount of category i mass moved to category j
```

---

## Integration with sunset-ecosystem

The `sunset-ecosystem` manages the lifecycle of agents: spawn, reallocate,
halt. It uses this crate to compute transport distances when deciding
how to redistribute agent budgets during lifecycle transitions.

### Fleet Reallocation via Barycenter

```rust
use optimal_transport_agents::{AgentDistribution, barycenter};

// Before reallocation: each agent's budget distribution
let agent_dists: Vec<AgentDistribution> = fleet_agents
    .iter()
    .map(|agent| {
        AgentDistribution::new(vec![
            vec![agent.gamma],
            vec![agent.eta],
        ])
    })
    .collect();

// Compute barycenter (average distribution) for fleet rebalancing
let weights = vec![0.25, 0.25, 0.25, 0.25]; // equal weight per agent
let fleet_center = barycenter(&agent_dists, &weights);

// Each agent moves toward the barycenter during sunset reallocation
```

### Sunset Transition Protocol

1. Compute current fleet distribution
2. Compute target distribution (barycenter or predefined)
3. Compute transport plan via Sinkhorn
4. Execute reallocation maintaining conservation law

---

## Integration with conservation-law-rs

`conservation-law-rs` provides the mathematical framework for the
conservation invariant: **γ + η = total** for every agent.

### How They Connect

- **conservation-law-rs** defines: the invariant, verification functions,
  tolerance parameters
- **optimal-transport-agents-rs** uses: transport plans that **preserve**
  the conservation invariant during agent redistribution

### Invariant-Preserving Transport

When computing transport plans for budget reallocation, the total mass
must be conserved:

```rust
use optimal_transport_agents::sinkhorn;

// Source and target weights must both sum to 1 (normalized budgets)
let source_weights = vec![0.25, 0.25, 0.25, 0.25];
let target_weights = vec![0.25, 0.25, 0.25, 0.25]; // same total

let plan = sinkhorn(&source_weights, &target_weights, &cost, 0.01, 500);

// Verify: row sums ≈ source weights, col sums ≈ target weights
for i in 0..4 {
    let row_sum: f64 = plan[i].iter().sum();
    assert!((row_sum - source_weights[i]).abs() < 0.01);
}
```

The Sinkhorn algorithm's row/column normalization inherently preserves
the total mass, ensuring the conservation law γ + η = const is maintained
throughout transport operations.

---

## Integration with si-cli

`si-cli` can invoke conservation checks and ranking on repos that depend
on this crate. The `fleet.toml` format used by si-cli defines agent
budgets that this crate operates on:

```toml
# fleet.toml consumed by si-cli's check command
[[agents]]
name = "transport-agent-0"
gamma = 0.35    # reasoning budget
h = 0.65        # execution budget
total = 1.0     # must equal gamma + h
```

This crate's `AgentDistribution` represents those same agents as
probability distributions for transport computations.

---

## Integration with si-fleet-api

The `si-fleet-api` serves budget data from the `fleet_budgets` table
that this crate's agents populate. The conservation verification in
`si-fleet-api` (`verifyBudget`) uses the same γ + η = total check
that si-cli performs locally.

---

## Integration with ecosystem-dashboard

The dashboard's conservation gauge visualizes the γ/η split for each
agent. When agents are redistributed using this crate's transport
plans, the dashboard reflects the updated budget allocations.

---

## Module Reference

### distribution — AgentDistribution

```rust
let dist = AgentDistribution::new(vec![
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
]);

dist.n();                    // 3 (number of support points)
dist.dim();                  // 2 (dimensionality)
dist.mean();                 // [0.333, 0.333]
dist.covariance();           // 2×2 covariance matrix
dist.spread();               // trace of covariance
dist.sample(10);             // 10 sampled points
dist.cost_matrix_to(&other); // pairwise squared distances
```

### sinkhorn — Entropic Regularized Transport

```rust
let plan = sinkhorn(&a, &b, &cost, reg, max_iter);
let w1 = wasserstein_1(&a, &b, &cost_1d);
let w2 = wasserstein_2(&a, &b, &cost_matrix);
```

### jko — Gradient Flow

```rust
let updated = jko_step(&dist, &cost, step_size, tau);
let trajectory = jko_flow(initial, &cost, steps, tau);
```

### barycenter — Fréchet Means

```rust
let mean = barycenter(&distributions, &weights);
let optimized = free_support_barycenter(&supports, &weights, iterations);
```

### earth_mover — Unregularized OT

```rust
let (distance, plan) = emd(&source_points, &target_points);
let w1 = emd_1d(&source_1d, &target_1d);
```

---

## Usage Examples

### Add to Cargo.toml

```toml
[dependencies]
optimal-transport-agents-rs = { git = "https://github.com/SuperInstance/optimal-transport-agents-rs" }
```

### Basic Transport Distance

```rust
use optimal_transport_agents::{AgentDistribution, wasserstein_2};

let a = AgentDistribution::new(vec![vec![0.0], vec![1.0]]);
let b = AgentDistribution::new(vec![vec![0.5], vec![1.5]]);

let cost = a.cost_matrix_to(&b);
let distance = wasserstein_2(a.weights(), b.weights(), &cost);
println!("W₂ distance: {:.4}", distance);
```

### Fleet Barycenter Computation

```rust
use optimal_transport_agents::{AgentDistribution, barycenter};

let agents: Vec<AgentDistribution> = vec![
    AgentDistribution::new(vec![vec![0.0], vec![1.0]]),
    AgentDistribution::new(vec![vec![0.5], vec![1.5]]),
    AgentDistribution::new(vec![vec![1.0], vec![2.0]]),
];

let weights = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];
let center = barycenter(&agents, &weights);
println!("Fleet barycenter mean: {:?}", center.mean());
```

---

## Conservation Law Connection

Every transport operation in this crate preserves total mass:

1. **Sinkhorn**: Row sums = source weights, column sums = target weights
2. **JKO flow**: Total mass conserved across all time steps
3. **Barycenter**: Output weights are normalized (sum to 1)
4. **EMD**: Transport plan satisfies supply/demand constraints

This directly supports the conservation law formalized in
`conservation-law-rs`: the fleet's aggregate budget is invariant under
transport operations.

---

## Fleet.toml Integration

This crate's agents define their budget allocations in fleet.toml format,
which si-cli reads and si-fleet-api serves:

```toml
[fleet]
name = "transport-fleet"

[[agents]]
name = "ot-agent-0"
gamma = 0.40
h = 0.60
total = 1.0
capabilities = ["wasserstein-transport", "sinkhorn-computation"]

[[agents]]
name = "ot-agent-1"
gamma = 0.30
h = 0.70
total = 1.0
capabilities = ["emd-computation", "barycenter-computation"]
```

Run `si check .` to verify conservation, `si rank .` to score importance.
