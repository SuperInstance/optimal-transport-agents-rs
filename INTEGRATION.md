# Integration Guide: optimal-transport-agents-rs

## What This Crate Provides

Optimal transport, Wasserstein distances, and distribution evolution for agent populations. Computes how much "work" is required to reshape one agent distribution into another.

- **`distribution::AgentDistribution`** — Probability distribution over agent states with support points and weights. Methods: `mean()`, `covariance()`, `support_points()`.
- **`sinkhorn::sinkhorn()`** — Entropy-regularized optimal transport via Sinkhorn-Knopp algorithm. Returns transport plan and cost.
- **`sinkhorn::wasserstein_1()`** — Wasserstein-1 distance (earth mover's) between 1D distributions.
- **`sinkhorn::wasserstein_2()`** — Wasserstein-2 distance between distributions with squared cost matrix.
- **`earth_mover::emd()`** — Exact earth mover's distance in arbitrary dimensions via linear programming. Returns `(cost, transport_plan)`.
- **`earth_mover::emd_1d()`** — Fast 1D EMD using sorting.
- **`jko::jko_step()`** — Single Jordan-Kinderlehrer-Otto gradient flow step with step size and potential.
- **`jko::jko_flow()`** — Full JKO flow evolving a distribution toward a potential minimum.
- **`barycenter::barycenter()`** — Wasserstein barycenter (Fréchet mean) of multiple distributions with weights.
- **`barycenter::free_support_barycenter()`** — Barycenter with free support points.
- **`sinkhorn::pairwise_cost()`** — Pairwise cost matrix between two sets of support points.
- **`sinkhorn::pairwise_squared_cost()`** — Pairwise squared Euclidean cost matrix.

## How to Add This Crate

```bash
cargo add optimal-transport-agents
```

```rust
use optimal_transport_agents::{
    AgentDistribution, sinkhorn, wasserstein_2,
    emd, jko_flow, barycenter,
};
```

## Cross-Repo Connections

### With `conservation-law-rs`: Energy-Minimizing Transport

Use optimal transport to reallocate agent budgets while conserving total energy:

```rust
use optimal_transport_agents::{sinkhorn, pairwise_squared_cost};
use conservation_law::lagrangian::total_energy;

fn conservative_reallocation(
    source: &AgentDistribution,
    target: &AgentDistribution,
    energy_budget: f64,
) -> f64 {
    let cost = pairwise_squared_cost(&source.support_points(), &target.support_points());
    let (plan, transport_cost) = sinkhorn(
        &source.weights,
        &target.weights,
        &cost,
        0.01,  // entropy regularization
        1000,  // max iterations
        1e-6,  // tolerance
    );
    
    assert!(transport_cost <= energy_budget,
        "Transport cost {:.4} exceeds energy budget {:.4}", transport_cost, energy_budget);
    
    transport_cost
}
```

### With `si-cli`: CLI Distribution Comparison

Compare agent population distributions via CLI:

```rust
use optimal_transport_agents::{wasserstein_2, pairwise_squared_cost};

fn cli_compare_distributions(a: &AgentDistribution, b: &AgentDistribution) {
    let cost = pairwise_squared_cost(&a.support_points(), &b.support_points());
    let dist = wasserstein_2(&a.weights, &b.weights, &cost);
    println!("Wasserstein-2 distance: {:.6}", dist);
    
    if dist < 0.1 {
        println!("Distributions are nearly identical");
    } else if dist > 1.0 {
        println!("Distributions are significantly different");
    }
}
```

### With `si-fleet-api`: REST Barycenter Endpoint

Expose fleet-wide average distribution via the REST API:

```rust
use optimal_transport_agents::barycenter;
use si_fleet_api::{HttpRequest, HttpResponse};

fn post_fleet_barycenter(req: HttpRequest) -> HttpResponse {
    let distributions: Vec<AgentDistribution> = req.json().unwrap();
    let weights = vec![1.0 / distributions.len() as f64; distributions.len()];
    let avg = barycenter(&distributions, &weights);
    
    HttpResponse::json(json!({
        "mean": avg.mean(),
        "covariance": avg.covariance(),
        "support_points": avg.support_points(),
    }))
}
```

### With Supabase: Persist Transport Plans

Store optimal transport plans for audit and replay:

```rust
use optimal_transport_agents::sinkhorn;
use supabase_rs::SupabaseClient;

async fn persist_transport_plan(
    client: &SupabaseClient,
    from_agent: &str,
    to_agent: &str,
    plan: &Vec<Vec<f64>>,
    cost: f64,
) {
    client.from("transport_plans")
        .insert(json!({
            "from_agent": from_agent,
            "to_agent": to_agent,
            "plan": serde_json::to_string(plan).unwrap(),
            "cost": cost,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        }))
        .execute()
        .await
        .unwrap();
}
```

## Design Patterns

### Pattern: JKO Flow for Distribution Evolution

Evolve an agent population toward a target configuration:

```rust
use optimal_transport_agents::{jko_flow, AgentDistribution};

fn evolve_population(initial: &AgentDistribution, steps: usize) -> Vec<AgentDistribution> {
    jko_flow(initial, steps, 0.1, |x| x.iter().map(|&v| v * v).sum())
}
```

### Pattern: Multi-Scale Barycenter Consensus

Aggregate regional agent distributions into a fleet-wide consensus:

```rust
use optimal_transport_agents::barycenter;

fn fleet_consensus(regional_dists: &[AgentDistribution], region_weights: &[f64]) -> AgentDistribution {
    barycenter(regional_dists, region_weights)
}
```

### Pattern: EMD-Based Anomaly Detection

Flag agents whose state distribution deviates significantly from the fleet average:

```rust
use optimal_transport_agents::{emd, pairwise_cost};

fn detect_anomalies(
    fleet_avg: &AgentDistribution,
    agents: &[AgentDistribution],
    threshold: f64,
) -> Vec<usize> {
    let mut anomalies = vec![];
    for (i, agent) in agents.iter().enumerate() {
        let cost = pairwise_cost(&fleet_avg.support_points(), &agent.support_points());
        let (dist, _) = emd(&fleet_avg.support_points(), &agent.support_points());
        if dist > threshold {
            anomalies.push(i);
        }
    }
    anomalies
}
```

### With `spectral-fleet-rs`: Spectral Transport Cost Analysis

Combine spectral clustering with optimal transport to measure cluster separation:

```rust
use optimal_transport_agents::{wasserstein_2, pairwise_squared_cost};
use spectral_fleet::spectral_clustering::SpectralResult;

fn cluster_transport_cost(
    result: &SpectralResult,
    distributions: &[AgentDistribution],
) -> Vec<Vec<f64>> {
    let k = result.labels.iter().max().unwrap() + 1;
    let mut cluster_costs = vec![vec![0.0; k]; k];
    
    for i in 0..k {
        for j in (i + 1)..k {
            let di = &distributions[i];
            let dj = &distributions[j];
            let cost = pairwise_squared_cost(&di.support_points(), &dj.support_points());
            let dist = wasserstein_2(&di.weights, &dj.weights, &cost);
            cluster_costs[i][j] = dist;
            cluster_costs[j][i] = dist;
        }
    }
    cluster_costs
}
```

### With `dial-theory-rs`: Cultural Distance Transport

Measure how much "cultural work" is needed to align agent traditions:

```rust
use optimal_transport_agents::{emd_1d, pairwise_cost};
use dial_theory::{Tradition, DistanceMetric, tradition_distance};

fn cultural_transport_cost(a: &Tradition, b: &Tradition) -> f64 {
    let dist = tradition_distance(a, b, &DistanceMetric::Euclidean);
    // Treat distance as 1D transport cost
    emd_1d(&[a.strength], &[b.strength], &[dist])
}
```

### With Supabase: Distribution Evolution Tracking

Track how agent distributions evolve over time:

```rust
use optimal_transport_agents::{AgentDistribution, jko_flow};
use supabase_rs::SupabaseClient;

async fn track_evolution(
    client: &SupabaseClient,
    agent_id: &str,
    initial: &AgentDistribution,
    steps: usize,
) {
    let evolution = jko_flow(initial, steps, 0.1, |x| {
        x.iter().map(|&v| v * v).sum()
    });
    
    for (step, dist) in evolution.iter().enumerate() {
        client.from("distribution_evolution")
            .insert(json!({
                "agent_id": agent_id,
                "step": step,
                "mean": dist.mean(),
                "covariance": dist.covariance(),
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }))
            .execute()
            .await
            .unwrap();
    }
}
```

## Design Patterns

### Pattern: Regularized Transport for Noisy Data

Use Sinkhorn regularization when agent positions are noisy:

```rust
use optimal_transport_agents::sinkhorn;

fn robust_transport(a: &AgentDistribution, b: &AgentDistribution, epsilon: f64) -> Vec<Vec<f64>> {
    let cost = pairwise_squared_cost(&a.support_points(), &b.support_points());
    let (plan, _) = sinkhorn(&a.weights, &b.weights, &cost, epsilon, 1000, 1e-6);
    plan
}
```

### Pattern: Hierarchical Barycenter

Compute barycenter at multiple scales for large fleets:

```rust
use optimal_transport_agents::barycenter;

fn hierarchical_barycenter(distributions: &[AgentDistribution], levels: usize) -> Vec<AgentDistribution> {
    let mut current = distributions.to_vec();
    for _ in 0..levels {
        let n = current.len();
        let mut next = vec![];
        for chunk in current.chunks(2) {
            let weights = vec![1.0 / chunk.len() as f64; chunk.len()];
            next.push(barycenter(chunk, &weights));
        }
        current = next;
    }
    current
}
```

### Pattern: Transport-Based Matching

Match agents between two fleets using optimal transport:

```rust
use optimal_transport_agents::{emd, pairwise_cost};

fn match_agents(fleet_a: &AgentDistribution, fleet_b: &AgentDistribution) -> Vec<(usize, usize, f64)> {
    let cost = pairwise_cost(&fleet_a.support_points(), &fleet_b.support_points());
    let (_, plan) = emd(&fleet_a.support_points(), &fleet_b.support_points());
    
    let mut matches = vec![];
    for (i, row) in plan.iter().enumerate() {
        for (j, &flow) in row.iter().enumerate() {
            if flow > 1e-6 {
                matches.push((i, j, flow));
            }
        }
    }
    matches
}
```
