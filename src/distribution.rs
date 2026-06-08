//! Agent distribution representation.
//!
//! An `AgentDistribution` represents a probability distribution over agent
//! states as a set of weighted support points in ℝᵈ. Each support point is
//! an agent's state vector, and the weights form a discrete probability
//! measure (they sum to 1).

use nalgebra::{DMatrix, DVector, RowDVector};

/// A discrete probability distribution over agent states.
///
/// Represented as a weighted set of support points in ℝᵈ.
/// The weights are uniform by default (1/n for each of n points).
#[derive(Clone, Debug)]
pub struct AgentDistribution {
    /// Support points as row vectors. Shape: (n, d).
    points: Vec<Vec<f64>>,
    /// Probability weights. Length: n. Sums to 1.0.
    weights: Vec<f64>,
}

impl AgentDistribution {
    /// Create a new uniform distribution over the given support points.
    ///
    /// Each point gets weight 1/n.
    ///
    /// # Panics
    ///
    /// Panics if `points` is empty.
    pub fn new(points: Vec<Vec<f64>>) -> Self {
        assert!(!points.is_empty(), "distribution must have at least one point");
        let n = points.len();
        let w = 1.0 / n as f64;
        Self {
            points,
            weights: vec![w; n],
        }
    }

    /// Create a distribution with explicit weights.
    ///
    /// Weights are automatically normalized to sum to 1.
    pub fn with_weights(points: Vec<Vec<f64>>, mut weights: Vec<f64>) -> Self {
        assert_eq!(points.len(), weights.len(), "points and weights must have same length");
        assert!(!points.is_empty(), "distribution must have at least one point");
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }
        Self { points, weights }
    }

    /// Number of support points.
    pub fn n(&self) -> usize {
        self.points.len()
    }

    /// Dimensionality of each support point.
    pub fn dim(&self) -> usize {
        self.points[0].len()
    }

    /// Reference to the support points.
    pub fn points(&self) -> &[Vec<f64>] {
        &self.points
    }

    /// Reference to the weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Mutable reference to weights.
    pub fn weights_mut(&mut self) -> &mut Vec<f64> {
        &mut self.weights
    }

    /// Compute the mean (expected value) of the distribution.
    ///
    /// Returns a vector in ℝᵈ.
    pub fn mean(&self) -> Vec<f64> {
        let d = self.dim();
        let mut mu = vec![0.0; d];
        for (i, pt) in self.points.iter().enumerate() {
            let w = self.weights[i];
            for j in 0..d {
                mu[j] += w * pt[j];
            }
        }
        mu
    }

    /// Compute the covariance matrix of the distribution.
    ///
    /// Returns a d×d matrix stored as row-major Vec<f64>.
    /// cov[i][j] = E[(X_i - μ_i)(X_j - μ_j)]
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        let d = self.dim();
        let mu = self.mean();
        let mut cov = vec![vec![0.0; d]; d];
        for (i, pt) in self.points.iter().enumerate() {
            let w = self.weights[i];
            let diff: Vec<f64> = pt.iter().zip(mu.iter()).map(|(a, b)| a - b).collect();
            for r in 0..d {
                for c in 0..d {
                    cov[r][c] += w * diff[r] * diff[c];
                }
            }
        }
        cov
    }

    /// Compute the spread (trace of covariance = total variance).
    ///
    /// This is the sum of variances along each dimension.
    pub fn spread(&self) -> f64 {
        let cov = self.covariance();
        let d = cov.len();
        let mut trace = 0.0;
        for i in 0..d {
            trace += cov[i][i];
        }
        trace
    }

    /// Sample n points from the distribution (with replacement).
    ///
    /// Uses the weights as probabilities. Returns n support points.
    pub fn sample(&self, n: usize) -> Vec<Vec<f64>> {
        // Build cumulative distribution
        let mut cdf = Vec::with_capacity(self.weights.len());
        let mut cum = 0.0;
        for &w in &self.weights {
            cum += w;
            cdf.push(cum);
        }

        let mut rng = SimpleRng::new(42);
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let u = rng.next();
            // Binary search for the index
            let idx = match cdf.binary_search_by(|v| v.partial_cmp(&u).unwrap()) {
                Ok(i) => i,
                Err(i) => i.min(self.points.len() - 1),
            };
            samples.push(self.points[idx].clone());
        }
        samples
    }

    /// Compute the weighted cost matrix between this distribution and another.
    ///
    /// C[i][j] = ||a_i - b_j||² (squared Euclidean distance).
    pub fn cost_matrix_to(&self, other: &AgentDistribution) -> Vec<Vec<f64>> {
        let n = self.points.len();
        let m = other.points.len();
        let mut cost = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                let d2: f64 = self.points[i]
                    .iter()
                    .zip(other.points[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                cost[i][j] = d2;
            }
        }
        cost
    }
}

/// Simple LCG random number generator for deterministic sampling.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        // LCG constants (Numerical Recipes)
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as f64 / (1u64 << 31) as f64
    }
}

/// Compute pairwise squared Euclidean distance matrix between two sets of points.
pub fn pairwise_squared_cost(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b.len();
    let mut cost = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            cost[i][j] = a[i]
                .iter()
                .zip(b[j].iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum();
        }
    }
    cost
}

/// Compute pairwise Euclidean distance matrix.
pub fn pairwise_cost(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    pairwise_squared_cost(a, b)
        .into_iter()
        .map(|row| row.into_iter().map(|v| v.sqrt()).collect())
        .collect()
}
