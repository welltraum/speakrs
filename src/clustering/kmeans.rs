//! K-Means that forces a speaker count after VBx, as pyannote `VBxClustering` does with
//! `sklearn.cluster.KMeans(n_clusters, n_init=3, random_state=42)`
//!
//! Ports scikit-learn's dense path: greedy k-means++ initialization, Lloyd iterations
//! (`max_iter=300`, `tol=1e-4` scaled by the mean feature variance) and the best of three
//! initializations by inertia. Random draws come from MT19937 seeded like
//! `numpy.random.RandomState(42)`, so the initial centers match scikit-learn's.

use ndarray::{Array2, ArrayView2};

const N_INIT: usize = 3;
const RANDOM_STATE: u32 = 42;
const MAX_ITER: usize = 300;
const TOL: f64 = 1e-4;

/// Cluster the rows of `data` into `n_clusters` groups, returning a label per row
///
/// `n_clusters` must be between 1 and the number of rows.
pub(crate) fn kmeans(data: &ArrayView2<f32>, n_clusters: usize) -> Vec<usize> {
    let points = data.mapv(f64::from);
    let n_samples = points.nrows();
    assert!(
        (1..=n_samples).contains(&n_clusters),
        "k-means needs 1..={n_samples} clusters, got {n_clusters}"
    );

    let tol = mean_variance(&points) * TOL;
    let mut rng = Mt19937::new(RANDOM_STATE);
    let mut best: Option<(Vec<usize>, f64)> = None;
    for _ in 0..N_INIT {
        let centers = kmeans_plusplus(&points, n_clusters, &mut rng);
        let (labels, inertia) = lloyd(&points, centers, tol);
        let better = match &best {
            None => true,
            Some((best_labels, best_inertia)) => {
                inertia < *best_inertia && !same_clustering(&labels, best_labels, n_clusters)
            }
        };
        if better {
            best = Some((labels, inertia));
        }
    }
    best.map(|(labels, _)| labels).unwrap_or_default()
}

/// scikit-learn `_kmeans_plusplus` with uniform sample weights
fn kmeans_plusplus(points: &Array2<f64>, n_clusters: usize, rng: &mut Mt19937) -> Array2<f64> {
    let n_samples = points.nrows();
    let n_local_trials = 2 + (n_clusters as f64).ln() as usize;
    let mut centers = Array2::<f64>::zeros((n_clusters, points.ncols()));

    let first = rng.choice_uniform(n_samples);
    centers.row_mut(0).assign(&points.row(first));
    let mut closest: Vec<f64> = (0..n_samples)
        .map(|idx| squared_distance(points, idx, first))
        .collect();
    let mut potential: f64 = closest.iter().sum();

    for center_idx in 1..n_clusters {
        let mut cumulative = Vec::with_capacity(n_samples);
        let mut running = 0.0;
        for &distance in &closest {
            running += distance;
            cumulative.push(running);
        }

        let mut best: Option<(usize, f64, Vec<f64>)> = None;
        let candidates: Vec<usize> = (0..n_local_trials)
            .map(|_| {
                let target = rng.next_f64() * potential;
                cumulative
                    .partition_point(|&value| value < target)
                    .min(n_samples - 1)
            })
            .collect();
        for candidate in candidates {
            let distances: Vec<f64> = (0..n_samples)
                .map(|idx| closest[idx].min(squared_distance(points, idx, candidate)))
                .collect();
            let candidate_potential: f64 = distances.iter().sum();
            if best
                .as_ref()
                .is_none_or(|(_, best_potential, _)| candidate_potential < *best_potential)
            {
                best = Some((candidate, candidate_potential, distances));
            }
        }

        let (chosen, chosen_potential, distances) = best.expect("at least two local trials");
        centers.row_mut(center_idx).assign(&points.row(chosen));
        potential = chosen_potential;
        closest = distances;
    }
    centers
}

/// scikit-learn `_kmeans_single_lloyd`: returns labels and inertia
fn lloyd(points: &Array2<f64>, mut centers: Array2<f64>, tol: f64) -> (Vec<usize>, f64) {
    let n_samples = points.nrows();
    let mut labels = vec![usize::MAX; n_samples];
    let mut labels_old = labels.clone();
    let mut strict_convergence = false;

    for _ in 0..MAX_ITER {
        assign_labels(points, &centers, &mut labels);
        let new_centers = update_centers(points, &centers, &labels);
        let shift: f64 = (&new_centers - &centers).mapv(|value| value * value).sum();
        centers = new_centers;
        if labels == labels_old {
            strict_convergence = true;
            break;
        }
        if shift <= tol {
            break;
        }
        labels_old.clone_from(&labels);
    }
    if !strict_convergence {
        // labels must match the final centers
        assign_labels(points, &centers, &mut labels);
    }

    let inertia = (0..n_samples)
        .map(|idx| squared_distance_to(points, idx, &centers, labels[idx]))
        .sum();
    (labels, inertia)
}

fn assign_labels(points: &Array2<f64>, centers: &Array2<f64>, labels: &mut [usize]) {
    for (idx, label) in labels.iter_mut().enumerate() {
        let mut best = (0, f64::INFINITY);
        for center_idx in 0..centers.nrows() {
            let distance = squared_distance_to(points, idx, centers, center_idx);
            if distance < best.1 {
                best = (center_idx, distance);
            }
        }
        *label = best.0;
    }
}

/// New centers as cluster means; an empty cluster takes the point farthest from its
/// current center, as scikit-learn `_relocate_empty_clusters_dense` does
fn update_centers(
    points: &Array2<f64>,
    old_centers: &Array2<f64>,
    labels: &[usize],
) -> Array2<f64> {
    let n_clusters = old_centers.nrows();
    let mut sums = Array2::<f64>::zeros(old_centers.raw_dim());
    let mut weights = vec![0.0f64; n_clusters];
    for (idx, &label) in labels.iter().enumerate() {
        sums.row_mut(label).scaled_add(1.0, &points.row(idx));
        weights[label] += 1.0;
    }

    let empty: Vec<usize> = (0..n_clusters).filter(|&k| weights[k] == 0.0).collect();
    if !empty.is_empty() {
        let mut far: Vec<usize> = (0..points.nrows()).collect();
        let distances: Vec<f64> = (0..points.nrows())
            .map(|idx| squared_distance_to(points, idx, old_centers, labels[idx]))
            .collect();
        far.sort_by(|&a, &b| distances[b].total_cmp(&distances[a]));
        for (&new_cluster, &point) in empty.iter().zip(&far) {
            let old_cluster = labels[point];
            sums.row_mut(old_cluster)
                .scaled_add(-1.0, &points.row(point));
            sums.row_mut(new_cluster).assign(&points.row(point));
            weights[new_cluster] = 1.0;
            weights[old_cluster] -= 1.0;
        }
    }

    for (mut row, &weight) in sums.rows_mut().into_iter().zip(&weights) {
        if weight > 0.0 {
            row /= weight;
        }
    }
    sums
}

/// Whether two labelings are the same partition up to renaming (`_is_same_clustering`)
fn same_clustering(lhs: &[usize], rhs: &[usize], n_clusters: usize) -> bool {
    let mut mapping = vec![usize::MAX; n_clusters];
    for (&left, &right) in lhs.iter().zip(rhs) {
        if mapping[left] == usize::MAX {
            mapping[left] = right;
        } else if mapping[left] != right {
            return false;
        }
    }
    true
}

fn mean_variance(points: &Array2<f64>) -> f64 {
    let n_samples = points.nrows() as f64;
    let variances = points.columns().into_iter().map(|column| {
        let mean = column.sum() / n_samples;
        column
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / n_samples
    });
    variances.sum::<f64>() / points.ncols() as f64
}

fn squared_distance(points: &Array2<f64>, lhs: usize, rhs: usize) -> f64 {
    squared_distance_to(points, lhs, points, rhs)
}

fn squared_distance_to(
    points: &Array2<f64>,
    idx: usize,
    centers: &Array2<f64>,
    center: usize,
) -> f64 {
    points
        .row(idx)
        .iter()
        .zip(centers.row(center))
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

/// MT19937 with numpy's legacy integer seeding and `random_sample` doubles
struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for idx in 1..624 {
            let previous = state[idx - 1];
            state[idx] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(idx as u32);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for idx in 0..624 {
                let y =
                    (self.state[idx] & 0x8000_0000) | (self.state[(idx + 1) % 624] & 0x7fff_ffff);
                let mut next = self.state[(idx + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= 0x9908_b0df;
                }
                self.state[idx] = next;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `RandomState.random_sample()`: a double in `[0, 1)` with 53 random bits
    fn next_f64(&mut self) -> f64 {
        let high = f64::from(self.next_u32() >> 5);
        let low = f64::from(self.next_u32() >> 6);
        (high * 67_108_864.0 + low) / 9_007_199_254_740_992.0
    }

    /// `RandomState.choice(n, p=uniform)`: one draw against the cumulative weights
    fn choice_uniform(&mut self, n: usize) -> usize {
        let weight = f64::from(1.0f32 / n as f32);
        let cdf: Vec<f64> = (1..=n)
            .scan(0.0, |sum, _| {
                *sum += weight;
                Some(*sum)
            })
            .collect();
        let total = cdf[n - 1];
        let sample = self.next_f64();
        cdf.partition_point(|&value| value / total <= sample)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn random_sample_matches_numpy_seed_42() {
        // numpy.random.RandomState(42).random_sample(5)
        let expected = [
            0.374_540_118_847_362_5,
            0.950_714_306_409_916_2,
            0.731_993_941_811_405_1,
            0.598_658_484_197_036_6,
            0.156_018_640_442_436_52,
        ];
        let mut rng = Mt19937::new(42);
        for value in expected {
            assert_eq!(rng.next_f64(), value);
        }
    }

    #[test]
    fn separates_obvious_groups() {
        let data = array![
            [1.0f32, 0.0],
            [0.99, 0.05],
            [0.0, 1.0],
            [0.05, 0.99],
            [-1.0, 0.0],
            [-0.99, -0.05],
        ];
        let labels = kmeans(&data.view(), 3);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[4], labels[5]);
        let distinct: std::collections::BTreeSet<_> = labels.iter().collect();
        assert_eq!(distinct.len(), 3);
    }

    #[test]
    fn one_cluster_per_point_when_k_equals_rows() {
        let data = array![[1.0f32, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        let labels = kmeans(&data.view(), 3);
        let distinct: std::collections::BTreeSet<_> = labels.iter().collect();
        assert_eq!(distinct.len(), 3);
    }

    #[test]
    fn is_deterministic() {
        let data = Array2::from_shape_fn((40, 4), |(row, col)| {
            ((row * 7 + col * 3) % 11) as f32 / 11.0 + (row % 4) as f32
        });
        assert_eq!(kmeans(&data.view(), 4), kmeans(&data.view(), 4));
    }

    #[test]
    fn same_clustering_ignores_label_names() {
        assert!(same_clustering(&[0, 0, 1, 2], &[2, 2, 0, 1], 3));
        assert!(!same_clustering(&[0, 0, 1, 2], &[2, 1, 0, 1], 3));
    }
}
