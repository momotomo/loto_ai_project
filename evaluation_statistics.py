import numpy as np


def bootstrap_mean_confidence_interval(values, confidence=0.95, n_bootstrap=2000, seed=42):
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {
            "confidence": confidence,
            "n_bootstrap": int(n_bootstrap),
            "lower": None,
            "upper": None,
            "mean": None,
        }

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, array.size, size=(n_bootstrap, array.size))
    bootstrap_means = array[bootstrap_indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower = float(np.quantile(bootstrap_means, alpha / 2.0))
    upper = float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0))
    return {
        "confidence": confidence,
        "n_bootstrap": int(n_bootstrap),
        "lower": lower,
        "upper": upper,
        "mean": float(array.mean()),
    }


def paired_permutation_test(differences, n_permutations=2000, seed=42):
    array = np.asarray(differences, dtype=np.float32)
    if array.size == 0:
        return {
            "n_permutations": int(n_permutations),
            "p_value": None,
            "observed_mean": None,
        }

    rng = np.random.default_rng(seed)
    sign_flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_permutations, array.size))
    permuted_means = np.abs((sign_flips * array).mean(axis=1))
    observed_mean = float(array.mean())
    p_value = float((np.count_nonzero(permuted_means >= abs(observed_mean)) + 1) / (n_permutations + 1))
    return {
        "n_permutations": int(n_permutations),
        "p_value": p_value,
        "observed_mean": observed_mean,
    }


def build_logloss_comparison_summary(
    candidate_losses,
    reference_losses,
    candidate_name,
    reference_name,
    confidence=0.95,
    n_bootstrap=2000,
    n_permutations=2000,
    seed=42,
):
    candidate = np.asarray(candidate_losses, dtype=np.float32)
    reference = np.asarray(reference_losses, dtype=np.float32)
    if candidate.shape != reference.shape:
        raise ValueError("candidate_losses and reference_losses must have the same shape")

    differences = candidate - reference
    bootstrap_ci = bootstrap_mean_confidence_interval(
        differences,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    permutation_test = paired_permutation_test(
        differences,
        n_permutations=n_permutations,
        seed=seed,
    )
    return {
        "candidate_name": candidate_name,
        "reference_name": reference_name,
        "metric": "per_draw_logloss",
        "sample_count": int(differences.size),
        "mean_delta": float(differences.mean()) if differences.size else None,
        "delta_definition": "candidate_logloss - reference_logloss (lower is better)",
        "bootstrap_ci": bootstrap_ci,
        "permutation_test": permutation_test,
    }
