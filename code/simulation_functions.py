import numpy as np
import pandas as pd
from itertools import product, permutations


# # # FUNCTIONS FOR SIMULATION
def assign_size(rand_gen):
    return rand_gen.choice(["large", "medium", "small"])


def get_draw(rand_gen, size="large"):
    sign = rand_gen.choice([-1, 1])
    bounds = {
        "large": (0.8, 1),
        "medium": (0.4, 0.6),
        "small": (0, 0.2)
    }
    value = rand_gen.uniform(bounds[size][0], bounds[size][1])

    return sign * value


def generate_psis(rand_gen, count, size="small"):
    psis = None

    if size == "small":
        psis = rand_gen.uniform(0, 0.2, count) + 0.1
    elif size == "medium":
        psis = rand_gen.uniform(0, 0.2, count) + 0.3
    elif size == "large":
        psis = rand_gen.uniform(0, 0.2, count) + 0.5

    return psis


def generate_generic_loadings(rand_gen, num_man, num_fac, psis):
    loading_mat = np.zeros((num_man, num_fac), dtype=float)
    for i in range(num_man):
        psi = psis[i]
        for j in range(num_fac):
            loading_mat[i, j] = get_draw(rand_gen, assign_size(rand_gen))
        loading_mat[i] = (loading_mat[i] / np.sqrt(np.sum(np.square(loading_mat[i])))) * np.sqrt(1 - psi)

    return loading_mat


def construct_simple_structure(num_man, num_fac, counts=None):
    counts = [] if counts is None else counts
    man_vars = list(range(num_man))
    if len(counts) == 0:
        counts = [int(num_man / num_fac)] * int(num_fac)
    groupings = []
    total = 0
    for i in range(num_fac):
        groupings.append(man_vars[total:(total + counts[i])])
        total += counts[i]
    simp_struc = np.zeros((num_man, num_fac), dtype=float)
    for i in range(num_fac):
        for man_var in groupings[i]:
            simp_struc[man_var, i] = 1

    return simp_struc


def generate_simple_loadings(rand_gen, simp_struc, psis):
    loading_mat = np.zeros(simp_struc.shape, dtype=float)
    for i in range(simp_struc.shape[0]):
        psi = psis[i]
        for j in range(simp_struc.shape[1]):
            if simp_struc[i, j] == 1:
                size = "large"
            elif rand_gen.uniform(0, 1) > 0.25:
                size = "small"
            else:
                size = "medium"
            loading_mat[i, j] = get_draw(rand_gen, size)
        loading_mat[i] = (loading_mat[i] / np.sqrt(np.sum(np.square(loading_mat[i])))) * np.sqrt(1 - psi)

    return loading_mat


def generate_loadings(rand_gen, num_man, num_fac, psis, kind="generic"):
    if kind == "generic":
        return generate_generic_loadings(rand_gen, num_man, num_fac, psis)
    elif kind == "simple":
        return generate_simple_loadings(
            rand_gen,
            construct_simple_structure(num_man, num_fac),
            psis
        )


def convert_to_simple_prior(simp_struc):
    num_man = simp_struc.shape[0]
    num_fac = simp_struc.shape[1]
    groupings = []
    for i in range(num_fac):
        grouping = []
        for j in range(num_man):
            if simp_struc[j, i] == 1:
                grouping.append(j)
        groupings.append(grouping)
    simp_prior = np.zeros(shape=(num_man, num_man), dtype=float)
    for group in groupings:
        for pair in product(group, group):
            simp_prior[pair[0], pair[1]] = 1.0

    return simp_prior


def perturb_prior_matrix(rand_gen, prior_matrix, perturb_type="contaminate", level=0.05):
    perturbed = np.zeros(prior_matrix.shape, dtype=float)
    num_man = perturbed.shape[0]

    if perturb_type == "exact":
        perturbed = prior_matrix
    elif perturb_type == "contaminate":
        perturbation = np.zeros(prior_matrix.shape, dtype=float)
        for i in range(num_man):
            for j in range(i + 1, num_man):
                perturbation[i, j] = rand_gen.uniform(-level, level)
                perturbation[j, i] = perturbation[i, j]
        perturbed = prior_matrix.copy() + perturbation
        min_val = np.min(perturbed)
        if min_val < 0:
            perturbed = perturbed + abs(min_val)
        max_val = np.max(perturbed)
        if max_val > 1:
            perturbed = perturbed / max_val
    elif perturb_type == "opposite":
        for i in range(num_man):
            for j in range(i + 1, num_man):
                perturbed[i, j] = 1 - prior_matrix[i, j]
                perturbed[j, i] = perturbed[i, j]
    elif perturb_type == "random":
        for i in range(num_man):
            for j in range(i + 1, num_man):
                perturbed[i, j] = rand_gen.uniform(0, 1)
                perturbed[j, i] = perturbed[i, j]

    np.fill_diagonal(perturbed, 1)

    return perturbed


def calculate_loading_similarities(loading_mat):
    num_man = loading_mat.shape[0]
    loading_sims = np.ones(shape=(num_man, num_man), dtype=float)
    for i in range(num_man):
        for j in range(i):
            x_1 = loading_mat[i, :]
            x_2 = loading_mat[j, :]
            val = 1 - np.sqrt((1 / 2) * np.sum(((x_1 ** 2) - (x_2 ** 2)) ** 2))
            loading_sims[i, j] = val
            loading_sims[j, i] = val

    return loading_sims


def generate_data(rand_gen, loading_mat, psis, samp_size):
    num_man, num_fac = loading_mat.shape
    data = {}
    for i in range(num_man):
        data[f"X{i + 1}"] = []

    for obs in range(samp_size):
        factor_scores = rand_gen.standard_normal(num_fac)
        for i in range(num_man):
            psi = psis[i]
            loadings = loading_mat[i, :][:, np.newaxis]
            data[f"X{i + 1}"].append((factor_scores @ loadings)[0] + rand_gen.normal(0, np.sqrt(psi)))

    return pd.DataFrame(data)


def calculate_rmse(true_matrix, estimated_matrix):
    rmse = np.inf
    for perm in permutations(range(estimated_matrix.shape[1])):
        est_mat = estimated_matrix[:, perm]
        for sig_mat in product([-1, 1], repeat=true_matrix.shape[1]):
            true_mat = true_matrix @ np.diag(sig_mat)
            curr_rmse = np.sqrt(np.average(np.square(true_mat - est_mat)))
            if curr_rmse < rmse:
                rmse = curr_rmse

    return rmse


def construct_result_template(loadings_type="generic", specific_variance="small", perturbation_type="contaminate",
                              perturbation_level=0.05):
    return {
        "loadings_type": loadings_type,
        "specific_variance": specific_variance,
        "perturbation_type": perturbation_type,
        "perturbation_level": perturbation_level if perturbation_type == "contaminate" else -1,
        "best_model": "NA",
        "worst_model": "NA",
        "actual_rotation_for_priorimax": "NA",
        "models": {
            "priorimax": {"rmse": np.nan, "v_index": np.nan},
            "unrotated": {"rmse": np.nan, "v_index": np.nan},
            "varimax": {"rmse": np.nan, "v_index": np.nan},
            "oblimax": {"rmse": np.nan, "v_index": np.nan},
            "quartimax": {"rmse": np.nan, "v_index": np.nan},
            "equamax": {"rmse": np.nan, "v_index": np.nan}
        }
    }
