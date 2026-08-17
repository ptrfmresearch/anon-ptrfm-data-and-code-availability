import pickle
import time
import warnings
from main import InterpretableFA
from simulation_functions import *

# # # CONSTANTS FOR SIMULATION
RAND_SEED = 123
RAND_GEN = np.random.RandomState(RAND_SEED)
NUM_ITER = 1000
NUM_MANIFEST = 18
NUM_FACTORS = 3
SAMP_SIZE = 300
ROTATIONS = ["unrotated", "varimax", "equamax", "quartimax"]

# # # SIMULATION
start_time = time.perf_counter()
loadings_kinds = ["simple", "generic"]
pert_types = ["contaminate"]
psi_vars = ["small", "medium", "large"]
pert_levels = [4.5, 5, 5.5]
results = []

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
iter_count = 0
total_iters = len(loadings_kinds) * len(pert_types) * len(psi_vars) * len(pert_levels) * NUM_ITER

for config in product(loadings_kinds, pert_types, psi_vars, pert_levels):
    loadings_kind = config[0]
    pert_type = config[1]
    psi_var = config[2]
    pert_level = config[3]

    for _ in range(NUM_ITER):
        temp_result = construct_result_template(loadings_kind, psi_var, pert_type, pert_level)
        psis = generate_psis(RAND_GEN, NUM_MANIFEST, psi_var)
        loadings = generate_loadings(RAND_GEN, NUM_MANIFEST, NUM_FACTORS, psis, loadings_kind)
        data = generate_data(RAND_GEN, loadings, psis, SAMP_SIZE)
        prior = perturb_prior_matrix(RAND_GEN, calculate_loading_similarities(loadings), pert_type, pert_level)
        analyzer = InterpretableFA(data, prior)

        fit_result = analyzer.fit_factor_model("priorimax", NUM_FACTORS,
                                               "priorimax", num_starts=5, max_time=30.0)
        temp_result["models"]["priorimax"]["rmse"] = calculate_rmse(loadings, analyzer.models["priorimax"].loadings_)
        temp_result["models"]["priorimax"]["v_index"] = analyzer.calculate_v_index("priorimax")
        temp_result["actual_rotation_for_priorimax"] = fit_result[1]
        best_v = 0
        worst_v = 1
        for rot in ROTATIONS:
            do_rot = None if rot == "unrotated" else rot
            if rot == "equamax":
                rot_kwargs = {
                    "kappa": NUM_FACTORS / (2 * NUM_MANIFEST)
                }
            else:
                rot_kwargs = None
            analyzer.fit_factor_model(rot, NUM_FACTORS, do_rot, rotation_kwargs=rot_kwargs)
            temp_result["models"][rot]["rmse"] = calculate_rmse(loadings, analyzer.models[rot].loadings_)
            temp_result["models"][rot]["v_index"] = analyzer.calculate_v_index(rot)
            if temp_result["models"][rot]["v_index"] > best_v:
                temp_result["best_model"] = rot
                best_v = temp_result["models"][rot]["v_index"]
            if temp_result["models"][rot]["v_index"] < worst_v:
                temp_result["worst_model"] = rot
                worst_v = temp_result["models"][rot]["v_index"]
        results.append(temp_result)

        iter_count += 1
        ave_elapsed = (time.perf_counter() - start_time) / iter_count
        time_left = np.round(ave_elapsed * (total_iters - iter_count) / 3600, 2)
        print(f"Iteration {iter_count} of {total_iters} done. Estimated time left: {time_left} hours.")

end_time = time.perf_counter()
elapsed_time = (end_time - start_time) / 3600
print(f"Execution time: {elapsed_time:.2f} hours")

fname = (f"./outputs/Simulation Results - Contaminate Perturbation - Pert {', '.join(map(str, pert_levels))} "
         f"- Others All.pickle")
with open(fname, "wb") as file:
    pickle.dump(results, file)
    file.close()
