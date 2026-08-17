import numpy as np
import pandas as pd
import pickle
import os

file_names = []
for entry in os.scandir("./outputs"):
    if entry.is_file():
        file_names.append(entry.name)

pickles = []
for file_name in file_names:
    with open(f"./outputs/{file_name}", "rb") as file:
        temp = pickle.load(file)
        pickles.append(temp)
        file.close()

loadings_types = []
specific_variances = []
perturbation_types = []
perturbation_levels = []
best_models = []
worst_models = []
actual_rotations_for_priorimax = []
priorimax_rmses = []
priorimax_vs = []
unrotated_rmses = []
unrotated_vs = []
varimax_rmses = []
varimax_vs = []
equamax_rmses = []
equamax_vs = []
quartimax_rmses = []
quartimax_vs = []
for batch in pickles:
    for row in batch:
        loadings_types.append(row["loadings_type"])
        specific_variances.append(row["specific_variance"])
        perturbation_types.append(row["perturbation_type"])
        perturbation_levels.append(row["perturbation_level"])
        best_models.append(row["best_model"])
        worst_models.append(row["worst_model"])
        actual_rotations_for_priorimax.append(row["actual_rotation_for_priorimax"])
        priorimax_rmses.append(row["models"]["priorimax"]["rmse"])
        priorimax_vs.append(row["models"]["priorimax"]["v_index"])
        unrotated_rmses.append(row["models"]["unrotated"]["rmse"])
        unrotated_vs.append(row["models"]["unrotated"]["v_index"])
        varimax_rmses.append(row["models"]["varimax"]["rmse"])
        varimax_vs.append(row["models"]["varimax"]["v_index"])
        equamax_rmses.append(row["models"]["equamax"]["rmse"])
        equamax_vs.append(row["models"]["equamax"]["v_index"])
        quartimax_rmses.append(row["models"]["quartimax"]["rmse"])
        quartimax_vs.append(row["models"]["quartimax"]["v_index"])

df = pd.DataFrame({
    "loadings_type": loadings_types,
    "specific_variance": specific_variances,
    "perturbation_type": perturbation_types,
    "perturbation_level": perturbation_levels,
    "best_model": best_models,
    "worst_model": worst_models,
    "actual_rotation_for_priorimax": actual_rotations_for_priorimax,
    "priorimax_rmse": priorimax_rmses,
    "priorimax_v": priorimax_vs,
    "unrotated_rmse": unrotated_rmses,
    "unrotated_v": unrotated_vs,
    "varimax_rmse": varimax_rmses,
    "varimax_v": varimax_vs,
    "equamax_rmse": equamax_rmses,
    "equamax_v": equamax_vs,
    "quartimax_rmse": quartimax_rmses,
    "quartimax_v": quartimax_vs
})


def get_mean_variance(r):
    size = r["specific_variance"]

    if size == "small":
        return 0.2
    elif size == "medium":
        return 0.4
    elif size == "large":
        return 0.6


df["mean_specific_variance"] = df.apply(get_mean_variance, axis=1)

df["priorimax_scaled_rmse"] = df["priorimax_rmse"] / np.sqrt((2 / 3) * (1 - df["mean_specific_variance"]))
df["unrotated_scaled_rmse"] = df["unrotated_rmse"] / np.sqrt((2 / 3) * (1 - df["mean_specific_variance"]))
df["varimax_scaled_rmse"] = df["varimax_rmse"] / np.sqrt((2 / 3) * (1 - df["mean_specific_variance"]))
df["equamax_scaled_rmse"] = df["equamax_rmse"] / np.sqrt((2 / 3) * (1 - df["mean_specific_variance"]))
df["quartimax_scaled_rmse"] = df["quartimax_rmse"] / np.sqrt((2 / 3) * (1 - df["mean_specific_variance"]))
df.to_csv("./outputs/simulation_results.csv", index=False)
