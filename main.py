import pandas as pd
import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 1. Setup
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/plots'):
    os.makedirs('results/plots')

df = pd.read_csv("data/experiments.csv", sep=",")
param_names = ["a", "n", "b", "ED50", "w", "c", "k1", "k2", "x10"]
mouse_ids = df["ID"].unique()

# 2. ODE Solver & Simulation
def rendszer(t, x, a, n, b, ED50, w, c, k1, k2):
    x = np.maximum(x, 1e-9)
    x1, x2, x3, x4 = x
    denom = ED50 + x3
    term = b * (x1 * x3) / denom
    return [(a - n) * x1 - term, n * x1 + term - w * x2, -(c + k1) * x3 + k2 * x4, k1 * x3 - k2 * x4]

def simulate(p, events, t_end):
    dose_dict = {t: amt for t, amt in events}
    breaks = sorted(list(set([0] + list(dose_dict.keys()) + [t_end])))
    x = [p["x10"], 0.0, 0.0, 0.0]
    if 0 in dose_dict: x[2] += dose_dict[0]

    all_t, all_y = [], []
    for i in range(len(breaks) - 1):
        sol = solve_ivp(rendszer, (breaks[i], breaks[i+1]), x,
                        args=(p["a"], p["n"], p["b"], p["ED50"], p["w"], p["c"], p["k1"], p["k2"]),
                        method='LSODA', rtol=1e-4, atol=1e-6)
        all_t.append(sol.t); all_y.append(sol.y)
        x = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1]]
        if breaks[i+1] in dose_dict: x[2] += dose_dict[breaks[i+1]]
    return np.concatenate(all_t), np.concatenate(all_y, axis=1)

def residuals(params_val, eger_df, events, t_end):
    p = dict(zip(param_names, params_val))
    t, y = simulate(p, events, t_end)
    f = interp1d(t, y[0] + y[1], kind='linear', fill_value="extrapolate")
    y_sim = np.maximum(f(eger_df["TIME"].values), 0)
    return y_sim - eger_df["DV"].values

# 3. Main Loop
all_results = []
initial_guess = [0.1, 0.01, 0.1, 1.0, 0.5, 1.0, 0.5, 0.5, 50.0]
lower_b = [0.0] * 9
upper_b = [np.inf] * 9

print(f"--- STARTING BATCH ESTIMATION: {len(mouse_ids)} mice ---")
for m_id in mouse_ids:
    subset = df[df["ID"] == m_id]
    m_data = subset[(subset["EVID"] == 0) & (subset["DV"].notna())]
    doses = list(zip(subset[subset["EVID"] == 301]["TIME"], subset[subset["EVID"] == 301]["AMT"]))

    guess = initial_guess.copy()
    guess[param_names.index("x10")] = max(m_data["DV"].iloc[0], 1.0)

    res = least_squares(residuals, guess, bounds=(lower_b, upper_b),
                        args=(m_data, doses, subset["TIME"].max() + 0.5),
                        x_scale='jac', max_nfev=5000)

    rmse = np.sqrt(np.mean(res.fun**2))

    # Store result
    row = {"ID": m_id, "RMSE": rmse}
    row.update(dict(zip(param_names, res.x)))
    all_results.append(row)

    print(f"Finished {m_id} (RMSE: {rmse:.2f})")

    # 4b. Plot: measured points (subsampled for confidentiality) vs fitted curve
    fitted_p = dict(zip(param_names, res.x))
    t_fit, y_fit = simulate(fitted_p, doses, subset["TIME"].max() + 0.5)
    y_fit_total = y_fit[0] + y_fit[1]  # x1 + x2 = total tumor burden

    plt.figure(figsize=(7, 5))
    plt.scatter(m_data["TIME"], m_data["DV"], color="black", label="Measured", zorder=3)
    plt.plot(t_fit, y_fit_total, color="crimson", label="Fitted curve", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Tumor burden")
    plt.title(f"Subject {m_id} — model fit (RMSE: {rmse:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/fit_{m_id}.png", dpi=150)
    plt.close()

# 5. Save to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("results/parameters.csv", index=False)
print("\n--- DONE. Parameters saved to 'results/parameters.csv' ---")