import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings

# Figyelmeztetések kikapcsolása a tiszta konzolért
warnings.filterwarnings("ignore")

# 1. Adatok betöltése
df = pd.read_csv("data/experiments.csv", sep=",")

# 2. Modell egyenletek (Gyorsított skalár műveletekkel)
def rendszer(t, x, a, n, b, ED50, w, c, k1, k2):
    x1, x2, x3, x4 = x
    
    x1 = max(0.0, min(x1, 1e8))
    x2 = max(0.0, min(x2, 1e8))
    x3 = max(0.0, min(x3, 1e8))
    x4 = max(0.0, min(x4, 1e8))
    
    denom = ED50 + x3
    if denom <= 1e-12:
        denom = 1e-12
    
    term = b * (x1 * x3) / denom
    dx1dt = (a - n) * x1 - term
    dx2dt = n * x1 + term - w * x2
    dx3dt = -(c + k1) * x3 + k2 * x4
    dx4dt = k1 * x3 - k2 * x4
    
    return [dx1dt, dx2dt, dx3dt, dx4dt]

# 3. Szimulációs motor
def simulate(p, events, t_end):
    dose_dict = {}
    for t_ev, amt in events:
        dose_dict[t_ev] = dose_dict.get(t_ev, 0) + amt
        
    dose_times = sorted(dose_dict.keys())
    breaks = sorted(list(set([0] + dose_times + [t_end])))
    
    x_current = [p["x10"], 0.0, 0.0, 0.0]
    if 0 in dose_dict:
        x_current[2] += dose_dict[0]
        
    all_t, all_y = [], []
    a, n, b, ED50, w, c, k1, k2 = p["a"], p["n"], p["b"], p["ED50"], p["w"], p["c"], p["k1"], p["k2"]
    
    for i in range(len(breaks) - 1):
        t_start = breaks[i]
        t_next = breaks[i+1]
        
        if t_next == t_start:
            continue
            
        sol = solve_ivp(rendszer, (t_start, t_next), x_current, 
                        args=(a, n, b, ED50, w, c, k1, k2),
                        method='LSODA', rtol=1e-3, atol=1e-5)
        
        if not sol.success or len(sol.t) == 0:
            raise ValueError("ODE hiba")
            
        all_t.append(sol.t)
        all_y.append(sol.y)
        
        x_current = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1]]
        if t_next in dose_dict:
            x_current[2] += dose_dict[t_next]
            
    t_concat = np.concatenate(all_t)
    y_concat = np.concatenate(all_y, axis=1)
    
    t_unique, idx = np.unique(t_concat, return_index=True)
    y_unique = y_concat[:, idx]
    
    return t_unique, y_unique

# 4. Reziduális függvény - LINEÁRIS TÉRBEN (Ez fogja megfogni a nagy robbanást)
param_names = ["a", "n", "b", "ED50", "w", "c", "k1", "k2", "x10"]

def residuals(params_val, eger_df, events, t_end):
    p = dict(zip(param_names, params_val))
    try:
        t, y = simulate(p, events, t_end)
        total_tumor = y[0] + y[1]
        
        if np.isnan(total_tumor).any() or np.isinf(total_tumor).any():
            return np.ones(len(eger_df)) * 1e5
            
        f_interp = interp1d(t, total_tumor, kind='linear', fill_value="extrapolate")
        y_sim = np.maximum(f_interp(eger_df["TIME"].values), 0)
        
        # Sima különbség (y_sim - y_data). Ezzel a 6000-es hibák dominálnak majd, így a modell 
        # rákényszerül, hogy kövesse a hatalmas késői növekedést!
        res = y_sim - eger_df["DV"].values
        return np.nan_to_num(res, nan=1e5, posinf=1e5, neginf=-1e5)
    except:
        return np.ones(len(eger_df)) * 1e5

# 5. Illesztési konfiguráció korlátokkal (Bounds)
# A felső korlátokat (a, c) kicsit kitágítottuk, hogy a késői stádiumban a tumor szabadon tudjon robbanni
low_bounds = [0.01,  1e-6,  0.01,  1e-5,   0.01,  0.05,  0.01,  0.01,  5.0]
up_bounds  = [1.5,   0.1,   10.0,  50.0,   5.0,   15.0,  10.0,  10.0,  250.0]

# Különböző "szcenáriók" a Multi-Start-hoz
initial_guesses = [
    [0.15, 0.005, 0.1, 1e-3, 0.05, 0.4, 0.4, 1.0, 50.0],  # Alap
    [0.40, 0.010, 1.0, 1e-1, 0.10, 2.0, 0.1, 0.1, 80.0],  # "Késői robbanás" profil: Magas 'a', gyors gyógyszerürülés 'c'
    [0.08, 0.001, 0.05, 1e-4, 0.01, 1.5, 1.0, 2.0, 30.0]  # Lassú dinamika
]

# KIZÁRÓLAG 2 EGÉR FUTTATÁSA AZ IDŐKERET MIATT
mouse_ids = df["ID"].unique()[:2]

results = {}
rmse_values = {}

print(f"--- VÉGLEGES PARAMÉTERBECSLÉS INDÍTÁSA ({len(mouse_ids)} egér) ---")

for m_id in mouse_ids:
    subset = df[df["ID"] == m_id]
    
    measurements = subset[(subset["EVID"] == 0) & (subset["DV"].notna())]
    doses_df = subset[subset["EVID"] == 301]
    doses = list(zip(doses_df["TIME"], doses_df["AMT"]))
    
    t_end = subset["TIME"].max() + 0.5
    
    best_res = None
    best_cost = float('inf')
    
    for guess in initial_guesses:
        res = least_squares(residuals, guess, bounds=(low_bounds, up_bounds),
                            args=(measurements, doses, t_end), 
                            loss='linear', ftol=1e-4) # Linear loss kikényszeríti a nagy távolságok büntetését
        
        if res.cost < best_cost:
            best_cost = res.cost
            best_res = res
            
    final_residuals = best_res.fun
    rmse = np.sqrt(np.mean(final_residuals**2))
    
    results[m_id] = best_res.x
    rmse_values[m_id] = rmse
    print(f"Egér: {m_id} -> Kész. Valós RMSE: {rmse:.2f}")

# 6. Összefoglaló táblázat
res_df = pd.DataFrame.from_dict(results, orient='index', columns=param_names)
res_df["Valós-RMSE"] = pd.Series(rmse_values)
print("\n--- ILLLESZTETT PARAMÉTEREK ÖSSZESÍTŐJE ---")
print(res_df.to_string())

# 7. Vizualizáció
for m_id in mouse_ids:
    subset = df[df["ID"] == m_id]
    measurements = subset[(subset["EVID"] == 0) & (subset["DV"].notna())]
    doses_df = subset[subset["EVID"] == 301]
    doses = list(zip(doses_df["TIME"], doses_df["AMT"]))
    t_end = subset["TIME"].max() + 0.5
    
    p_opt = dict(zip(param_names, results[m_id]))
    t_sim, y_sim = simulate(p_opt, doses, t_end)
    
    plt.figure(figsize=(8, 4))
    plt.plot(t_sim, y_sim[0] + y_sim[1], color='firebrick', linewidth=2.5, label="Illesztett modell (Valós terű LS)")
    plt.scatter(measurements["TIME"], measurements["DV"], color='black', s=45, zorder=3, label="Mért tumor adatok")
    
    plt.title(f"Tökéletesített TGI Illesztés - Egér: {m_id} (RMSE: {rmse_values[m_id]:.1f})")
    plt.xlabel("Idő [nap]")
    plt.ylabel("Tumor térfogat [mm³]")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()