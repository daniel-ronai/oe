import pandas as pd
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/params.csv", sep=";")
df = df[["ID", "variable", "value"]]

def get_params(df, eger_id):
    eger = df[df["ID"] == eger_id]
    params = {}
    for _, sor in eger.iterrows():
        params[sor["variable"]] = sor["value"]
    return params

p = get_params(df, 6)   # hanyas PLD
p["n"] = 0.05

injection_days = [1, 3, 6]
doses = [0.2, 0.4, 0.6]

def rendszer(t, x, p):  
    x1, x2, x3, x4 = x

    dx1dt = (p["a"] - p["n"]) * x1 - p["b"] * (x1 * x3) / (p["ED50"] + x3)
    dx2dt = p["n"] * x1 + p["b"] * (x1 * x3) / (p["ED50"] + x3) - p["w"] * x2
    dx3dt = -(p["c"] + p["k1"]) * x3 + p["k2"] * x4
    dx4dt = p["k1"] * x3 - p["k2"] * x4

    return [dx1dt, dx2dt, dx3dt, dx4dt]

def simulate(p, injection_days, doses, t_end=14):
    all_t = []
    all_y = []

    events = sorted(zip(injection_days, doses))

    t_start = 0
    x_current = [p["x10"], 0, 0, 0]

    breakpoints = [nap for nap, _ in events] + [t_end]
    dose_map = dict(events)

    for bp in breakpoints:
        t_eval = np.linspace(t_start, bp, 300)
        sol = solve_ivp(rendszer, (t_start, bp), x_current,
                        t_eval=t_eval, args=(p,), method='RK45')
        all_t.append(sol.t)
        all_y.append(sol.y)

        if bp < t_end:
            x_current = [sol.y[0][-1],
                         sol.y[1][-1],
                         sol.y[2][-1] + dose_map[bp],
                         sol.y[3][-1]]
        t_start = bp

    t_full = np.concatenate(all_t)
    y_full = np.concatenate(all_y, axis=1)
    return t_full, y_full

t, y = simulate(p, injection_days, doses)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# tumor
ax1.plot(t, y[0], color="green", label="élő tumorsejt")
ax1.plot(t, y[1], color="red", label="halott tumorsejt")
ax1.plot(t, y[0] + y[1], color="blue", label="teljes tumor")
ax1.set_title("Tumor növekedés")
ax1.set_xlabel("Idő [nap]")
ax1.set_ylabel("Tumor térfogat [mm³]")
ax1.legend()

# gyógyszerszint
ax2.plot(t, y[2], color="blue", label="szimulált")

for nap, dozis in zip(injection_days, doses):
    ax2.vlines(nap, 0, dozis, color="gold", linewidth=1.5)
    ax2.plot(nap, dozis, 'o', color="gold", markersize=8,
             label="injekció" if nap == injection_days[0] else "")

ax2.set_title("Gyógyszerszint")
ax2.set_xlabel("Idő [nap]")
ax2.set_ylabel("Gyógyszerszint [mg/kg]")
ax2.legend()

plt.tight_layout()
plt.show()