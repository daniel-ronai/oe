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

p = get_params(df, 2)

injection_days = [2, 3, 6]
doses = [0.5, 1.0, 1.5]
sigma = 0.1  #gauss-görbe

def u(t):
    result = 0.0
    for nap, dozis in zip(injection_days, doses):
        result += dozis * np.exp(-((t - nap)**2) / (2 * sigma**2))
    return result

def rendszer(t, x, p):
    x1, x2, x3, x4 = x  

    dx1dt = (p["a"] - p["n"]) * x1 - p["b"] * (x1 * x3) / (p["ED50"] + x3)
    dx2dt = p["n"] * x1 + p["b"] * (x1 * x3) / (p["ED50"] + x3) - p["w"] * x2
    dx3dt = -(p["c"] + p["k1"]) * x3 + p["k2"] * x4 + u(t)
    dx4dt = p["k1"] * x3 - p["k2"] * x4

    return [dx1dt, dx2dt, dx3dt, dx4dt]

x0 = [p["x10"], 0, 0, 0]

t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

sol = solve_ivp(rendszer, t_span, x0, t_eval=t_eval, args=(p,), method='RK45')

y = sol.y[0] + sol.y[1]

plt.plot(sol.t, sol.y[0], color="green", label="élő tumorsejt")
plt.plot(sol.t, sol.y[1], color="red", label="halott tumorsejt")
plt.plot(sol.t, y, color="blue", label="teljes tumor")

plt.title("Tumor növekedés")
plt.xlabel("Idő [nap]")
plt.ylabel("Tumor térfogat [mm³]")
plt.legend()
plt.show()