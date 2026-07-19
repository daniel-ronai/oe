# Tumor Growth & Dosing Response Modeling

## Overview
This project models how tumors respond to drug dosing over time, using a 
system of ordinary differential equations (ODEs) fitted to real experimental 
mouse data. It estimates key biological parameters — growth rate, drug 
effect, decay, and distribution — individually for each subject, enabling 
per-subject dose-response analysis.

Developed as part of Óbuda Excellence Tehetséggondozó Ösztöndíjprogram.

## What it does
- Simulates tumor + drug dynamics using a 4-compartment ODE system
- Fits model parameters to experimental time-series data via nonlinear 
  least squares (`scipy.optimize.least_squares`)
- Handles multiple dosing events per subject
- Outputs fitted parameters and fit quality (RMSE) per subject to CSV

## Result
The fitted model was validated against real experimental dosing data on a 
per-subject basis, tracking the dominant late-stage tumor growth trend closely. 
The underlying experimental data and fitted results are not included in this 
repository, as the dataset is confidential.

**Known limitation:** the model doesn't fully capture a smaller early-stage 
rise-and-dip transient seen in some subjects before the dominant growth phase 
takes over. This is a structural limitation, not a fitting issue — the current 
4-compartment system is built around a single dominant growth mode, so it 
can't reproduce a rise-fall-plateau-rise pattern regardless of how well the 
parameters are tuned. Capturing that transient would require a richer 
compartment structure. The late-stage fit — which reflects the clinically 
more significant growth trend — is what this version prioritizes.

## Output
- `results/parameters.csv` — fitted parameters (a, n, b, ED50, w, c, k1, k2, x10) 
  and RMSE per subject (not included in this repository — confidential)

## Tech
Python · SciPy (`solve_ivp`, `least_squares`) · pandas · NumPy

## Running it
```bash
pip install -r requirements.txt
python main.py
```

Requires `data/experiments.csv` (not included — confidential experimental data).