# Urban Air Quality Forecasting via Advection-Diffusion Numerical Methods

**Course:** Numerical Methods for Computational Mathematics  
**Institution:** Cluster Innovation Centre, University of Delhi  
**Authors:** Anjali Yadav · Sajal Samveg  
**Mentors:** Prof. Sonam Tanwar · Prof. Nirmal Yadav

---

## Overview

A hybrid physics-AI pipeline — a Spatio-Temporal Digital Twin — for 
forecasting neighborhood-level PM2.5 pollution plumes in real time over 
the Anand Vihar corridor, Delhi. The core insight: physics-informed 
synthetic data can train a neural network to internalize fluid dynamics, 
giving the spatial accuracy of CFD at the speed of deep learning.

## Methodology

**1. Data Preprocessing**  
43,825 hourly PM2.5/NO2 samples (2020–2025) from Anand Vihar CPCB station.  
Piecewise Cubic Spline interpolation (C² continuity) for missing data imputation — 
RMSE: 19.03 µg/m³.

**2. Physics Engine (Crank-Nicolson PDE Solver)**  
Solved the Advection-Diffusion PDE using an implicit Crank-Nicolson 
finite-difference scheme (unconditionally stable).  
- CFL = 1.73 → explicit solver would have diverged; C-N was mathematically necessary  
- Péclet = 2.70 → advection-dominated, correctly capturing wind-driven plume behavior  
- Solved via Thomas Algorithm on a tridiagonal matrix at each timestep

**3. GIS Integration (OSMnx + Graph Theory)**  
Extracted Anand Vihar road network via OSMnx. Applied Edge Betweenness 
Centrality to identify top-15% high-congestion road segments as dynamic 
emission sources.

**4. Deep Learning (ConvLSTM)**  
Trained a ConvLSTM on Lagrangian-to-Eulerian KDE synthetic heatmaps.  
The model successfully learned wind-advection dynamics, forecasting 2D 
plume positions with structural fidelity — surpassing the spatial ceiling 
of the 1D LSTM baseline (R² = 0.68).

## Key Results

| Model | Metric | Score |
|-------|--------|-------|
| 1D LSTM baseline | R² | 0.68 |
| 1D LSTM baseline | RMSE | 16.35 µg/m³ |
| ConvLSTM | 2D plume forecast | Structural fidelity ✓ |

## Tech Stack

- **MATLAB** — Crank-Nicolson PDE solver, Hovmöller diagrams
- **Python** — NumPy, TensorFlow (ConvLSTM), OSMnx, GeoPandas, Matplotlib
- **Data Source** — CPCB Open Data (cpcb.nic.in)

## How to Run

```bash
pip install -r requirements.txt
```

1. Run `preprocessing.py` to perform spline imputation and generate source terms  
2. Run `SpatioTemporal_PDE_Heatmap.m` in MATLAB to generate synthetic heatmaps  
3. Run `convlstm_training.py` to train and evaluate the ConvLSTM  

## References

- Crank & Nicolson (1947) — implicit finite-difference scheme  
- Boeing (2017) — OSMnx street network extraction  
- Shi et al. (2015) — ConvLSTM for spatiotemporal forecasting  
- CPCB Open Data (2020–2025)

## Report

Full project report available [here](YOUR_GOOGLE_DRIVE_LINK).
