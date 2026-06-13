# Urban Air Quality Forecasting — Anand Vihar, Delhi

Hybrid physics-AI pipeline that combines a **Crank-Nicolson PDE solver** 
with a **ConvLSTM** to forecast 2D PM2.5 pollution plumes at 
neighbourhood level — overcoming the spatial blindness of standard 1D models.

## What it does
- Imputes 43,825 hourly PM2.5/NO2 samples via Piecewise Cubic Splines (RMSE: 19.03 µg/m³)
- Solves the Advection-Diffusion PDE (CFL=1.73 — explicit solvers diverge here)
- Maps road congestion to emission sources using Edge Betweenness Centrality on OSMnx graphs
- Trains a ConvLSTM on physics-generated KDE heatmaps to forecast plume positions in 2D

## Results
1D LSTM ceiling: R² = 0.68. ConvLSTM successfully learns wind-advection dynamics 
and predicts future plume structure with spatial fidelity.

## Stack
MATLAB · Python · TensorFlow · OSMnx · GeoPandas · NumPy

## Report
[Full report](https://drive.google.com/file/d/10fZUjS1VcGnnRGx0QtFYJIGwdH3K4ecX/view)
