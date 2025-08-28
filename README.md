# Composite-Adaptive Lyapunov-Based Deep Neural Network Control 🚀

Proveably-stable deep learning for nonlinear control. This repo implements composite adaptive control with a deep neural network regressor, evaluated on two classic benchmarks: a 2‑DOF twolink robot and a 6state underwater vehicle (UUV) with intermittent feedback.

- Composite adaptation = tracking-error + prediction-error driven weight updates
- Lyapunov-based stability with projection and adaptive gain updates
- Head-to-head comparisons with observer-based, PID/PD, and nonlinear MPC baselines

---

## Whats inside

### 1) TwoLink Robot (folder: Two Link/)
Entry point: Two Link/Simulate.m

- States: n = 2 (positions and velocities)
- DNN: k = 10 hidden layers, L = 10 neurons/layer, tanh activations
- Controllers compared:
  - Tracking-error-based adaptive DNN ("Traditional")
  - Composite adaptive DNN (tracking + DNN prediction error)
  - Observer-based (disturbance estimation)
  - PID
  - Nonlinear MPC (fmincon)
- Plots: tracking error and control effort with zoomed insets; summary tables with RMS metrics

Key implementation files:
- DNNSim.m  main simulation with online weight adaptation and Lyapunov-safe updates
- DNNGrad.m, LayerGrad.m  fast vectorized regressor and Jacobians for the DNN
- 	wo_link.m, dynamics.m  plant dynamics and input matrix g(x)
- proj.m  parameter projection to keep estimates bounded
- DNN_eval.m  RMS model error evaluation on random test points

### 2) Underwater Vehicle, Intermittent Feedback (folder: UUV/)
Entry point: UUV/Simulate.m

- States: n = 6 (3 linear + 3 angular)
- DNN: k = 5, L = 5
- Intermittent feedback: configurable feedback denied zones highlighted in plots via gray patches (see intervals in Simulate.m)
- Controllers compared: Traditional vs Composite DNN, Observer-based, Nonlinear PD, Nonlinear MPC
- Plots: linear/angular tracking errors, control norms, and model error (DNN vs observer disturbance estimate)

Key implementation files:
- DNNSimIF.m  intermittent-feedback variant of the DNN controller
- MPCSimIF.m, ObserverSimIF.m, PIDSimIF.m  baselines
- DNNGrad.m, LayerGrad.m  shared DNN gradient machinery
- Plot3D_9.m  helper for 3D visualization

---

## Why composite adaptation?
Classic adaptive controllers rely only on tracking error. Here, we augment with the DNNs prediction error (between the learned model and inferred dynamics), yielding faster, more accurate learning while maintaining Lyapunov stability. The adaptation law uses:
- Online regressor gradients from DNNGrad.m
- Projection proj.m to keep parameters bounded
- Adaptive learning-rate matrix Gamma with Riccati-like update and forgetting factor

Noise robustness is included via SNR-based measurement noise; the UUV case also covers intermittent feedback.

---

## Quick start

1) Requirements
- MATLAB (R2020b or newer recommended)
- Optimization Toolbox (for MPC via mincon)

2) Run the demos
- TwoLink: open Two Link/Simulate.m in MATLAB and click Run
- UUV: open UUV/Simulate.m and click Run

Optional: for reproducible random initializations, add ng(0) near the top of Simulate.m.

Outputs include comparison plots and a small table of RMS metrics printed to the console.

---

## Repo layout

- Two Link/  twolink robot simulations (DNN, gradients, dynamics, controllers)
- UUV/  underwater vehicle with intermittent feedback (IF) variants
- .gitignore  MATLAB/Simulink and editor artifacts
- LICENSE  MIT License

---

## Citing
If you use this code, please cite the repository. A paper citation will be added when available.

`	ext
@software{patil2025_composite_adaptive_dnn_control,
  author  = {Omkar S. Patil},
  title   = {Composite-Adaptive Lyapunov-Based Deep Neural Network Control},
  year    = {2025},
  url     = {https://github.com/patilomkarsudhir/Composite-Adaptive-Lyapunov-Based-Deep-Neural-Network},
  version = {main}
}
`

---

## License
MIT  see LICENSE.

## Contact
Open an issue or reach out at patilomkarsudhir@gmail.com.
