# Physics-Informed Neural Network for the 1D Diffusion Equation

A Physics-Informed Neural Network (PINN) implementation in TensorFlow for solving the one-dimensional inhomogeneous diffusion equation

\[
\frac{\partial u}{\partial t}
=
\frac{\partial^2 u}{\partial x^2}
+
\cos(x)
\]

on the domain

\[
x \in [0,\pi], \quad t \in [0,1]
\]

without using training data. The network learns the solution solely from the governing physical equations and constraints. 【2-b5060c】

---

## Problem Statement

The considered PDE is

```math
u_t = u_{xx} + \cos(x)
```

subject to

```math
u(x,0)=0
```

```math
u(0,t)=1-e^{-t}
```

```math
u(\pi,t)=-1+e^{-t}
```

The analytical solution is

```math
u(x,t)=\cos(x)-\cos(x)e^{-t}
```

which is used for validation of the trained PINN model. 【2-b5060c】

---

## Project Goals

- Implement a Physics-Informed Neural Network from scratch using TensorFlow
- Solve the 1D diffusion equation without supervised training data
- Investigate the influence of hyperparameters
- Compare PINN predictions against the analytical solution
- Analyze solution accuracy and error distribution 【2-b5060c】

---

## Methodology

Instead of learning from labeled data, the network is trained by minimizing a physics-based loss function consisting of:

### PDE Residual Loss

The governing equation is enforced through the residual

```math
R(x,t)=u_t-u_{xx}-\cos(x)
```

and the corresponding mean squared error.

### Initial Condition Loss

```math
u(x,0)=0
```

### Boundary Condition Loss

```math
u(0,t)=1-e^{-t}
```

```math
u(\pi,t)=-1+e^{-t}
```

The total loss is defined as

```math
L = L_{PDE} + L_{IC} + L_{BC}
```

In the final implementation, the boundary condition loss is weighted more heavily to improve accuracy near the domain boundaries. 【2-b5060c】【1-23756a】

---

## Neural Network Architecture

The PINN is implemented as a fully connected feed-forward neural network.

### Inputs

- Spatial coordinate `x`
- Time coordinate `t`

### Output

- Approximation of the solution `u(x,t)`

### Selected Architecture

```text
Input Layer (2)
        ↓
Dense(32, tanh)
        ↓
Dense(32, tanh)
        ↓
Dense(32, tanh)
        ↓
Output Layer (1)
```

Configuration:

| Parameter | Value |
|------------|---------|
| Hidden Layers | 3 |
| Neurons per Layer | 32 |
| Activation Function | tanh |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Epochs | 20000 |
| Trainable Parameters | 2240 |

The architecture was selected after a hyperparameter study comparing different learning rates, network depths, and activation functions. The `tanh` activation significantly outperformed ReLU for this problem. 【2-b5060c】

---

## Training Data Generation

The model uses randomly sampled collocation points:

| Point Type | Count |
|------------|---------|
| Interior Points | 1000 |
| Initial Condition Points | 200 |
| Boundary Points | 200 |

Uniform random sampling is used throughout the domain. 【2-b5060c】【1-23756a】

---

## Automatic Differentiation

TensorFlow's `GradientTape` is used to compute

- First-order derivatives
  - `u_t`
  - `u_x`
- Second-order derivative
  - `u_xx`

This allows direct evaluation of the PDE residual without numerical differentiation. 【1-23756a】

---

## Results

After training, the network successfully reconstructs the analytical solution.

Validation includes:

- Solution contour plots
- Error contour plots
- Comparison of predicted and exact solution at `t = 1`
- Three-dimensional error surface visualization
- Average error computation

Error analysis indicates that the largest deviations occur near the spatial boundaries and at later times. Weighting the boundary-condition loss by a factor of five reduces the maximum error. 【2-b5060c】【1-23756a】

---

## Repository Structure

```text
.
├── PINN_1D_Diffusion.py
├── PIML_Report.pdf
├── complete_validation_bc_weighting.png
├── error_surface.png
├── solution_surface.png
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/PINN-1D-Diffusion.git
cd PINN-1D-Diffusion
```

Install dependencies

```bash
pip install tensorflow numpy matplotlib keras
```

---

## Usage

Run the training script:

```bash
python PINN_1D_Diffusion.py
```

The script will:

1. Generate collocation points
2. Train the PINN
3. Evaluate the learned solution
4. Compute error metrics
5. Generate validation figures

---

## Example Outputs

The script generates:

- Predicted solution contour
- Exact vs predicted solution comparison
- Global error contour
- Error surface plot
- Solution surface plot

Example:

```text
average error value: 1.0e-03
```

(depending on initialization and hardware). 【1-23756a】

---

## Key Learning Outcomes

This project demonstrates:

- Physics-Informed Machine Learning (PIML)
- Solving PDEs using PINNs
- Automatic differentiation
- Constraint-based learning
- Hyperparameter sensitivity in PINNs
- Validation against analytical solutions 【2-b5060c】

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs.
2. TensorFlow Documentation
3. Project report: *Solving the 1D Linear Diffusion Equation with PINNs* 【2-b5060c】

---

## Author

Elvis Johannes Sattler

Course:
**Physics-Informed Machine Learning (WiSe 24/25)** 【2-b5060c】
