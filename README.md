# HeatFlowClean  
A clean, modern, and mathematically rigorous Python implementation of the **1‑D and 2‑D Heat Equation** using finite‑difference numerical methods.  
This project is designed to be **educational**, **modular**, and **recruiter‑friendly**, showcasing scientific Python, numerical simulation, and visualization.

---

# 🌡️ Overview

The **heat equation** models how temperature diffuses over time:

\[
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
\]

This repository implements:

- 1‑D and 2‑D finite‑difference solvers  
- Explicit Euler time stepping  
- Stability‑checked simulation  
- Visualization tools (line plots + heatmaps)  
- Clean examples and unit tests  

---

# 🧮 Mathematical Derivation

## 1. Spatial Discretization (1‑D)

For a rod of length \(L\):

\[
x_i = i\Delta x,\quad i = 0,1,\dots,N
\]

The second derivative is approximated by:

\[
\frac{\partial^2 u}{\partial x^2} \approx 
\frac{u_{i+1} - 2u_i + u_{i-1}}{(\Delta x)^2}
\]

---

## 2. Time Discretization (Forward Euler)

\[
\frac{\partial u}{\partial t} \approx 
\frac{u_i^{n+1} - u_i^n}{\Delta t}
\]

---

## 3. Combined Update Rule

\[
u_i^{n+1} = u_i^n + 
\alpha \frac{\Delta t}{(\Delta x)^2}
\left( u_{i+1}^n - 2u_i^n + u_{i-1}^n \right)
\]

Define:

\[
r = \alpha \frac{\Delta t}{(\Delta x)^2}
\]

Then:

\[
u_i^{n+1} = u_i^n + r \left( u_{i+1}^n - 2u_i^n + u_{i-1}^n \right)
\]

---

## 4. Stability Condition

Explicit Euler requires:

\[
r \le \frac{1}{2}
\]

This ensures the simulation does not blow up.

---

## 5. 2‑D Heat Equation

\[
\frac{\partial u}{\partial t} = 
\alpha \left(
\frac{\partial^2 u}{\partial x^2} +
\frac{\partial^2 u}{\partial y^2}
\right)
\]

Discretized:

\[
u_{i,j}^{n+1} = u_{i,j}^n + r \left(
u_{i+1,j}^n + u_{i-1,j}^n +
u_{i,j+1}^n + u_{i,j-1}^n -
4u_{i,j}^n
\right)
\]

---

# 📁 Project Structure
