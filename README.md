# Hermite Neural Operator (HNO)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official source code for the paper:

> **Hermite Neural Operator for Solving Partial Differential Equations on Unbounded Domains**
>
> *Ruijie Bai, Ziyuan Liu, Xiangyao Wu, Yuhang Wu, and Xu Qian*
>
> *(Link to published paper will be added upon acceptance)*

## 📖 Overview

The **Hermite Neural Operator (HNO)** is a novel framework that solves PDEs on unbounded domains by integrating the classical Hermite spectral method into a neural operator architecture. By using Hermite functions as the basis, HNO inherently enforces the correct far-field decay conditions required for unbounded problems, avoiding periodic artifacts.

As demonstrated in the paper, HNO achieves significantly high accuracy and robust off-grid extrapolation on challenging benchmarks, including the Nonlinear Schrödinger (NLS) and Heat equations.

## 🚀 Getting Started

### 1. Install the required packages
    ```bash
    # (A requirements.txt file will be added shortly)
    pip install torch numpy scipy
    ```

### 2. Repository Structure

The code is organized by experiment, located within the `Hermite/` directory:

* `/Hermite/sc1d/`: 1D Nonlinear Schrödinger Equation
    * `hnosc1d.py`: Main training and evaluation script.
    * `hermite_pack.py`: HNO model definition.
* `/Hermite/heat2d/`: 2D Heat Equation
    * `hno2dheat.py`: Main training and evaluation script.
    * `hermitepack2d1012.py`: HNO model definition.
* `/Hermite/sc2d/`: 2D Nonlinear Schrödinger Equation
    * `hnosc2d.py`: Main training and evaluation script.
    * `hermitepack_sc2d.py`: HNO model definition.
* `/Hermite/data/`: All datasets (`.mat` files) used in the paper.


