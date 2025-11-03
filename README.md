# Hermite Neural Operator (HNO)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official source code, datasets, and pre-trained models for the paper:

> **Hermite Neural Operator for Solving Partial Differential Equations on Unbounded Domains**
>
> *Ruijie Bai, Ziyuan Liu, Xiangyao Wu, Yuhang Wu, and Xu Qian*
>
> *(Link to published paper will be added upon acceptance)*

## 📖 Overview

Solving partial differential equations (PDEs) on unbounded domains is a fundamental challenge in scientific computing.
1.  **Classical methods** rely on complex Artificial Boundary Conditions (ABCs) to approximate an infinite domain with a finite one.
2.  [cite_start]**Modern neural operators**, like the Fourier Neural Operator (FNO), are restricted by their Fourier basis, which imposes a non-physical periodicity on the problem[cite: 8].

The **Hermite Neural Operator (HNO)** is a novel framework that solves this by integrating the classical Hermite spectral method into a neural operator architecture. [cite_start]By using Hermite functions as the basis, HNO inherently enforces the correct far-field decay conditions required for unbounded problems, completely avoiding periodic artifacts[cite: 9, 70].

[cite_start]As demonstrated in the paper, HNO achieves significantly higher accuracy and robust off-grid extrapolation on challenging benchmarks, including the Nonlinear Schrödinger (NLS) and Heat equations, when compared to FNO, POD-DeepONet, and LSM[cite: 12, 280, 298].

## 🚀 Getting Started

### 1. Installation

1.  Clone this repository. **You must have Git LFS installed** to download the large model and data files.
    ```bash
    # Install Git LFS ([https://git-lfs.github.com/](https://git-lfs.github.com/)) if you don't have it
    # git lfs install
    
    git clone [https://github.com/lanyukeren/HNO.git](https://github.com/lanyukeren/HNO.git)
    cd HNO
    ```
    *If you have already cloned the repository without LFS, run `git lfs pull` to download the large files.*

2.  (Recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install the required packages.
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
* [cite_start]`/Hermite/data/`: All datasets (`.mat` files) used in the paper[cite: 238, 278, 296].
* `/Hermite/model/`: All pre-trained models (`.pkl` files) used in the paper.

### 3. Running the Code

To train a new model from scratch, navigate to one of the experiment directories and run the main Python script.

**Example: Train the 2D Heat Equation model**
```bash
cd Hermite/heat2d/
python hno2dheat.py
