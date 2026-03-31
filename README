# Depth from Coupled Optical Differentiation — Reproduction

Reproduction of [Luo et al., IJCV 2025](https://doi.org/10.1007/s11263-025-02534-z)

## Overview

This project reproduces the depth estimation pipeline from **Depth from Coupled Optical Differentiation (CoD)**.
The core idea: per-pixel depth can be recovered from a ratio of two optical derivatives of a defocused image.

$$Z = \frac{Z_s}{\rho Z_s - 1 - AZ_s \cdot I_A/I_\rho}$$

## Setup

```bash
pip install numpy scipy matplotlib
```

## Data

Place `.pkl` dataset files in `./data/`:

```
cod/
├── data/
│   ├── Motorized_SingleScene_Allmethods_*.pkl
│   └── ...
├── src/
│   └── depth.py
├── results/
└── README.md
```

Dataset available at [cod.qiguo.org](https://cod.qiguo.org)

## Usage

```bash
cd cod
python3 src/depth.py
```

Results saved to `./results/all_scenes.png`

## Pipeline

```
4 images captured with different optical settings
  I(ρ+Δρ, A),  I(ρ-Δρ, A)    ← optical power pair
  I(ρ, A+ΔA),  I(ρ, A-ΔA)    ← aperture pair
        ↓
Preprocessing
  - Geometric alignment   (magnification shift from ρ change)
  - Brightness correction (aperture area ratio, Eq. 13)
  - Background removal    (Eq. 23)
  - 4×4 pixel binning
        ↓
Optical Derivatives
  I_ρ = (I(ρ+Δρ) - I(ρ-Δρ)) / (2Δρ)
  I_A = (I(A+ΔA) - I(A-ΔA)) / (2ΔA)
        ↓
Depth (least squares over 5×5 window)
  ratio = Σ(I_ρ · I_A) / Σ(I_ρ²)
  Z = Zs / (ρ·Zs - 1 - A·Zs · ratio)
        ↓
Confidence filtering
  C = I_ρ²  →  top 50% pixels retained
```

## Key Equations

**Image formation:**

$$I(x,y;Z) = k(x,y;Z) \circledast P(x,y;Z)$$

**Blur scale:**

$$\sigma(Z;\,A,\rho,Z_s) = A + \left(\rho - \frac{1}{Z}\right)AZ_s$$

**Optical derivatives (chain rule):**

$$I_\rho = AZ_s\,[k_\sigma \circledast P], \quad I_A = \left(1 + \left(\rho - \frac{1}{Z}\right)Z_s\right)[k_\sigma \circledast P]$$

**Depth (derived from ratio — scene texture and PSF cancel out):**

$$Z = \frac{Z_s}{\rho Z_s - 1 - AZ_s \cdot I_A/I_\rho}$$

## Optical Parameters

| Parameter | Value | Description |
|---|---|---|
| ρ | 8.9 + lens OP (dpt) | Total optical power |
| A | 0.0025 m | Aperture radius |
| Zs | 0.1100 m | Sensor distance |
| Δρ | 0.06 dpt | Optical power step |
| ΔA | 0.0010 m | Aperture step |

## Reference

```
Luo, J., Liu, Y., Alexander, E., & Guo, Q. (2025).
Depth from Coupled Optical Differentiation.
International Journal of Computer Vision, 133, 8109–8126.
```