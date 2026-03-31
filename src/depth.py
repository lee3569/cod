"""
Depth from Coupled Optical Differentiation
All scenes visualization — Fig. 14 style
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path


def rho_aligned(img, current_rho, target_rho):
    u = np.arange(0, 1920, 1)
    v = np.arange(0, 1200, 1)
    uu, vv = np.meshgrid(u, v)
    alpha1, beta1, gamma1 = -0.02003, 0.0002439, 20.8386
    alpha2, beta2, gamma2 =  0.0001309, -0.02032, 16.39735
    u0 = (uu - beta1 * target_rho / (beta2 * target_rho + 1) * (vv - gamma2 * target_rho) - gamma1 * target_rho) / (alpha1 * target_rho + 1)
    v0 = (vv - gamma2 * target_rho) / (beta2 * target_rho + 1)
    a_ = alpha1 * u0 + beta1 * v0 + gamma1
    b_ = alpha2 * u0 + beta2 * v0 + gamma2
    u1 = np.clip(a_ * current_rho + u0, 0, 1918)
    v1 = np.clip(b_ * current_rho + v0, 0, 1198)
    x, y = u1.astype(int), v1.astype(int)
    f1 = (x+1 - u1) * img[y+1, x] + (u1 - x) * img[y+1, x+1]
    f2 = (x+1 - u1) * img[y,   x] + (u1 - x) * img[y,   x+1]
    return (v1 - y) * f1 + (y+1 - v1) * f2


def bin_image(image, r=4):
    if r <= 1:
        return image
    s = np.zeros(image.shape)
    for i in range(r):
        for j in range(r):
            s += np.roll(image, (-i, -j), axis=(0, 1))
    return (s / r**2)[::r, ::r]


def remove_background(image, window=21):
    k = np.ones((window, 1)) / window
    blurred = signal.convolve2d(image,   k,   "same", "symm")
    blurred = signal.convolve2d(blurred, k.T, "same", "symm")
    return image - blurred


def compute_depth(I_rho, I_A, params, kernel_size=5):
    """
    Z = Zs / (rho*Zs - 1 - A*Zs * I_A/I_rho)
    ratio = Σ(I_ρ * I_A) / Σ(I_ρ²)  [least squares, Eq. 31]
    """
    rho = params["rho"]
    A   = params["A"]
    Zs  = params["Zs"]

    k = np.ones((kernel_size, 1))
    num = signal.convolve2d(I_rho * I_A, k,   "same", "symm")
    num = signal.convolve2d(num,          k.T, "same", "symm")
    den = signal.convolve2d(I_rho ** 2,  k,   "same", "symm")
    den = signal.convolve2d(den,          k.T, "same", "symm")

    ratio = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

    # Z = Zs / (ρ*Zs - 1 - A*Zs * I_A/I_ρ)
    denom = rho * Zs - 1 - A * Zs * ratio
    Z = np.divide(Zs, denom, out=np.zeros_like(denom), where=denom != 0)
    return Z


def filter_by_confidence(Z, I_rho, sparsity=0.5):
    confidence = I_rho ** 2
    c = confidence.flatten()
    c = c[np.isfinite(c)]
    threshold = np.sort(c)[int(len(c) * sparsity)]
    return np.where(confidence > threshold, Z, np.nan)


def process_pkl(filepath):
    data = pickle.load(open(filepath, "rb"))
    A_minus_idx, A_idx, A_plus_idx       = 0, 1, 2
    rho_minus_idx, rho_idx, rho_plus_idx = 0, 1, 2

    rho       = data[0][A_idx][rho_idx]["OP"]
    rho_plus  = data[0][A_idx][rho_plus_idx]["OP"]
    rho_minus = data[0][A_idx][rho_minus_idx]["OP"]
    A_sigma       = data[0][A_idx][rho_idx]["Sigma"]
    A_plus_sigma  = data[0][A_plus_idx][rho_idx]["Sigma"]
    A_minus_sigma = data[0][A_minus_idx][rho_idx]["Sigma"]

    img_rho_plus  = data[0][A_idx][rho_plus_idx]["Img"].astype(np.float64)
    img_rho_minus = data[0][A_idx][rho_minus_idx]["Img"].astype(np.float64)
    img_A_plus    = data[0][A_plus_idx][rho_idx]["Img"].astype(np.float64)
    img_A_minus   = data[0][A_minus_idx][rho_idx]["Img"].astype(np.float64)
    img_scene     = data[0][A_idx][rho_idx]["Img"].astype(np.float64)

    params = {
        "rho": 8.9 + rho,   # total optical power (dpt)
        "A":   0.0025,       # aperture radius (m)
        "Zs":  0.1100,       # sensor distance (m)
    }
    delta_rho = (rho_plus - rho_minus) / 2   # = 0.06 dpt
    delta_A   = 0.0010                        # ΔA in meters (박사생 params 기준)

    # Brightness correction (Eq. 13) — aperture pair only
    img_A_plus  *= (A_sigma / A_plus_sigma)  ** 2
    img_A_minus *= (A_sigma / A_minus_sigma) ** 2

    # Geometric alignment — rho pair only
    img_rho_plus  = rho_aligned(img_rho_plus,  rho_plus,  rho)
    img_rho_minus = rho_aligned(img_rho_minus, rho_minus, rho)

    # Binning
    BIN = 4
    img_rho_plus  = bin_image(img_rho_plus,  BIN)
    img_rho_minus = bin_image(img_rho_minus, BIN)
    img_A_plus    = bin_image(img_A_plus,    BIN)
    img_A_minus   = bin_image(img_A_minus,   BIN)
    img_scene     = bin_image(img_scene,     BIN)

    # Background removal (Eq. 23) — aperture pair only
    img_A_plus  = remove_background(img_A_plus)
    img_A_minus = remove_background(img_A_minus)

    # Optical derivatives (central difference, normalized)
    # I_ρ = (I(ρ+Δρ) - I(ρ-Δρ)) / (2Δρ)
    # I_A = (I(A+ΔA) - I(A-ΔA)) / (2ΔA)
    I_rho = (img_rho_plus - img_rho_minus) / (2 * delta_rho)
    I_A   = (img_A_plus   - img_A_minus)   / (2 * delta_A)

    Z          = compute_depth(I_rho, I_A, params, kernel_size=5)
    Z_filtered = filter_by_confidence(Z, I_rho, sparsity=0.5)
    return img_scene, Z_filtered


def main():
    data_dir   = Path("./data")
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)

    pkl_files = sorted([
        p for p in data_dir.glob("*.pkl")
        if "LinearSlide" not in p.stem
    ])
    if not pkl_files:
        print("data/ 폴더에 pkl 파일이 없어요.")
        return

    print(f"총 {len(pkl_files)}개 파일 처리 중...")

    n            = len(pkl_files)
    cols_per_row = 4
    rows         = int(np.ceil(n / cols_per_row))
    fig, axes    = plt.subplots(rows * 2, cols_per_row, figsize=(cols_per_row * 4, rows * 4))
    axes         = np.array(axes).reshape(rows * 2, cols_per_row)

    for i, pkl_path in enumerate(pkl_files):
        scene_name = pkl_path.stem.split("_")[-1]
        row_pair   = (i // cols_per_row) * 2
        col        = i % cols_per_row
        print(f"[{i+1}/{n}] {scene_name}")
        try:
            img_scene, Z_filtered = process_pkl(pkl_path)

            axes[row_pair][col].imshow(img_scene, cmap="gray")
            axes[row_pair][col].set_title(scene_name, fontsize=8)
            axes[row_pair][col].axis("off")

            is_close = "Close" in pkl_path.stem
            vmin, vmax = (0.4, 0.8) if is_close else (0.4, 1.4)
            im = axes[row_pair + 1][col].imshow(Z_filtered, vmin=vmin, vmax=vmax, cmap="rainbow")
            axes[row_pair + 1][col].axis("off")
            fig.colorbar(im, ax=axes[row_pair + 1][col], fraction=0.046, pad=0.04)

        except Exception as e:
            print(f"  failed: {e}")
            axes[row_pair][col].axis("off")
            axes[row_pair + 1][col].axis("off")

    for j in range(n, rows * cols_per_row):
        r, c = (j // cols_per_row) * 2, j % cols_per_row
        axes[r][c].axis("off")
        axes[r + 1][c].axis("off")

    plt.suptitle("Depth from Coupled Optical Differentiation", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = result_dir / "all_scenes.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n저장 완료: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()