"""
===============================================================
白層形成の感度解析 - Phase 1 & 2 完全統合版

Analysis Modules:
- Phase 1: K0, λ, Λ_c sensitivity (with optimized version)
- Phase 2: Defect sensitivity (standard + critical conditions)
- Comprehensive metrics tracking
- Enhanced visualization suite
- Theoretical comparison

Author: 飯泉環 (Iizumi Tamaki) & 飯泉真道 (Iizumi Masamichi)
Date: 2025-01-25 (Updated)
"""

import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
from scipy.stats import linregress
from copy import deepcopy
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ========================================
# Core Functions
# ========================================

@jit(nopython=True)
def generate_fcc_lattice_anisotropic(Nx, Ny, Nz, a0=3.6):
    """異方性FCC格子生成"""
    n_atoms = 4 * Nx * Ny * Nz
    positions = np.empty((n_atoms, 3), dtype=np.float64)

    fcc_basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ], dtype=np.float64)

    idx = 0
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for b in range(4):
                    positions[idx, 0] = (i + fcc_basis[b, 0]) * a0
                    positions[idx, 1] = (j + fcc_basis[b, 1]) * a0
                    positions[idx, 2] = (k + fcc_basis[b, 2]) * a0
                    idx += 1

    return positions


def assign_composition(positions, comp_dict):
    """組成割り当て"""
    n_atoms = len(positions)
    elements = []
    probs = []

    for elem, frac in comp_dict.items():
        elements.append(elem)
        probs.append(frac)

    atom_types = np.random.choice(elements, size=n_atoms, p=probs)
    return atom_types


def introduce_imperfections(positions, atom_types, Cr_substitution_rate=0.0, vacancy_ratio=0.0, verbose=False):
    """不完全性導入"""
    if Cr_substitution_rate > 0:
        fe_indices = np.where(atom_types == 'Fe')[0]
        n_substitute = int(len(fe_indices) * Cr_substitution_rate)
        substitute_indices = np.random.choice(fe_indices, n_substitute, replace=False)
        atom_types[substitute_indices] = 'Cr'
        if verbose:
            print(f"  ✓ Cr substitution: {n_substitute} Fe→Cr ({Cr_substitution_rate*100:.1f}%)")

    if vacancy_ratio > 0:
        n_vacancies = int(len(positions) * vacancy_ratio)
        vacancy_indices = np.random.choice(len(positions), n_vacancies, replace=False)
        positions = np.delete(positions, vacancy_indices, axis=0)
        atom_types = np.delete(atom_types, vacancy_indices)
        if verbose:
            print(f"  ✓ Vacancies: {n_vacancies} ({vacancy_ratio*100:.1f}%)")

    return positions, atom_types


@jit(nopython=True)
def get_pair_potential(type_i, type_j):
    """ペアポテンシャル"""
    if type_i > type_j:
        type_i, type_j = type_j, type_i

    if type_i == 0 and type_j == 0:  # Fe-Fe
        return 1.00
    elif type_i == 0 and type_j == 1:  # Fe-Cr
        return 0.85
    elif type_i == 0 and type_j == 2:  # Fe-Ni
        return 0.80
    elif type_i == 1 and type_j == 1:  # Cr-Cr
        return 0.90
    elif type_i == 1 and type_j == 2:  # Cr-Ni
        return 0.75
    elif type_i == 2 and type_j == 2:  # Ni-Ni
        return 0.85
    else:
        return 0.80


@jit(nopython=True, parallel=True)
def compute_Veff(positions, atom_types, r_cutoff=6.0):
    """有効ポテンシャル計算"""
    n_atoms = len(positions)
    V_eff = np.zeros(n_atoms, dtype=np.float64)

    for i in prange(n_atoms):
        V_sum = 0.0
        n_neighbors = 0

        for j in range(n_atoms):
            if i == j:
                continue

            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            r_ij = np.sqrt(dx*dx + dy*dy + dz*dz)

            if r_ij < r_cutoff:
                V_pair = get_pair_potential(atom_types[i], atom_types[j])
                V_sum += V_pair
                n_neighbors += 1

        if n_neighbors > 0:
            V_eff[i] = V_sum / n_neighbors
        else:
            V_eff[i] = 1.0

    return V_eff


@jit(nopython=True)
def compute_K_field(positions, K0=2.0, lambda_decay=22.0, noise_std=0.0):
    """エネルギー入力場計算"""
    z_coords = positions[:, 2]
    z_min = np.min(z_coords)

    K_field = K0 * np.exp(-(z_coords - z_min) / lambda_decay)

    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, size=len(K_field))
        K_field *= (1.0 + noise)

    return K_field


@jit(nopython=True)
def compute_Lambda_field(K_field, V_eff):
    """Λ場計算"""
    return K_field / V_eff


def identify_percolation_clusters(positions, Lambda_field, atom_types,
                                  threshold=0.9, grid_size=2.5):
    """パーコレーションクラスタ検出"""
    n_atoms = len(positions)

    fe_index = 0
    mask = (Lambda_field > threshold) & (atom_types == fe_index)

    grid_pos = (positions / grid_size).astype(np.int32)

    grid_min = grid_pos.min(axis=0)
    grid_max = grid_pos.max(axis=0)
    grid_shape = tuple(grid_max - grid_min + 1)

    grid = np.zeros(grid_shape, dtype=np.int32)

    for i, m in enumerate(mask):
        if m:
            gp = grid_pos[i] - grid_min
            grid[tuple(gp)] = 1

    structure = generate_binary_structure(3, 3)
    labeled_grid, num_clusters = label(grid, structure=structure)

    cluster_sizes = []
    for cluster_id in range(1, num_clusters + 1):
        size = np.sum(labeled_grid == cluster_id)
        cluster_sizes.append(size)

    if cluster_sizes:
        largest_cluster = max(cluster_sizes)
        total_fe = np.sum(atom_types == fe_index)
        percolation_ratio = largest_cluster / total_fe if total_fe > 0 else 0
    else:
        largest_cluster = 0
        percolation_ratio = 0

    return labeled_grid, num_clusters, largest_cluster, percolation_ratio, mask


# ========================================
# Comprehensive Metrics Module
# ========================================

def measure_white_layer_metrics(positions, Lambda_field, atom_types,
                                Lc_values=[0.9, 0.95, 1.0, 1.05, 1.1],
                                grid_size=2.5):
    """
    包括的白層メトリクス測定

    Parameters:
        Lc_values: Λ閾値のリスト

    Returns:
        metrics: 各Λ閾値でのメトリクス辞書
    """
    metrics = {
        'Lc': [],
        'thickness': [],
        'z_min': [],
        'z_max': [],
        'coverage': [],
        'percolation_ratio': [],
        'cluster_count': [],
        'largest_cluster_size': [],
        'active_atom_fraction': []
    }

    for Lc in Lc_values:
        # クラスタ解析
        labeled_grid, num_clusters, largest_cluster, perc_ratio, mask = \
            identify_percolation_clusters(positions, Lambda_field, atom_types,
                                         Lc, grid_size)

        # 厚さ測定
        active_mask = Lambda_field > Lc
        if np.sum(active_mask) > 0:
            z_coords = positions[active_mask, 2]
            z_min = np.min(z_coords)
            z_max = np.max(z_coords)
            thickness = z_max - z_min
        else:
            z_min = z_max = thickness = 0.0

        # 表面カバー率
        if np.sum(mask) > 0:
            surface_threshold = z_min + thickness * 0.5
            surface_mask = (positions[:, 2] <= surface_threshold) & mask

            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            x_min_pos, x_max_pos = x_coords.min(), x_coords.max()
            y_min_pos, y_max_pos = y_coords.min(), y_coords.max()

            nx = int((x_max_pos - x_min_pos) / 1.0) + 1
            ny = int((y_max_pos - y_min_pos) / 1.0) + 1

            xy_grid = np.zeros((nx, ny), dtype=bool)

            for i in np.where(surface_mask)[0]:
                ix = int((positions[i, 0] - x_min_pos) / 1.0)
                iy = int((positions[i, 1] - y_min_pos) / 1.0)
                if 0 <= ix < nx and 0 <= iy < ny:
                    xy_grid[ix, iy] = True

            coverage = np.sum(xy_grid) / (nx * ny) if (nx * ny) > 0 else 0.0
        else:
            coverage = 0.0

        # 活性原子比率
        active_fraction = np.sum(active_mask) / len(Lambda_field)

        # 記録
        metrics['Lc'].append(Lc)
        metrics['thickness'].append(thickness)
        metrics['z_min'].append(z_min)
        metrics['z_max'].append(z_max)
        metrics['coverage'].append(coverage)
        metrics['percolation_ratio'].append(perc_ratio)
        metrics['cluster_count'].append(num_clusters)
        metrics['largest_cluster_size'].append(largest_cluster)
        metrics['active_atom_fraction'].append(active_fraction)

    return pd.DataFrame(metrics)

# ========================================
# Simulation Core
# ========================================

def run_single_simulation(Nx, Ny, Nz, K0, lambda_decay, Lambda_threshold,
                         Cr_sub=0.0, vacancy=0.0, K_noise=0.0,
                         composition={'Fe': 0.72, 'Cr': 0.18, 'Ni': 0.10},
                         verbose=False):
    """
    単一シミュレーション実行（軽量版）

    Returns:
        dict: 白層メトリクスのみ
    """
    # 格子生成
    positions = generate_fcc_lattice_anisotropic(Nx, Ny, Nz, a0=3.6)
    atom_types_str = assign_composition(positions, composition)

    # 不完全性
    if Cr_sub > 0 or vacancy > 0:
        positions, atom_types_str = introduce_imperfections(
            positions, atom_types_str, Cr_sub, vacancy, verbose
        )

    elem_to_int = {'Fe': 0, 'Cr': 1, 'Ni': 2}
    atom_types = np.array([elem_to_int[e] for e in atom_types_str], dtype=np.int32)

    # 計算
    V_eff = compute_Veff(positions, atom_types, r_cutoff=6.0)
    K_field = compute_K_field(positions, K0, lambda_decay, K_noise)
    Lambda_field = compute_Lambda_field(K_field, V_eff)

    # メトリクス
    metrics = measure_white_layer_metrics(
        positions, Lambda_field, atom_types,
        Lc_values=[Lambda_threshold]
    )

    return metrics.iloc[0].to_dict()


def run_full_simulation(Nx, Ny, Nz, K0, lambda_decay, Lambda_threshold,
                       Cr_sub=0.0, vacancy=0.0, K_noise=0.0,
                       composition={'Fe': 0.72, 'Cr': 0.18, 'Ni': 0.10},
                       verbose=False):
    """
    完全シミュレーション実行（全データ返却版）

    Returns:
        dict: {
            'positions': 原子座標,
            'atom_types': 原子種,
            'Lambda_field': Λ場,
            'K_field': K場,
            'V_eff': 有効ポテンシャル,
            'metrics': 白層メトリクス
        }
    """
    # 格子生成
    positions = generate_fcc_lattice_anisotropic(Nx, Ny, Nz, a0=3.6)
    atom_types_str = assign_composition(positions, composition)

    # 不完全性
    if Cr_sub > 0 or vacancy > 0:
        positions, atom_types_str = introduce_imperfections(
            positions, atom_types_str, Cr_sub, vacancy, verbose
        )

    elem_to_int = {'Fe': 0, 'Cr': 1, 'Ni': 2}
    atom_types = np.array([elem_to_int[e] for e in atom_types_str], dtype=np.int32)

    # 計算
    V_eff = compute_Veff(positions, atom_types, r_cutoff=6.0)
    K_field = compute_K_field(positions, K0, lambda_decay, K_noise)
    Lambda_field = compute_Lambda_field(K_field, V_eff)

    # メトリクス
    metrics = measure_white_layer_metrics(
        positions, Lambda_field, atom_types,
        Lc_values=[Lambda_threshold]
    )

    # ✅ 全データを返す！
    return {
        'positions': positions,
        'atom_types': atom_types,
        'Lambda_field': Lambda_field,
        'K_field': K_field,
        'V_eff': V_eff,
        'metrics': metrics.iloc[0].to_dict()
    }

# ========================================
# Phase 1: K0, λ, Λ_c Sensitivity
# ========================================

def sensitivity_K0(K0_range=np.linspace(0.8, 3.0, 12),
                  Nx=30, Ny=30, Nz=6,
                  lambda_decay=22.0, Lc=0.9,
                  Cr_sub=0.0, vacancy=0.0):
    """Phase 1-1: K0感度解析"""
    print("="*60)
    print("Phase 1-1: K0 Sensitivity Analysis")
    print("="*60)

    results = []

    for i, K0 in enumerate(K0_range):
        print(f"\rProgress: {i+1}/{len(K0_range)} (K0={K0:.2f})", end='')

        metrics = run_single_simulation(
            Nx, Ny, Nz, K0, lambda_decay, Lc,
            Cr_sub, vacancy, K_noise=0.0
        )

        metrics['K0'] = K0
        results.append(metrics)

    print("\n" + "="*60)

    return pd.DataFrame(results)


def sensitivity_lambda(lambda_range=np.linspace(10, 40, 16),
                      Nx=30, Ny=30, Nz=6,
                      K0=2.0, Lc=0.9,
                      Cr_sub=0.0, vacancy=0.0):
    """Phase 1-2: λ感度解析（標準版）"""
    print("="*60)
    print("Phase 1-2: λ (Decay Length) Sensitivity Analysis")
    print("="*60)

    results = []

    for i, lam in enumerate(lambda_range):
        print(f"\rProgress: {i+1}/{len(lambda_range)} (λ={lam:.1f})", end='')

        metrics = run_single_simulation(
            Nx, Ny, Nz, K0, lam, Lc,
            Cr_sub, vacancy, K_noise=0.0
        )

        metrics['lambda'] = lam
        results.append(metrics)

    print("\n" + "="*60)

    return pd.DataFrame(results)


def sensitivity_lambda_optimized(lambda_range=np.linspace(15, 30, 31),
                                 Nx=30, Ny=30, Nz=6,
                                 K0=2.0, Lc=0.9,
                                 Cr_sub=0.0, vacancy=0.0):
    """
    Phase 1-2 (Optimized): λ感度解析 - 線形領域最適化

    範囲の根拠：
    - Phase 1-2の結果から、λ≈15Åで p>p_c
    - λ>30Åで厚さ飽和の兆候
    - よって λ=15-30Åが「線形成長領域」

    期待される結果：
    - R² > 0.90（高い相関！）
    - slope ≈ 0.5-0.8（熱拡散係数と関連）
    """
    print("="*60)
    print("Phase 1-2 (Optimized): λ Sensitivity - Linear Region")
    print(f"  Range: λ = {lambda_range[0]:.1f} - {lambda_range[-1]:.1f} Å")
    print("="*60)

    results = []

    for i, lam in enumerate(lambda_range):
        print(f"\rProgress: {i+1}/{len(lambda_range)} (λ={lam:.1f})", end='')

        metrics = run_single_simulation(
            Nx, Ny, Nz, K0, lam, Lc,
            Cr_sub, vacancy, K_noise=0.0
        )

        metrics['lambda'] = lam
        results.append(metrics)

    print("\n" + "="*60)

    return pd.DataFrame(results)


def sensitivity_Lc(Lc_range=np.linspace(0.85, 1.15, 13),
                  Nx=30, Ny=30, Nz=6,
                  K0=2.0, lambda_decay=22.0,
                  Cr_sub=0.0, vacancy=0.0):
    """Phase 1-3: Λ_c感度解析"""
    print("="*60)
    print("Phase 1-3: Λ_c (Threshold) Sensitivity Analysis")
    print("="*60)

    results = []

    for i, Lc in enumerate(Lc_range):
        print(f"\rProgress: {i+1}/{len(Lc_range)} (Λ_c={Lc:.2f})", end='')

        metrics = run_single_simulation(
            Nx, Ny, Nz, K0, lambda_decay, Lc,
            Cr_sub, vacancy, K_noise=0.0
        )

        metrics['Lc'] = Lc
        results.append(metrics)

    print("\n" + "="*60)

    return pd.DataFrame(results)


# ========================================
# Phase 2: Defect Sensitivity (Standard)
# ========================================

def sensitivity_vacancy(vacancy_range=np.linspace(0, 0.05, 11),
                       Nx=30, Ny=30, Nz=6,
                       K0=2.0, lambda_decay=22.0, Lc=0.9,
                       Cr_sub=0.0):
    """Phase 2-1: 空孔率感度解析（標準版）"""
    print("="*60)
    print("Phase 2-1: Vacancy Ratio Sensitivity Analysis")
    print("="*60)

    results = []

    for i, vac in enumerate(vacancy_range):
        print(f"\rProgress: {i+1}/{len(vacancy_range)} (vacancy={vac*100:.1f}%)", end='')

        metrics = run_single_simulation(
            Nx, Ny, Nz, K0, lambda_decay, Lc,
            Cr_sub, vac, K_noise=0.05
        )

        metrics['vacancy'] = vac
        results.append(metrics)

    print("\n" + "="*60)

    return pd.DataFrame(results)


def sensitivity_Cr_substitution(Cr_range=np.linspace(0, 0.10, 11),
                                Nx=30, Ny=30, Nz=6,
                                K0=2.0, lambda_decay=22.0, Lc=0.9,
                                vacancy=0.0):
    """Phase 2-2: Cr置換率感度解析（標準版）"""
    print("="*60)
    print("Phase 2-2: Cr Substitution Rate Sensitivity Analysis")
    print("="*60)

    results = []

    for i, Cr_sub in enumerate(Cr_range):
        print(f"\rProgress: {i+1}/{len(Cr_range)} (Cr_sub={Cr_sub*100:.1f}%)", end='')

        metrics = run_single_simulation(
            Nx, Ny, Nz, K0, lambda_decay, Lc,
            Cr_sub, vacancy, K_noise=0.05
        )

        metrics['Cr_substitution'] = Cr_sub
        results.append(metrics)

    print("\n" + "="*60)

    return pd.DataFrame(results)


# ========================================
# Phase 2: Defect Sensitivity (Critical)
# ========================================

def sensitivity_vacancy_critical(vacancy_range=np.linspace(0, 0.10, 21),
                                Nx=30, Ny=30, Nz=6,
                                K0=1.5, lambda_decay=16.0, Lc=0.9,
                                Cr_sub=0.0):
    """
    Phase 2-1 (Critical): 臨界条件での空孔率感度

    期待される結果：
    - vacancy=0%: p≈0.35-0.45（p_c超えたばかり）
    - vacancy↑: pが減少してp_c=0.31に近づく
    - vacancy≈5-8%: p<p_c（パーコレーション崩壊！）
    """
    print("="*60)
    print("Phase 2-1 (Critical): Vacancy Sensitivity at K0≈K0_c")
    print(f"  Conditions: K0={K0:.2f}, λ={lambda_decay:.1f}Å")
    print("="*60)

    results = []

    for i, vac in enumerate(vacancy_range):
        print(f"\rProgress: {i+1}/{len(vacancy_range)} (vacancy={vac*100:.1f}%)", end='')

        metrics = run_single_simulation(
            Nx, Ny, Nz, K0, lambda_decay, Lc,
            Cr_sub, vac, K_noise=0.05
        )

        metrics['vacancy'] = vac
        results.append(metrics)

    print("\n" + "="*60)

    return pd.DataFrame(results)


def sensitivity_Cr_substitution_critical(Cr_range=np.linspace(0, 0.15, 21),
                                         Nx=30, Ny=30, Nz=6,
                                         K0=1.5, lambda_decay=16.0, Lc=0.9,
                                         vacancy=0.0):
    """
    Phase 2-2 (Critical): 臨界条件でのCr置換率感度

    期待される結果：
    - Cr置換でV_effが変化
    - Fe-Cr結合強度=0.85（Feより弱い）
    - クラスタ数↑、p↓の予想
    """
    print("="*60)
    print("Phase 2-2 (Critical): Cr Substitution at K0≈K0_c")
    print(f"  Conditions: K0={K0:.2f}, λ={lambda_decay:.1f}Å")
    print("="*60)

    results = []

    for i, Cr_sub in enumerate(Cr_range):
        print(f"\rProgress: {i+1}/{len(Cr_range)} (Cr_sub={Cr_sub*100:.1f}%)", end='')

        metrics = run_single_simulation(
            Nx, Ny, Nz, K0, lambda_decay, Lc,
            Cr_sub, vacancy, K_noise=0.05
        )

        metrics['Cr_substitution'] = Cr_sub
        results.append(metrics)

    print("\n" + "="*60)

    return pd.DataFrame(results)


# ========================================
# Visualization Suite
# ========================================

def plot_K0_sensitivity(df):
    """K0感度解析の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 厚さ vs K0
    ax1 = axes[0, 0]
    ax1.plot(df['K0'], df['thickness'], 'o-', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Surface Energy K₀', fontsize=12)
    ax1.set_ylabel('White Layer Thickness [Å]', fontsize=12)
    ax1.set_title('Thickness vs K₀', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. パーコレーション比 vs K0
    ax2 = axes[0, 1]
    ax2.plot(df['K0'], df['percolation_ratio'], 's-', linewidth=2, markersize=8, color='blue')
    ax2.axhline(y=0.31, color='green', linestyle='--', label='p_c (3D)')
    ax2.set_xlabel('Surface Energy K₀', fontsize=12)
    ax2.set_ylabel('Percolation Ratio p', fontsize=12)
    ax2.set_title('Percolation Ratio vs K₀', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. カバー率 vs K0
    ax3 = axes[1, 0]
    ax3.plot(df['K0'], df['coverage'], '^-', linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('Surface Energy K₀', fontsize=12)
    ax3.set_ylabel('Surface Coverage Ratio', fontsize=12)
    ax3.set_title('Coverage vs K₀', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. クラスタ数 vs K0
    ax4 = axes[1, 1]
    ax4.plot(df['K0'], df['cluster_count'], 'd-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Surface Energy K₀', fontsize=12)
    ax4.set_ylabel('Number of Clusters', fontsize=12)
    ax4.set_title('Cluster Count vs K₀', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Phase 1-1: K₀ Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_lambda_sensitivity(df):
    """λ感度解析の可視化（標準版）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 厚さ vs λ（線形フィット）
    ax1 = axes[0]
    ax1.plot(df['lambda'], df['thickness'], 'o', markersize=10, color='red', label='Data')

    # 線形フィット
    slope, intercept, r_value, p_value, std_err = linregress(df['lambda'], df['thickness'])
    fit_line = slope * df['lambda'] + intercept
    ax1.plot(df['lambda'], fit_line, '--', linewidth=2, color='blue',
             label=f'Linear Fit: y={slope:.2f}x+{intercept:.2f}\nR²={r_value**2:.3f}')

    ax1.set_xlabel('Decay Length λ [Å]', fontsize=12)
    ax1.set_ylabel('White Layer Thickness [Å]', fontsize=12)
    ax1.set_title('Thickness ∝ λ (Heat Diffusion Evidence)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. パーコレーション比 vs λ
    ax2 = axes[1]
    ax2.plot(df['lambda'], df['percolation_ratio'], 's-', linewidth=2, markersize=8, color='blue')
    ax2.axhline(y=0.31, color='green', linestyle='--', label='p_c (3D)')
    ax2.set_xlabel('Decay Length λ [Å]', fontsize=12)
    ax2.set_ylabel('Percolation Ratio p', fontsize=12)
    ax2.set_title('Percolation Ratio vs λ', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Phase 1-2: λ Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_lambda_sensitivity_improved(df):
    """λ感度解析の可視化（改良版）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 厚さ vs λ（改善された線形フィット）
    ax1 = axes[0]

    # 線形領域のマスク
    mask = (df['lambda'] >= 15) & (df['lambda'] <= 30)
    df_linear = df[mask]

    # 全データプロット（薄く）
    ax1.plot(df['lambda'], df['thickness'], 'o',
             markersize=8, color='lightcoral', alpha=0.5, label='All Data')

    # 線形領域を強調
    ax1.plot(df_linear['lambda'], df_linear['thickness'], 'o',
             markersize=10, color='red', label='Linear Region (15-30Å)')

    # 線形フィット
    slope, intercept, r_value, p_value, std_err = linregress(
        df_linear['lambda'], df_linear['thickness']
    )
    fit_line = slope * df_linear['lambda'] + intercept

    ax1.plot(df_linear['lambda'], fit_line, '--', linewidth=3, color='blue',
             label=f'Linear Fit:\n'
                   f'y = {slope:.3f}x + {intercept:.2f}\n'
                   f'R² = {r_value**2:.3f}\n'
                   f'p < {p_value:.1e}')

    # 線形領域をハイライト
    ax1.axvspan(15, 30, alpha=0.1, color='green',
                label='Linear Growth Region')

    ax1.set_xlabel('Decay Length λ [Å]', fontsize=12)
    ax1.set_ylabel('White Layer Thickness [Å]', fontsize=12)
    ax1.set_title('Thickness ∝ λ (Heat Diffusion Evidence)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 残差プロット（フィット品質の確認）
    ax2 = axes[1]
    residuals = df_linear['thickness'] - (slope * df_linear['lambda'] + intercept)
    ax2.scatter(df_linear['lambda'], residuals, s=100, color='blue', alpha=0.6)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.fill_between(df_linear['lambda'], -2, 2, alpha=0.1, color='gray')
    ax2.set_xlabel('Decay Length λ [Å]', fontsize=12)
    ax2.set_ylabel('Residuals [Å]', fontsize=12)
    ax2.set_title('Residual Analysis (Quality Check)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Phase 1-2 (Optimized): λ Sensitivity Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return slope, intercept, r_value


def plot_Lc_sensitivity(df):
    """Λ_c感度解析の可視化"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 厚さ vs Λ_c
    ax1 = axes[0]
    ax1.plot(df['Lc'], df['thickness'], 'o-', linewidth=2, markersize=8, color='red')
    ax1.axvline(x=0.9, color='blue', linestyle='--', alpha=0.5, label='Λ_c=0.9')
    ax1.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Λ_c=1.0')
    ax1.set_xlabel('Threshold Λ_c', fontsize=12)
    ax1.set_ylabel('Thickness [Å]', fontsize=12)
    ax1.set_title('Thickness vs Λ_c', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. パーコレーション比 vs Λ_c
    ax2 = axes[1]
    ax2.plot(df['Lc'], df['percolation_ratio'], 's-', linewidth=2, markersize=8, color='blue')
    ax2.axhline(y=0.31, color='green', linestyle='--', label='p_c (3D)')
    ax2.set_xlabel('Threshold Λ_c', fontsize=12)
    ax2.set_ylabel('Percolation Ratio p', fontsize=12)
    ax2.set_title('Percolation Ratio vs Λ_c', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 活性原子比率 vs Λ_c
    ax3 = axes[2]
    ax3.plot(df['Lc'], df['active_atom_fraction'], '^-', linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('Threshold Λ_c', fontsize=12)
    ax3.set_ylabel('Active Atom Fraction', fontsize=12)
    ax3.set_title('Active Fraction vs Λ_c', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Phase 1-3: Λ_c Robustness Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_defect_sensitivity(df_vac, df_Cr):
    """欠陥感度解析の可視化（標準版）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. クラスタ数 vs 空孔率
    ax1 = axes[0, 0]
    ax1.plot(df_vac['vacancy']*100, df_vac['cluster_count'], 'o-',
             linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Vacancy Ratio [%]', fontsize=12)
    ax1.set_ylabel('Number of Clusters', fontsize=12)
    ax1.set_title('Cluster Count vs Vacancy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. クラスタ数 vs Cr置換率
    ax2 = axes[0, 1]
    ax2.plot(df_Cr['Cr_substitution']*100, df_Cr['cluster_count'], 's-',
             linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('Cr Substitution [%]', fontsize=12)
    ax2.set_ylabel('Number of Clusters', fontsize=12)
    ax2.set_title('Cluster Count vs Cr Substitution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. パーコレーション比 vs 空孔率
    ax3 = axes[1, 0]
    ax3.plot(df_vac['vacancy']*100, df_vac['percolation_ratio'], '^-',
             linewidth=2, markersize=8, color='orange')
    ax3.axhline(y=0.31, color='green', linestyle='--', label='p_c (3D)')
    ax3.set_xlabel('Vacancy Ratio [%]', fontsize=12)
    ax3.set_ylabel('Percolation Ratio p', fontsize=12)
    ax3.set_title('Percolation Ratio vs Vacancy', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. パーコレーション比 vs Cr置換率
    ax4 = axes[1, 1]
    ax4.plot(df_Cr['Cr_substitution']*100, df_Cr['percolation_ratio'], 'd-',
             linewidth=2, markersize=8, color='purple')
    ax4.axhline(y=0.31, color='green', linestyle='--', label='p_c (3D)')
    ax4.set_xlabel('Cr Substitution [%]', fontsize=12)
    ax4.set_ylabel('Percolation Ratio p', fontsize=12)
    ax4.set_title('Percolation Ratio vs Cr Substitution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Phase 2: Defect Sensitivity Analysis (Standard)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_defect_sensitivity_critical(df_vac, df_Cr):
    """欠陥感度解析の可視化（臨界条件版）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. クラスタ数 vs 空孔率
    ax1 = axes[0, 0]
    ax1.plot(df_vac['vacancy']*100, df_vac['cluster_count'], 'o-',
             linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Vacancy Ratio [%]', fontsize=12)
    ax1.set_ylabel('Number of Clusters', fontsize=12)
    ax1.set_title('Cluster Count vs Vacancy (K0≈K0_c)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. クラスタ数 vs Cr置換率
    ax2 = axes[0, 1]
    ax2.plot(df_Cr['Cr_substitution']*100, df_Cr['cluster_count'], 's-',
             linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('Cr Substitution [%]', fontsize=12)
    ax2.set_ylabel('Number of Clusters', fontsize=12)
    ax2.set_title('Cluster Count vs Cr Substitution (K0≈K0_c)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. パーコレーション比 vs 空孔率（臨界点を強調）
    ax3 = axes[1, 0]
    ax3.plot(df_vac['vacancy']*100, df_vac['percolation_ratio'], '^-',
             linewidth=2, markersize=8, color='orange')
    ax3.axhline(y=0.31, color='green', linestyle='--', linewidth=2, label='p_c (3D)')
    ax3.axhspan(0.29, 0.33, alpha=0.2, color='green', label='Critical Region')
    ax3.set_xlabel('Vacancy Ratio [%]', fontsize=12)
    ax3.set_ylabel('Percolation Ratio p', fontsize=12)
    ax3.set_title('Percolation Ratio vs Vacancy (K0≈K0_c)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. パーコレーション比 vs Cr置換率（臨界点を強調）
    ax4 = axes[1, 1]
    ax4.plot(df_Cr['Cr_substitution']*100, df_Cr['percolation_ratio'], 'd-',
             linewidth=2, markersize=8, color='purple')
    ax4.axhline(y=0.31, color='green', linestyle='--', linewidth=2, label='p_c (3D)')
    ax4.axhspan(0.29, 0.33, alpha=0.2, color='green', label='Critical Region')
    ax4.set_xlabel('Cr Substitution [%]', fontsize=12)
    ax4.set_ylabel('Percolation Ratio p', fontsize=12)
    ax4.set_title('Percolation Ratio vs Cr Substitution (K0≈K0_c)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Phase 2: Defect Sensitivity Analysis (Critical Conditions)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ========================================
# Theoretical Comparison
# ========================================

def theoretical_comparison(slope, df):
    """
    理論値との比較

    熱拡散方程式から：
    thickness ≈ √(2Dt)

    1D熱拡散長：
    λ_thermal = √(D·τ)

    比較：
    slope_theory vs slope_experiment
    """
    print("\n" + "="*60)
    print("Theoretical Comparison: Heat Diffusion Analysis")
    print("="*60)

    # SUS304の熱物性値
    thermal_diffusivity = 4.0e-6  # m²/s
    print(f"Thermal diffusivity (SUS304): {thermal_diffusivity:.2e} m²/s")

    # 実験的スロープ
    print(f"\nExperimental slope: {slope:.4f} Å/Å")
    print(f"Physical meaning: Δthickness / Δλ ≈ {slope:.2f}")

    # 解釈
    print("\nInterpretation:")
    if 0.3 < slope < 0.8:
        print("  ✓ Consistent with parabolic diffusion growth")
        print("    thickness ∝ √λ would give slope ≈ 0.5-0.7")
        print("    → Heat diffusion-controlled mechanism confirmed!")
    elif slope > 0.8:
        print("  ⚠ Slope higher than expected")
        print("    → Possible collective/percolation enhancement")
    else:
        print("  ⚠ Slope lower than expected")
        print("    → Possible saturation effects")

    # 統計情報
    print("\nStatistical Summary:")
    print(f"  Mean thickness: {df['thickness'].mean():.2f} ± {df['thickness'].std():.2f} Å")
    print(f"  Min/Max λ: {df['lambda'].min():.1f} - {df['lambda'].max():.1f} Å")
    print(f"  Thickness range: {df['thickness'].min():.2f} - {df['thickness'].max():.2f} Å")

    print("="*60 + "\n")


# ========================================
# Layer Quantization Analysis
# ========================================

def analyze_thickness_quantization(df, a0=3.6):
    """
    厚さの離散性（量子化）を解析

    Parameters:
        df: DataFrame with 'thickness' column
        a0: FCC lattice parameter (Å)

    Returns:
        analysis_results: dict with quantization metrics
    """
    from scipy.signal import find_peaks

    print("\n" + "="*60)
    print("Layer Quantization Analysis")
    print("="*60)

    thicknesses = df['thickness'].values

    # ヒストグラム作成
    hist, bin_edges = np.histogram(thicknesses, bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # ピーク検出（prominence調整で感度制御）
    peaks, properties = find_peaks(hist, prominence=1, distance=2)

    if len(peaks) > 0:
        peak_thicknesses = bin_centers[peaks]
        peak_heights = hist[peaks]

        print(f"\nDetected {len(peaks)} thickness peaks:")
        for i, (thick, height) in enumerate(zip(peak_thicknesses, peak_heights)):
            print(f"  Peak {i+1}: {thick:.2f} Å (count: {height})")

        # 層間隔の計算
        if len(peak_thicknesses) > 1:
            spacings = np.diff(peak_thicknesses)
            mean_spacing = np.mean(spacings)
            std_spacing = np.std(spacings)

            print(f"\nLayer spacing analysis:")
            print(f"  Individual spacings: {', '.join([f'{s:.2f}' for s in spacings])} Å")
            print(f"  Mean spacing: {mean_spacing:.2f} ± {std_spacing:.2f} Å")

            # FCC格子との比較
            d_111 = a0 / np.sqrt(3)  # {111} plane spacing
            d_200 = a0 / 2            # {200} plane spacing
            d_220 = a0 / (2*np.sqrt(2))  # {220} plane spacing

            print(f"\nComparison with FCC lattice (a0={a0} Å):")
            print(f"  d_{{111}} = {d_111:.2f} Å (closest packed)")
            print(f"  d_{{200}} = {d_200:.2f} Å")
            print(f"  d_{{220}} = {d_220:.2f} Å")

            # 最も近い面を判定
            distances = {
                'd_111': abs(mean_spacing - d_111),
                'd_200': abs(mean_spacing - d_200),
                'd_220': abs(mean_spacing - d_220)
            }
            closest_plane = min(distances, key=distances.get)

            print(f"\n  → Best match: {closest_plane} (Δ = {distances[closest_plane]:.3f} Å)")

            # 層数の推定
            if mean_spacing > 0:
                n_layers = thicknesses / mean_spacing
                print(f"\nEstimated number of layers:")
                print(f"  Min: {n_layers.min():.1f}")
                print(f"  Max: {n_layers.max():.1f}")
                print(f"  Mean: {n_layers.mean():.1f}")
        else:
            mean_spacing = None
            std_spacing = None
            print("\n⚠ Insufficient peaks for spacing analysis")
    else:
        peak_thicknesses = None
        mean_spacing = None
        std_spacing = None
        print("\n⚠ No clear peaks detected in thickness distribution")

    print("="*60 + "\n")

    return {
        'histogram': (hist, bin_centers),
        'peaks': peak_thicknesses,
        'mean_spacing': mean_spacing,
        'std_spacing': std_spacing,
        'a0': a0
    }


def calculate_Lambda_stress_field(positions, Lambda_field, atom_types,
                                  E0=193e9, beta=0.5, sigma_y=250e6):
    """
    Λ場から応力場を直接計算（統合式！）

    σ(Λ) = E_0 (1-Λ)^β H(1-Λ)

    Parameters:
        positions: 原子座標 (N, 3)
        Lambda_field: Λ値配列 (N,)
        atom_types: 原子種 (N,)
        E0: ゼロΛでのヤング率 [Pa]
        beta: E(Λ)のべき指数
        sigma_y: 降伏応力 [Pa]

    Returns:
        dict: {
            'stress_field': 応力場 [Pa]
            'E_field': Λ依存ヤング率場 [Pa]
            'regime_map': 各原子のレジーム分類
            'region_stats': 領域別統計
        }
    """
    n_atoms = len(Lambda_field)

    # === レジーム分類 ===
    regime_map = np.zeros(n_atoms, dtype='<U20')
    regime_map[Lambda_field < 0.8] = 'bulk'
    regime_map[(Lambda_field >= 0.8) & (Lambda_field < 1.0)] = 'transition'
    regime_map[(Lambda_field >= 1.0) & (Lambda_field < 1.3)] = 'pre-melt'
    regime_map[Lambda_field >= 1.3] = 'melt'

    # === Λ依存ヤング率 ===
    E_field = np.zeros(n_atoms)
    elastic_mask = Lambda_field < 1.0
    E_field[elastic_mask] = E0 * (1 - Lambda_field[elastic_mask])**beta

    # === 応力計算（弾性領域のみ） ===
    stress_field = np.zeros(n_atoms)

    # ひずみ推定（簡易版：Λ→ε変換）
    strain_field = np.zeros(n_atoms)
    strain_field[elastic_mask] = Lambda_field[elastic_mask] * 0.1  # スケール調整

    # σ = E(Λ) × ε
    stress_field[elastic_mask] = E_field[elastic_mask] * strain_field[elastic_mask]

    # 降伏条件チェック
    yield_mask = np.abs(stress_field) > sigma_y
    stress_field[yield_mask] = np.sign(stress_field[yield_mask]) * sigma_y

    # === 領域別統計 ===
    region_stats = {}
    for regime in ['bulk', 'transition', 'pre-melt', 'melt']:
        mask = regime_map == regime
        if np.sum(mask) > 0:
            region_stats[regime] = {
                'n_atoms': np.sum(mask),
                'Lambda_mean': np.mean(Lambda_field[mask]),
                'E_mean': np.mean(E_field[mask]) / 1e9 if regime != 'melt' else 0,
                'stress_mean': np.mean(stress_field[mask]) / 1e9 if regime in ['bulk', 'transition'] else None
            }

    return {
        'stress_field': stress_field,
        'E_field': E_field,
        'regime_map': regime_map,
        'region_stats': region_stats
    }

def quantify_lattice_compression(analysis_results, E0=193e9, sigma_y=250e6, beta=0.5):
    """
    Λ³理論に基づく格子圧縮の定量化

    統合式：σ(Λ) = E_0 (1-Λ)^β H(1-Λ)

    Parameters:
        analysis_results: dict from analyze_thickness_quantization
        E0: Young's modulus at Λ=0 (Pa) [default: 193 GPa]
        sigma_y: Yield strength (Pa) [default: 250 MPa]
        beta: Exponent for E(Λ) softening [default: 0.5]

    Returns:
        compression_metrics: dict with Λ-consistent strain, stress, and energy
    """
    print("\n" + "="*60)
    print("Λ³-Consistent Lattice Compression Quantification")
    print("="*60)

    mean_spacing = analysis_results['mean_spacing']
    std_spacing = analysis_results['std_spacing']
    a0 = analysis_results['a0']

    if mean_spacing is None:
        print("\n⚠ No spacing data available for compression analysis")
        print("="*60 + "\n")
        return None

    # FCC理論値
    d_111 = a0 / np.sqrt(3)
    d_200 = a0 / 2
    d_220 = a0 / (2*np.sqrt(2))

    # 各理論値に対する圧縮率
    compression_111 = (d_111 - mean_spacing) / d_111
    compression_200 = (d_200 - mean_spacing) / d_200
    compression_220 = (d_220 - mean_spacing) / d_220

    print(f"\nObserved spacing: {mean_spacing:.3f} ± {std_spacing:.3f} Å")
    print(f"\nCompression analysis:")
    print(f"  vs d_{{111}} ({d_111:.3f}Å): {compression_111*100:+.2f}% " +
          f"({'compression' if compression_111 > 0 else 'expansion'})")
    print(f"  vs d_{{200}} ({d_200:.3f}Å): {compression_200*100:+.2f}% " +
          f"({'compression' if compression_200 > 0 else 'expansion'})")
    print(f"  vs d_{{220}} ({d_220:.3f}Å): {compression_220*100:+.2f}% " +
          f"({'compression' if compression_220 > 0 else 'expansion'})")

    # 最も近い面を基準
    distances = {
        'd_111': abs(mean_spacing - d_111),
        'd_200': abs(mean_spacing - d_200),
        'd_220': abs(mean_spacing - d_220)
    }
    closest_plane = min(distances, key=distances.get)

    if closest_plane == 'd_111':
        reference_d = d_111
        compression_ratio = compression_111
        plane_name = '{111}'
    elif closest_plane == 'd_200':
        reference_d = d_200
        compression_ratio = compression_200
        plane_name = '{200}'
    else:
        reference_d = d_220
        compression_ratio = compression_220
        plane_name = '{220}'

    print(f"\n→ Using {plane_name} as reference plane")

    # ひずみ
    strain = compression_ratio
    print(f"\nLattice strain:")
    print(f"  ε = Δd/d₀ = ({reference_d:.3f} - {mean_spacing:.3f})/{reference_d:.3f}")
    print(f"  ε = {strain:.4f} ({abs(strain)*100:.2f}%)")

    # ========================================
    # Λ³理論に基づく応力推定
    # ========================================

    # Λの簡易推定（first-order approximation: ε ≈ Λ for small Λ）
    # より正確には Λ = K/V で決まるが、ここでは観測ひずみから逆算
    Lambda_eff = abs(strain)  # 圧縮の場合は正

    print(f"\n--- Λ³ Theory Integration ---")
    print(f"Effective Λ estimation (from strain):")
    print(f"  Λ_eff ≈ |ε| = {Lambda_eff:.3f}")

    # Λ領域の判定
    if Lambda_eff < 0.8:
        regime = "Bulk (Elastic)"
        regime_desc = "Far from white layer, standard elastic response"
    elif Lambda_eff < 1.0:
        regime = "Transition (Elasto-Plastic)"
        regime_desc = "Near white layer boundary, softening begins"
    elif Lambda_eff < 1.3:
        regime = "Pre-Melt (White Layer)"
        regime_desc = "White layer region, quasi-liquid state"
    else:
        regime = "Melt/Flow"
        regime_desc = "Complete flow, no elastic stress"

    print(f"  Regime: {regime}")
    print(f"  → {regime_desc}")

    # 応力計算（Λ依存）
    if Lambda_eff >= 1.0:
        print(f"\n⚠ Λ ≥ 1.0: Pre-melt/melt regime")
        print(f"  → Elastic modulus E(Λ) → 0")
        print(f"  → No coherent elastic stress (quasi-liquid state)")
        print(f"  → Material in flow/plastic regime")

        stress_GPa = None
        E_Lambda = 0.0
        energy_density = 0.0

        print(f"\n→ Stress cannot be defined in quasi-liquid regime")
        print(f"→ Use flow stress (~0.5-2 GPa) for engineering estimates")

    else:
        # Λ<1: 弾性領域
        # E(Λ) = E_0 (1-Λ)^β
        E_Lambda = E0 * (1 - Lambda_eff)**beta

        print(f"\nΛ-dependent elastic modulus:")
        print(f"  E(Λ) = E_0 (1-Λ)^β")
        print(f"  E({Lambda_eff:.3f}) = {E0/1e9:.0f} × (1-{Lambda_eff:.3f})^{beta}")
        print(f"  E(Λ) = {E_Lambda/1e9:.1f} GPa")
        print(f"  (Reduction: {(1 - E_Lambda/E0)*100:.1f}% from E_0)")

        # σ(Λ) = E(Λ) × ε
        stress_Pa = E_Lambda * strain
        stress_GPa = stress_Pa / 1e9

        print(f"\nStress calculation:")
        print(f"  σ(Λ) = E(Λ) × ε")
        print(f"  σ = {E_Lambda/1e9:.1f} GPa × {strain:.4f}")
        print(f"  σ = {stress_GPa:+.2f} GPa")

        # 降伏条件チェック
        sigma_y_GPa = sigma_y / 1e9
        elastic_limit_strain = sigma_y / E_Lambda

        print(f"\nYield condition check:")
        print(f"  σ_y = {sigma_y_GPa:.2f} GPa")
        print(f"  ε_y = σ_y/E(Λ) = {elastic_limit_strain:.4f} ({elastic_limit_strain*100:.2f}%)")

        if abs(stress_GPa) > sigma_y_GPa:
            print(f"  ⚠ Calculated stress ({abs(stress_GPa):.2f} GPa) > yield strength")
            print(f"  → Plastic deformation regime")
            print(f"  → Actual stress saturates at σ_y ≈ {sigma_y_GPa:.2f} GPa")

            # 応力を降伏応力で飽和
            stress_GPa = sigma_y_GPa if stress_GPa > 0 else -sigma_y_GPa
            stress_Pa = stress_GPa * 1e9
        else:
            print(f"  ✓ Elastic regime (σ < σ_y)")

        # エネルギー密度
        # U(Λ) = (1/2) E(Λ) ε²
        energy_density = 0.5 * E_Lambda * strain**2
        energy_density_MJ = energy_density / 1e6

        print(f"\nElastic energy density:")
        print(f"  U(Λ) = (1/2) E(Λ) ε²")
        print(f"  U = {energy_density_MJ:.2f} MJ/m³")

    # 比較
    print(f"\n--- Comparison with Physical Regimes ---")
    print(f"Typical stress ranges:")
    print(f"  Elastic deformation: 0-0.3 GPa")
    print(f"  Plastic flow (cutting): 0.5-2 GPa")
    print(f"  Hydrostatic pressure: 1-5 GPa")
    print(f"  Phase transformation: 2-10 GPa")

    if stress_GPa is not None:
        if stress_GPa < 0.3:
            print(f"  → Observed ({abs(stress_GPa):.2f} GPa): Elastic regime")
        elif stress_GPa < 2.0:
            print(f"  → Observed ({abs(stress_GPa):.2f} GPa): Plastic flow regime")
        elif stress_GPa < 5.0:
            print(f"  → Observed ({abs(stress_GPa):.2f} GPa): High-pressure machining")
        else:
            print(f"  → Observed ({abs(stress_GPa):.2f} GPa): Phase transformation likely")

    print("="*60 + "\n")

    return {
        'mean_spacing': mean_spacing,
        'std_spacing': std_spacing,
        'reference_plane': plane_name,
        'reference_d': reference_d,
        'compression_ratio': compression_ratio,
        'strain': strain,
        'Lambda_eff': Lambda_eff,
        'regime': regime,
        'E_Lambda': E_Lambda,
        'stress_Pa': stress_Pa if stress_GPa is not None else None,
        'stress_GPa': stress_GPa,
        'energy_density': energy_density,
        'E0': E0,
        'beta': beta,
        'sigma_y': sigma_y
    }

def plot_Lambda_stress_correlation(positions, Lambda_field, stress_results):
    """
    Λ場と応力場の相関可視化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    stress_field = stress_results['stress_field']
    E_field = stress_results['E_field']
    regime_map = stress_results['regime_map']

    # 1. Λ vs 応力（散布図）
    ax1 = axes[0, 0]
    elastic_mask = Lambda_field < 1.0
    ax1.scatter(Lambda_field[elastic_mask],
                stress_field[elastic_mask]/1e9,
                c=E_field[elastic_mask]/1e9,
                cmap='viridis', s=10, alpha=0.6)
    ax1.axvline(x=1.0, color='red', linestyle='--', label='Λ=1 (Critical)')
    ax1.set_xlabel('Λ', fontsize=12)
    ax1.set_ylabel('Stress [GPa]', fontsize=12)
    ax1.set_title('σ(Λ) = E₀(1-Λ)^β H(1-Λ)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. E(Λ)分布
    ax2 = axes[0, 1]
    ax2.scatter(Lambda_field[elastic_mask],
                E_field[elastic_mask]/1e9,
                s=10, alpha=0.6, color='orange')
    ax2.set_xlabel('Λ', fontsize=12)
    ax2.set_ylabel('E(Λ) [GPa]', fontsize=12)
    ax2.set_title('Λ-Dependent Young\'s Modulus', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 空間分布（z方向）
    ax3 = axes[1, 0]
    z_coords = positions[:, 2]
    scatter = ax3.scatter(z_coords[elastic_mask],
                         stress_field[elastic_mask]/1e9,
                         c=Lambda_field[elastic_mask],
                         cmap='hot', s=20, alpha=0.7)
    plt.colorbar(scatter, ax=ax3, label='Λ')
    ax3.set_xlabel('Depth z [Å]', fontsize=12)
    ax3.set_ylabel('Stress [GPa]', fontsize=12)
    ax3.set_title('Stress vs Depth (with Λ)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. レジーム別統計
    ax4 = axes[1, 1]
    stats = stress_results['region_stats']
    regimes = list(stats.keys())
    n_atoms = [stats[r]['n_atoms'] for r in regimes]
    colors_map = {'bulk': 'blue', 'transition': 'orange',
                  'pre-melt': 'red', 'melt': 'purple'}
    colors = [colors_map.get(r, 'gray') for r in regimes]

    ax4.bar(regimes, n_atoms, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Number of Atoms', fontsize=12)
    ax4.set_title('Regime Classification', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # ✅ 修正版：普通にループで構築
    textstr_lines = []
    for r in regimes:
        textstr_lines.append(f"{r.capitalize()}:")
        textstr_lines.append(f"  N={stats[r]['n_atoms']}")
        textstr_lines.append(f"  Λ={stats[r]['Lambda_mean']:.2f}")

        # E_meanがあるかチェック
        if stats[r]['E_mean'] is not None and stats[r]['E_mean'] > 0:
            textstr_lines.append(f"  E={stats[r]['E_mean']:.1f} GPa")
        else:
            textstr_lines.append("  E=0 (flow)")

    textstr = '\n'.join(textstr_lines)

    ax4.text(0.95, 0.95, textstr, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def plot_thickness_quantization(df, analysis_results):
    """
    厚さの量子化を可視化

    Parameters:
        df: DataFrame with 'thickness' and 'lambda' columns
        analysis_results: dict from analyze_thickness_quantization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    hist, bin_centers = analysis_results['histogram']
    peaks = analysis_results['peaks']
    mean_spacing = analysis_results['mean_spacing']
    a0 = analysis_results['a0']

    # 1. ヒストグラム
    ax1 = axes[0]
    ax1.bar(bin_centers, hist, width=np.diff(bin_centers)[0]*0.8,
            alpha=0.7, color='skyblue', edgecolor='black')

    if peaks is not None and len(peaks) > 0:
        # ピークをマーク
        for peak_thick in peaks:
            ax1.axvline(peak_thick, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax1.text(peak_thick, ax1.get_ylim()[1]*0.9, f'{peak_thick:.1f}Å',
                    rotation=90, va='top', ha='right', fontsize=10, color='red')

    ax1.set_xlabel('White Layer Thickness [Å]', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Thickness Distribution (Quantization)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Thickness vs λ with layer annotations
    ax2 = axes[1]
    ax2.scatter(df['lambda'], df['thickness'], s=100, alpha=0.6, color='blue')

    if peaks is not None and len(peaks) > 0:
        # 各層をハイライト
        for i, peak_thick in enumerate(peaks):
            ax2.axhline(peak_thick, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax2.text(df['lambda'].max()*0.95, peak_thick, f'Layer {i+1}',
                    va='bottom', ha='right', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Decay Length λ [Å]', fontsize=12)
    ax2.set_ylabel('White Layer Thickness [Å]', fontsize=12)
    ax2.set_title('Layer-by-Layer Growth', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Layer spacing comparison
    ax3 = axes[2]

    # FCC理論値
    d_111 = a0 / np.sqrt(3)
    d_200 = a0 / 2
    d_220 = a0 / (2*np.sqrt(2))

    planes = ['d_{111}', 'd_{200}', 'd_{220}']
    d_values = [d_111, d_200, d_220]
    colors = ['blue', 'green', 'orange']

    x_pos = np.arange(len(planes))
    ax3.bar(x_pos, d_values, color=colors, alpha=0.7, edgecolor='black', label='FCC Theory')

    if mean_spacing is not None:
        ax3.axhline(mean_spacing, color='red', linestyle='--', linewidth=3,
                   label=f'Observed: {mean_spacing:.2f}Å')
        ax3.fill_between([-0.5, len(planes)-0.5],
                        mean_spacing - analysis_results['std_spacing'],
                        mean_spacing + analysis_results['std_spacing'],
                        color='red', alpha=0.2)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(planes, fontsize=12)
    ax3.set_ylabel('Spacing [Å]', fontsize=12)
    ax3.set_title('Layer Spacing vs FCC Lattice', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle('White Layer Quantization Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_depth_profile_enhanced(positions, Lambda_field,
                                stress_results, z_max=25):
    """
    深さ方向の完全プロファイル（完全修正版）
    """
    z_coords = positions[:, 2]
    z_min = z_coords.min()
    depth = z_coords - z_min

    # ビン分割
    n_bins = 30
    z_bins = np.linspace(0, z_max, n_bins+1)

    # ✅ 全てのプロファイルを同時に構築
    Lambda_profile = []
    E_profile = []
    stress_profile = []
    z_centers = []
    regime_profiles = {'bulk': [], 'transition': [], 'pre-melt': [], 'melt': []}

    E_field = stress_results['E_field']
    stress_field = stress_results['stress_field']
    regime_map = stress_results['regime_map']

    # ✅ 統一されたループ（データがあるビンのみ）
    for i in range(n_bins):
        mask = (depth >= z_bins[i]) & (depth < z_bins[i+1])

        if np.sum(mask) > 0:  # ← 同じ条件で全て追加
            # 基本プロファイル
            Lambda_profile.append(Lambda_field[mask].mean())
            E_profile.append(E_field[mask].mean() / 1e9)
            stress_profile.append(stress_field[mask].mean() / 1e9)
            z_centers.append((z_bins[i] + z_bins[i+1]) / 2)

            # レジーム分布
            total = np.sum(mask)
            for regime in regime_profiles.keys():
                mask_regime = mask & (regime_map == regime)
                fraction = np.sum(mask_regime) / total * 100
                regime_profiles[regime].append(fraction)

    # numpy配列化
    z_centers = np.array(z_centers)
    Lambda_profile = np.array(Lambda_profile)
    E_profile = np.array(E_profile)
    stress_profile = np.array(stress_profile)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========================================
    # 1. Λプロファイル（背景色付き）
    # ========================================
    ax1 = axes[0, 0]

    # 背景領域
    ax1.axhspan(0, 0.8, alpha=0.15, color='blue', label='Bulk (Λ<0.8)')
    ax1.axhspan(0.8, 1.0, alpha=0.15, color='orange', label='Transition (0.8≤Λ<1.0)')
    ax1.axhspan(1.0, 1.3, alpha=0.15, color='red', label='Pre-melt (1.0≤Λ<1.3)')
    ax1.axhspan(1.3, 2.0, alpha=0.15, color='purple', label='Melt (Λ≥1.3)')

    # 境界線
    ax1.axhline(0.8, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(1.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(1.3, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)

    # データプロット
    ax1.plot(z_centers, Lambda_profile, 'o-', linewidth=2.5,
             markersize=6, color='black', zorder=10)

    ax1.set_xlabel('Depth from Surface [Å]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Λ', fontsize=12, fontweight='bold')
    ax1.set_title('Λ Depth Profile', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(Lambda_profile.max()*1.1, 1.5)])

    # ========================================
    # 2. E(Λ)プロファイル
    # ========================================
    ax2 = axes[0, 1]

    ax2.plot(z_centers, E_profile, 's-', linewidth=2.5,
             markersize=6, color='green', label='E(Λ)')
    ax2.axhline(193, color='gray', linestyle='--', linewidth=2,
                alpha=0.5, label='E₀ (193 GPa)')

    ax2.set_xlabel('Depth from Surface [Å]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('E(Λ) [GPa]', fontsize=12, fontweight='bold')
    ax2.set_title('Young\'s Modulus Profile', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 210])

    # ========================================
    # 3. 応力プロファイル
    # ========================================
    ax3 = axes[1, 0]

    ax3.plot(z_centers, stress_profile, '^-', linewidth=2.5,
             markersize=6, color='blue', label='σ(Λ)')
    ax3.axhline(0.25, color='red', linestyle='--', linewidth=2,
                label='σ_y (0.25 GPa)')
    ax3.fill_between(z_centers, 0, stress_profile, alpha=0.3, color='blue')

    ax3.set_xlabel('Depth from Surface [Å]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('σ [GPa]', fontsize=12, fontweight='bold')
    ax3.set_title('Stress Profile', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, max(0.3, stress_profile.max()*1.1)])

    # ========================================
    # 4. レジーム比率（積み上げ棒グラフ）
    # ========================================
    ax4 = axes[1, 1]

    colors_map = {'bulk': 'blue', 'transition': 'orange',
                  'pre-melt': 'red', 'melt': 'purple'}

    bar_width = (z_bins[1] - z_bins[0]) * 0.9
    bottom = np.zeros(len(z_centers))

    for regime, color in colors_map.items():
        values = np.array(regime_profiles[regime])
        ax4.bar(z_centers, values, bottom=bottom,
                color=color, alpha=0.7, edgecolor='black', linewidth=0.5,
                label=regime.capitalize(), width=bar_width)
        bottom += values

    ax4.set_xlabel('Depth from Surface [Å]', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Regime Fraction [%]', fontsize=12, fontweight='bold')
    ax4.set_title('Regime Distribution vs Depth', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 100])

    plt.suptitle('Depth Profile Analysis (K0=1.2, λ=15.0)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Λ³-Percolation Sensitivity Analysis Suite")
    print("   Phase 1 & Phase 2 - Complete Optimized Version")
    print("="*60 + "\n")

    # ====================================
    # Phase 1-1: K0 sensitivity
    # ====================================
    df_K0 = sensitivity_K0(
        K0_range=np.linspace(0.8, 3.0, 12),
        Nx=30, Ny=30, Nz=6
    )
    plot_K0_sensitivity(df_K0)
    df_K0.to_csv('/sensitivity_K0.csv', index=False)
    print("✓ K0 sensitivity results saved\n")

    # ====================================
    # Phase 1-2: λ sensitivity (Standard)
    # ====================================
    df_lambda = sensitivity_lambda(
        lambda_range=np.linspace(10, 40, 16),
        Nx=30, Ny=30, Nz=6
    )
    plot_lambda_sensitivity(df_lambda)
    df_lambda.to_csv('/sensitivity_lambda.csv', index=False)
    print("✓ λ sensitivity (standard) results saved\n")

    # ====================================
    # Phase 1-2 (Optimized): λ sensitivity
    # ====================================
    df_lambda_opt = sensitivity_lambda_optimized(
        lambda_range=np.linspace(5, 20, 21),
        Nx=30, Ny=30, Nz=6
    )
    slope, intercept, r_value = plot_lambda_sensitivity_improved(df_lambda_opt)
    df_lambda_opt.to_csv('/sensitivity_lambda_optimized.csv',
                         index=False)
    print("✓ λ sensitivity (optimized) results saved\n")

    # 理論値との比較
    theoretical_comparison(slope, df_lambda_opt)

    # ====================================
    # Layer Quantization Analysis
    # ====================================
    print("\n" + "="*60)
    print("🔬 Analyzing discrete layer structure...")
    print("="*60 + "\n")

    quantization_results = analyze_thickness_quantization(df_lambda_opt, a0=3.6)
    plot_thickness_quantization(df_lambda_opt, quantization_results)

    # 圧縮率の定量化
    compression_metrics = quantify_lattice_compression(quantization_results, E0=193e9)

    # 結果を保存
    if quantization_results['peaks'] is not None:
        peaks_df = pd.DataFrame({
            'peak_thickness': quantization_results['peaks'],
            'peak_index': range(1, len(quantization_results['peaks'])+1)
        })
        peaks_df.to_csv('/layer_quantization_peaks.csv', index=False)
        print("✓ Layer quantization analysis saved\n")

    if compression_metrics is not None:
        compression_df = pd.DataFrame([compression_metrics])
        compression_df.to_csv('/lattice_compression_metrics.csv', index=False)
        print("✓ Compression metrics saved\n")

    # ====================================
    # Phase 1-3: Λ_c sensitivity
    # ====================================
    df_Lc = sensitivity_Lc(
        Lc_range=np.linspace(0.85, 1.15, 13),
        Nx=30, Ny=30, Nz=6
    )
    plot_Lc_sensitivity(df_Lc)
    df_Lc.to_csv('/sensitivity_Lc.csv', index=False)
    print("✓ Λ_c sensitivity results saved\n")

    # ====================================
    # Phase 2-1: Vacancy (Standard)
    # ====================================
    df_vac = sensitivity_vacancy_critical(
        vacancy_range=np.linspace(0, 0.20, 21),
        K0=1.2, lambda_decay=15.0, Lc=0.9
    )
    df_vac.to_csv('/sensitivity_vacancy.csv', index=False)
    print("✓ Vacancy sensitivity (standard) results saved\n")

    # ====================================
    # Phase 2-2: Cr substitution (Standard)
    # ====================================
    df_Cr = sensitivity_Cr_substitution_critical(
        Cr_range=np.linspace(0, 0.25, 21),
        K0=1.2, lambda_decay=15.0, Lc=0.9
    )
    df_Cr.to_csv('/sensitivity_Cr.csv', index=False)
    print("✓ Cr substitution sensitivity (standard) results saved\n")

    # Phase 2 combined plot (Standard)
    plot_defect_sensitivity(df_vac, df_Cr)

    # ====================================
    # Phase 2-1 (Critical): Vacancy
    # ====================================
    df_vac_crit = sensitivity_vacancy_critical(
        vacancy_range=np.linspace(0, 0.10, 21),
        Nx=30, Ny=30, Nz=6,
        K0=1.5, lambda_decay=16.0
    )
    df_vac_crit.to_csv('/sensitivity_vacancy_critical.csv',
                       index=False)
    print("✓ Vacancy sensitivity (critical) results saved\n")

    # ====================================
    # Phase 2-2 (Critical): Cr substitution
    # ====================================
    df_Cr_crit = sensitivity_Cr_substitution_critical(
        Cr_range=np.linspace(0, 0.15, 21),
        Nx=30, Ny=30, Nz=6,
        K0=1.5, lambda_decay=16.0
    )
    df_Cr_crit.to_csv('/sensitivity_Cr_critical.csv',
                      index=False)
    print("✓ Cr substitution sensitivity (critical) results saved\n")

    # Phase 2 combined plot (Critical)
    plot_defect_sensitivity_critical(df_vac_crit, df_Cr_crit)

    # ====================================
    # Λ³統合式による応力場解析（修正版！）
    # ====================================
    print("\n" + "="*60)
    print("🔬 Λ³-Consistent Stress Field Analysis")
    print("="*60 + "\n")

    # ✅ 完全版シミュレーション実行
    print("Running full simulation for stress field analysis...")
    full_sim = run_full_simulation(
        Nx=30, Ny=30, Nz=6,
        K0=1.2, lambda_decay=15.0, Lambda_threshold=0.9,
        verbose=True
    )

    # データ展開
    positions = full_sim['positions']
    atom_types = full_sim['atom_types']
    Lambda_field = full_sim['Lambda_field']
    K_field = full_sim['K_field']
    V_eff = full_sim['V_eff']

    print(f"✓ Simulation complete!")
    print(f"  - Atoms: {len(positions)}")
    print(f"  - Λ range: {Lambda_field.min():.2f} - {Lambda_field.max():.2f}")
    print(f"  - K range: {K_field.min():.2f} - {K_field.max():.2f}\n")

    # 応力場計算
    print("Calculating Λ³-consistent stress field...")
    stress_results = calculate_Lambda_stress_field(
        positions, Lambda_field, atom_types,
        E0=193e9, beta=0.5
    )
    print("✓ Stress field calculated!\n")

    # レジーム統計表示
    print("Regime Statistics:")
    print("-" * 40)
    for regime, stats in stress_results['region_stats'].items():
        print(f"{regime.capitalize()}:")
        print(f"  N atoms: {stats['n_atoms']}")
        print(f"  Λ mean: {stats['Lambda_mean']:.3f}")
        if stats['E_mean'] > 0:
            print(f"  E mean: {stats['E_mean']:.1f} GPa")
        else:
            print(f"  E mean: 0 (flow state)")
        if stats['stress_mean'] is not None:
            print(f"  σ mean: {stats['stress_mean']:.2f} GPa")
        print()

    # 可視化
    print("Generating visualizations...")
    plot_Lambda_stress_correlation(positions, Lambda_field, stress_results)
    plot_depth_profile_enhanced(positions, Lambda_field, stress_results, z_max=25)

    # CSV保存
    print("Saving results...")
    stress_df = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
        'Lambda': Lambda_field,
        'K': K_field,
        'V_eff': V_eff,
        'E_GPa': stress_results['E_field'] / 1e9,
        'stress_GPa': stress_results['stress_field'] / 1e9,
        'regime': stress_results['regime_map']
    })
    stress_df.to_csv('/Lambda_stress_field.csv', index=False)
    print("✓ Λ³ stress field saved to /Lambda_stress_field.csv\n")

    print("="*60)
    print("✨ Λ³-Consistent Stress Analysis Complete!")
    print("="*60)
