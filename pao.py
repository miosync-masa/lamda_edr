"""
PAO Noncommutative Boundary Diagnostics
AdS/CFT対応版（環が作ったよ！）

FLCの非可換境界診断を2次元空間(T,P)に拡張
"""

import numpy as np
from scipy.interpolate import interp1d, griddata
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

# =============================================================================
# Section 1: 2D版Ξパケット拡張計算
# =============================================================================

def compute_boundary_info_packet_pao_enhanced(
    params_dict, 
    boundary_points: List[Tuple[float, float]], 
    exp_data,
    delta_n: float = 0.5,  # 法線方向の微分幅
    n_samples_curvature: int = 5  # 曲率計算用のサンプル数
):
    """
    ★拡張版PAO境界情報パケット★
    
    FLCのΞパケットを2次元に拡張：
    - grad_n_Lambda: 法線方向の勾配
    - j_n: フラックス（法線成分）
    - omega_Lambda: 渦度（2階微分）
    - curvature: 境界Σの曲率
    
    Args:
        params_dict: パラメータ辞書
        boundary_points: 境界Σの点列 [(T1,P1), (T2,P2), ...]
        exp_data: 実験データ
        delta_n: 法線方向の数値微分幅
        n_samples_curvature: 曲率計算用サンプル数
    
    Returns:
        Ξ: 拡張境界情報パケット
    """
    from pao_holographic_fixed import (
        compute_Lambda_field_pao_ultimate,
        compute_K_pao,
        compute_V_pao
    )
    
    if len(boundary_points) == 0:
        return {}
    
    Ξ = {}
    Ξ['Sigma'] = boundary_points
    
    grad_norms = []
    j_n_values = []
    omega_values = []
    curvature_values = []
    normal_vectors = []
    
    for idx, (T, P) in enumerate(boundary_points):
        # === 1. 勾配計算（数値微分） ===
        dT, dP = 0.5, 5.0
        
        Lambda_0 = compute_Lambda_field_pao_ultimate(params_dict, T, P, exp_data)
        Lambda_T_plus = compute_Lambda_field_pao_ultimate(params_dict, T + dT, P, exp_data)
        Lambda_T_minus = compute_Lambda_field_pao_ultimate(params_dict, T - dT, P, exp_data)
        Lambda_P_plus = compute_Lambda_field_pao_ultimate(params_dict, T, P + dP, exp_data)
        Lambda_P_minus = compute_Lambda_field_pao_ultimate(params_dict, T, P - dP, exp_data)
        
        # 勾配ベクトル ∇Λ = (∂Λ/∂T, ∂Λ/∂P)
        grad_T = (Lambda_T_plus - Lambda_T_minus) / (2 * dT)
        grad_P = (Lambda_P_plus - Lambda_P_minus) / (2 * dP)
        grad_norm = np.sqrt(grad_T**2 + grad_P**2)
        
        # 法線ベクトル n = ∇Λ / |∇Λ|
        if grad_norm > 1e-10:
            n_T = grad_T / grad_norm
            n_P = grad_P / grad_norm
        else:
            n_T, n_P = 1.0, 0.0  # デフォルト
        
        normal_vectors.append((n_T, n_P))
        grad_norms.append(grad_norm)
        
        # === 2. フラックス（法線成分）===
        # j_n = -|∇Λ| （境界から外向きが正）
        j_n = -grad_norm
        j_n_values.append(j_n)
        
        # === 3. 渦度（2階微分）===
        # ω_Λ ≈ ∂²Λ/∂n² （法線方向の2階微分）
        # 法線方向にサンプリングして2階差分
        Lambda_n_minus = compute_Lambda_field_pao_ultimate(
            params_dict, T - delta_n * n_T, P - delta_n * n_P, exp_data
        )
        Lambda_n_plus = compute_Lambda_field_pao_ultimate(
            params_dict, T + delta_n * n_T, P + delta_n * n_P, exp_data
        )
        
        # 2階差分近似
        omega = (Lambda_n_plus - 2*Lambda_0 + Lambda_n_minus) / (delta_n**2)
        omega_values.append(omega)
        
        # === 4. 境界Σの曲率 ===
        # 近傍の境界点から曲率を推定
        if len(boundary_points) >= n_samples_curvature:
            # 現在の点の前後でサンプリング
            start_idx = max(0, idx - n_samples_curvature//2)
            end_idx = min(len(boundary_points), idx + n_samples_curvature//2 + 1)
            
            local_points = boundary_points[start_idx:end_idx]
            
            if len(local_points) >= 3:
                # 曲率の簡易推定：境界曲線のフィッティングから
                T_local = np.array([t for t, p in local_points])
                P_local = np.array([p for t, p in local_points])
                
                # 2次多項式フィット
                if len(set(T_local)) >= 3:  # T方向に変化がある場合
                    coeffs = np.polyfit(T_local, P_local, 2)
                    kappa = abs(2 * coeffs[0]) / (1 + coeffs[1]**2)**(1.5)
                else:
                    kappa = 0.0
            else:
                kappa = 0.0
        else:
            kappa = 0.0
        
        curvature_values.append(kappa)
    
    # 結果を格納
    Ξ['grad_n_Lambda'] = np.array(grad_norms)
    Ξ['j_n'] = np.array(j_n_values)
    Ξ['omega_Lambda'] = np.array(omega_values)
    Ξ['curvature'] = np.array(curvature_values)
    Ξ['normal_vectors'] = normal_vectors
    
    # 境界上の(T,P)座標
    Ξ['O_T'] = np.array([T for T, P in boundary_points])
    Ξ['O_P'] = np.array([P for T, P in boundary_points])
    
    # 境界上のΛ値
    Ξ['O_Lambda'] = np.array([
        compute_Lambda_field_pao_ultimate(params_dict, T, P, exp_data)
        for T, P in boundary_points
    ])
    
    # 統計量
    Ξ['grad_n_mean'] = float(np.mean(grad_norms))
    Ξ['grad_n_std'] = float(np.std(grad_norms))
    Ξ['grad_n_max'] = float(np.max(grad_norms))
    Ξ['grad_n_min'] = float(np.min(grad_norms))
    Ξ['num_points'] = len(boundary_points)
    
    return Ξ

# =============================================================================
# Section 2: 非可換パラメータθ_eff（2D版）
# =============================================================================

def compute_theta_eff_pao(Xi_packet: Dict, epsilon: float = 1e-6) -> np.ndarray:
    """
    ★PAO用非可換パラメータθ_effの計算★
    
    FLCと同じ定義式：
    θ_eff = ω_Λ / (|∂_nΛ| × |j_n| + ε)
    
    物理的意味（2D版も同じ）:
      - 渦が強い（ω_Λ大）→ 非可換性大
      - 境界が硬い（|∂_nΛ|大）→ 非可換性小
      - 駆動が強い（|j_n|大）→ 非可換性小
    
    Args:
        Xi_packet: 拡張境界情報パケット
        epsilon: ゼロ除算防止
    
    Returns:
        theta_eff: 各境界点での非可換パラメータ [array]
    """
    omega = Xi_packet['omega_Lambda']
    grad_n = Xi_packet['grad_n_Lambda']
    flux_n = Xi_packet['j_n']
    
    # 非可換パラメータの計算（FLCと同じ）
    denominator = np.abs(grad_n) * (np.abs(flux_n) + epsilon)
    theta_eff = omega / (denominator + epsilon)
    
    return theta_eff

# =============================================================================
# Section 3: 非可換シグネチャΔ_NC（2D版）
# =============================================================================

def compute_noncommutative_signature_pao(
    boundary_points: List[Tuple[float, float]],
    theta_eff: np.ndarray,
    field_f: np.ndarray,
    field_g: np.ndarray,
) -> Dict:
    """
    ★PAO用非可換性のシグネチャΔ_NC★
    
    FLCと同じ定義式：
    Δ_NC = Σ[f_{i+1}g_i - f_ig_{i+1}]θ_eff(i)
    
    可換なら Δ_NC = 0
    非可換なら Δ_NC ≠ 0
    
    Args:
        boundary_points: 境界Σの点列
        theta_eff: 非可換パラメータの配列
        field_f: 第1の場（例：予測誤差場）
        field_g: 第2の場（例：マージン |1-Λ|）
    
    Returns:
        result: {
            'Delta_NC': 総和,
            'contributions': 各点の寄与,
            'mean_abs': 平均絶対値,
            'std': 標準偏差,
            'max_abs': 最大絶対値,
        }
    """
    n = len(boundary_points)
    
    # 順序依存項の計算
    delta_nc = 0.0
    contributions = []
    
    for i in range(n - 1):
        # 非可換項: [f, g]_θ = (f_{i+1}g_i - f_ig_{i+1})θ_eff
        nc_term = (field_f[i+1] * field_g[i] -
                   field_f[i] * field_g[i+1]) * theta_eff[i]
        delta_nc += nc_term
        contributions.append(nc_term)
    
    contributions = np.array(contributions)
    
    # 統計量
    result = {
        'Delta_NC': delta_nc,
        'contributions': contributions,
        'mean_abs': np.mean(np.abs(contributions)),
        'std': np.std(contributions),
        'max_abs': np.max(np.abs(contributions)),
    }
    
    return result

# =============================================================================
# Section 4: 境界Σ近傍サンプリング（2D版）
# =============================================================================

def sample_boundary_neighborhood_pao(
    boundary_points: List[Tuple[float, float]],
    normal_vectors: List[Tuple[float, float]],
    local_width: float = 5.0,  # 法線方向の幅（温度単位）
    n_samples: int = 20,
) -> Dict:
    """
    ★境界Σ近傍の法線方向サンプリング（2D版）★
    
    各境界点で法線方向に±local_widthの範囲をサンプリング
    
    Args:
        boundary_points: 境界Σの点列 [(T1,P1), ...]
        normal_vectors: 各点での法線ベクトル [(nT1,nP1), ...]
        local_width: 法線方向のサンプリング幅
        n_samples: 各点での法線方向サンプル数
    
    Returns:
        samples: {
            'T': T座標配列,
            'P': P座標配列,
            'boundary_idx': 各サンプルが属する境界点のindex,
            'distance': 境界からの法線距離,
        }
    """
    T_samples = []
    P_samples = []
    boundary_indices = []
    distances = []
    
    for idx, ((T, P), (nT, nP)) in enumerate(zip(boundary_points, normal_vectors)):
        # 法線方向に±local_widthでサンプリング
        dist_range = np.linspace(-local_width, local_width, n_samples)
        
        for d in dist_range:
            T_sample = T + d * nT
            P_sample = P + d * nP
            
            T_samples.append(T_sample)
            P_samples.append(P_sample)
            boundary_indices.append(idx)
            distances.append(d)
    
    return {
        'T': np.array(T_samples),
        'P': np.array(P_samples),
        'boundary_idx': np.array(boundary_indices),
        'distance': np.array(distances),
    }

# =============================================================================
# Section 5: 完全診断関数（2D版）
# =============================================================================

def diagnose_noncommutative_boundary_pao(
    params: Dict,
    boundary_points: List[Tuple[float, float]],
    exp_data,
    local_width: float = 5.0,
    n_local: int = 20,
    verbose: bool = True,
) -> Dict:
    """
    ★PAO用非可換境界の完全診断★
    
    FLCの diagnose_noncommutative_boundary_local を2次元に拡張
    
    Args:
        params: 最適化されたパラメータ辞書
        boundary_points: 境界Σの点列 [(T1,P1), ...]
        exp_data: 実験データ（BurnoutData のリスト）
        local_width: Σ近傍の幅（温度単位）
        n_local: 各Σ点での法線方向サンプル数
        verbose: 結果表示のON/OFF
    
    Returns:
        result: 診断結果辞書
    """
    from pao_holographic_fixed import compute_Lambda_field_pao_ultimate
    
    if len(boundary_points) == 0:
        print("警告: 境界Σが検出されませんでした")
        return {}
    
    if verbose:
        print(f"\n境界Σ検出: {len(boundary_points)}点")
        print(f"  T範囲: [{min(t for t,p in boundary_points):.1f}, "
              f"{max(t for t,p in boundary_points):.1f}]°C")
        print(f"  P範囲: [{min(p for t,p in boundary_points):.1f}, "
              f"{max(p for t,p in boundary_points):.1f}] MPa")
    
    # 1. 拡張Ξパケット計算
    Xi_packet = compute_boundary_info_packet_pao_enhanced(
        params, boundary_points, exp_data
    )
    
    # 2. θ_effの計算
    theta_eff = compute_theta_eff_pao(Xi_packet)
    
    # 3. 境界Σ近傍サンプリング
    normal_vectors = Xi_packet['normal_vectors']
    samples = sample_boundary_neighborhood_pao(
        boundary_points, normal_vectors, local_width, n_local
    )
    
    if verbose:
        print(f"\n近傍サンプリング:")
        print(f"  法線方向幅: ±{local_width}°C")
        print(f"  各Σ点あたり: {n_local}点")
        print(f"  総サンプル数: {len(samples['T'])}点")
    
    # 4. 近傍でのΛ値計算
    Lambda_samples = np.array([
        compute_Lambda_field_pao_ultimate(params, T, P, exp_data)
        for T, P in zip(samples['T'], samples['P'])
    ])
    
    # 5. 誤差場とマージン場の構築
    # PAOは二値分類なので、「予測の確信度」を誤差場とする
    
    # 動的閾値の計算
    Lambda_safe = []
    Lambda_danger = []
    for d in exp_data:
        Lambda = compute_Lambda_field_pao_ultimate(params, d.temperature, d.pressure, exp_data)
        if d.burnout:
            Lambda_danger.append(Lambda)
        else:
            Lambda_safe.append(Lambda)
    
    if len(Lambda_safe) > 0 and len(Lambda_danger) > 0:
        threshold = (np.max(Lambda_safe) + np.min(Lambda_danger)) / 2.0
    else:
        threshold = 0.9
    
    # 境界上での誤差場：Λからの距離
    Lambda_boundary = Xi_packet['O_Lambda']
    error_field_boundary = Lambda_boundary - threshold  # 閾値からのずれ
    
    # マージン場：|1-Λ|（Λ=1からの距離）
    margin_field_boundary = np.abs(1.0 - Lambda_boundary)
    
    # 6. 非可換性の検出
    if len(boundary_points) >= 2:
        nc_signature = compute_noncommutative_signature_pao(
            boundary_points, theta_eff, error_field_boundary, margin_field_boundary
        )
    else:
        nc_signature = {
            'Delta_NC': 0.0,
            'contributions': np.array([]),
            'mean_abs': 0.0,
            'std': 0.0,
            'max_abs': 0.0
        }
    
    # 7. 結果の整理
    result = {
        'boundary_points': boundary_points,
        'theta_eff': theta_eff,
        'Xi_packet': Xi_packet,
        'nc_signature': nc_signature,
        'samples': samples,
        'Lambda_samples': Lambda_samples,
        'Lambda_boundary': Lambda_boundary,
        'error_field_boundary': error_field_boundary,
        'margin_field_boundary': margin_field_boundary,
        'threshold': threshold,
    }
    
    # 8. 結果表示
    if verbose:
        print("\n" + "="*60)
        print("非可換境界診断（AdS/CFT対応・PAO版）")
        print("="*60)
        
        print(f"\n【境界Σ情報】")
        print(f"  検出点数: {len(boundary_points)}点")
        print(f"  境界長さ（近似）: {estimate_boundary_length(boundary_points):.1f}")
        
        print(f"\n【近傍評価領域】")
        print(f"  Λ範囲: [{np.min(Lambda_samples):.3f}, {np.max(Lambda_samples):.3f}]")
        print(f"  動的閾値: {threshold:.3f}")
        
        print(f"\n【非可換パラメータ θ_eff】")
        print(f"  平均: {np.mean(np.abs(theta_eff)):.6e}")
        print(f"  最大: {np.max(np.abs(theta_eff)):.6e}")
        print(f"  最小: {np.min(np.abs(theta_eff)):.6e}")
        print(f"  標準偏差: {np.std(theta_eff):.6e}")
        
        print(f"\n【非可換シグネチャ Δ_NC】")
        print(f"  Δ_NC = {nc_signature['Delta_NC']:.6e}")
        print(f"  平均寄与: {nc_signature['mean_abs']:.6e}")
        print(f"  最大寄与: {nc_signature['max_abs']:.6e}")
        
        # 判定
        if np.abs(nc_signature['Delta_NC']) > 1e-6:
            print(f"  ✓ 非可換性検出！（Δ_NC ≠ 0）")
        else:
            print(f"  - 可換極限近傍（Δ_NC ≈ 0）")
        
        print(f"\n【Ξパケット統計（境界平均）】")
        print(f"  ω_Λ平均: {np.mean(np.abs(Xi_packet['omega_Lambda'])):.6e}")
        print(f"  |∂_nΛ|平均: {np.mean(np.abs(Xi_packet['grad_n_Lambda'])):.6e}")
        print(f"  j_n平均: {np.mean(np.abs(Xi_packet['j_n'])):.6e}")
        print(f"  曲率平均: {np.mean(Xi_packet['curvature']):.6e}")
    
    return result

# =============================================================================
# Section 6: ユーティリティ関数
# =============================================================================

def estimate_boundary_length(boundary_points: List[Tuple[float, float]]) -> float:
    """境界Σの長さを推定"""
    if len(boundary_points) < 2:
        return 0.0
    
    length = 0.0
    for i in range(len(boundary_points) - 1):
        T1, P1 = boundary_points[i]
        T2, P2 = boundary_points[i+1]
        # ユークリッド距離（温度と圧力のスケール調整が必要）
        # 簡易版：Tを主軸として計算
        length += np.sqrt((T2-T1)**2 + ((P2-P1)/10)**2)  # P/10でスケール調整
    
    return length

# =============================================================================
# Section 7: 可視化（2D版）
# =============================================================================

def plot_noncommutative_boundary_pao(result: Dict, exp_data, save_path: str = None):
    """
    ★PAO用非可換境界の可視化★
    
    6つのサブプロット：
      (A) θ_eff分布（境界曲線上）
      (B) Ξパケット3成分（境界曲線上）
      (C) 非可換寄与Δ_NC
      (D) Λ場ヒートマップ + 境界Σ + 近傍サンプル
      (E) 境界上のΛ値と動的閾値
      (F) 曲率とθ_effの関係
    """
    from pao_holographic_fixed import compute_Lambda_field_pao_ultimate
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    boundary_points = result['boundary_points']
    theta_eff = result['theta_eff']
    Xi = result['Xi_packet']
    nc_sig = result['nc_signature']
    samples = result['samples']
    
    # パラメータ取得
    params = None  # 外から渡す必要がある
    
    # (A) θ_effの分布（境界曲線に沿って）
    ax = fig.add_subplot(gs[0, 0])
    boundary_indices = list(range(len(boundary_points)))
    ax.plot(boundary_indices, theta_eff, 'b-', linewidth=2, label='θ_eff')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(boundary_indices, 0, theta_eff, alpha=0.2)
    ax.set_xlabel('Boundary Point Index', fontsize=12)
    ax.set_ylabel('θ_eff (noncommutativity)', fontsize=12)
    ax.set_title(f'(A) Noncommutative Parameter\n<|θ_eff|> = {np.mean(np.abs(theta_eff)):.3e}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # (B) Ξの3成分（境界曲線に沿って）
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(boundary_indices, Xi['omega_Lambda'], 'r-', linewidth=2, 
            label='ω_Λ (vorticity)')
    ax.plot(boundary_indices, Xi['grad_n_Lambda'], 'g-', linewidth=2, 
            label='|∂_nΛ| (hardness)')
    ax.plot(boundary_indices, np.abs(Xi['j_n']), 'b-', linewidth=2, 
            label='|j_n| (flux)')
    ax.set_xlabel('Boundary Point Index', fontsize=12)
    ax.set_ylabel('Ξ components', fontsize=12)
    ax.set_title('(B) Boundary Information Packet', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # (C) 順序依存性の寄与
    ax = fig.add_subplot(gs[0, 2])
    contributions = nc_sig['contributions']
    if len(contributions) > 0:
        colors = ['red' if c > 0 else 'blue' for c in contributions]
        ax.bar(range(len(contributions)), contributions, color=colors, alpha=0.7)
        ax.axhline(0, color='k', linestyle='-', linewidth=1)
    ax.set_xlabel('Segment Index', fontsize=12)
    ax.set_ylabel('NC contribution [f,g]_θ', fontsize=12)
    ax.set_title(f"(C) Order Dependence\nΔ_NC = {nc_sig['Delta_NC']:.6e}",
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # (D) Λ場ヒートマップ + 境界Σ + 近傍サンプル
    ax = fig.add_subplot(gs[1, :])
    
    # ヒートマップ用のグリッド
    T_fine = np.linspace(100, 250, 50)
    P_fine = np.linspace(100, 600, 50)
    TT, PP = np.meshgrid(T_fine, P_fine)
    
    # Λ場計算（簡易版：時間がかかるので粗く）
    print("  Λ場を計算中...")
    Lambda_field = np.zeros_like(TT)
    for i in range(0, len(T_fine), 2):  # 2点おきに計算（高速化）
        for j in range(0, len(P_fine), 2):
            if params:
                Lambda_field[j, i] = compute_Lambda_field_pao_ultimate(
                    params, T_fine[i], P_fine[j], exp_data
                )
    
    # ヒートマップ
    im = ax.contourf(TT, PP, Lambda_field, levels=20, cmap='RdYlBu_r', alpha=0.6)
    ax.contour(TT, PP, Lambda_field, levels=[result['threshold']], 
              colors='black', linewidths=3)
    
    # 境界Σ
    T_boundary = Xi['O_T']
    P_boundary = Xi['O_P']
    ax.plot(T_boundary, P_boundary, 'r-', linewidth=3, label='Boundary Σ', zorder=5)
    ax.scatter(T_boundary, P_boundary, c='red', s=50, zorder=6, edgecolors='darkred')
    
    # 近傍サンプル
    ax.scatter(samples['T'], samples['P'], c='cyan', s=5, alpha=0.3, 
              label='Neighborhood Samples')
    
    # 実験データ点
    safe_T = [d.temperature for d in exp_data if not d.burnout]
    safe_P = [d.pressure for d in exp_data if not d.burnout]
    danger_T = [d.temperature for d in exp_data if d.burnout]
    danger_P = [d.pressure for d in exp_data if d.burnout]
    
    ax.scatter(safe_T, safe_P, c='blue', s=100, marker='o', 
              edgecolors='black', linewidth=2, label='Safe', zorder=7)
    ax.scatter(danger_T, danger_P, c='red', s=100, marker='x',
              linewidth=3, label='Burnout', zorder=7)
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Pressure (MPa)', fontsize=12)
    ax.set_title('(D) Λ Field + Boundary Σ + Neighborhood', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.colorbar(im, ax=ax, label='Λ')
    
    # (E) 境界上のΛ値と動的閾値
    ax = fig.add_subplot(gs[2, 0])
    Lambda_boundary = result['Lambda_boundary']
    threshold = result['threshold']
    
    ax.plot(boundary_indices, Lambda_boundary, 'b-', linewidth=2, label='Λ on Σ')
    ax.axhline(threshold, color='r', linestyle='--', linewidth=2, 
              label=f'Threshold = {threshold:.3f}')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Λ=1')
    ax.fill_between(boundary_indices, threshold, Lambda_boundary, 
                    where=Lambda_boundary>threshold, alpha=0.3, color='red')
    ax.fill_between(boundary_indices, threshold, Lambda_boundary, 
                    where=Lambda_boundary<=threshold, alpha=0.3, color='blue')
    ax.set_xlabel('Boundary Point Index', fontsize=12)
    ax.set_ylabel('Λ', fontsize=12)
    ax.set_title('(E) Λ on Boundary vs Threshold', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # (F) 曲率とθ_effの関係
    ax = fig.add_subplot(gs[2, 1])
    curvature = Xi['curvature']
    ax.scatter(curvature, np.abs(theta_eff), c='purple', s=50, alpha=0.6)
    ax.set_xlabel('Curvature κ', fontsize=12)
    ax.set_ylabel('|θ_eff|', fontsize=12)
    ax.set_title('(F) Curvature vs Noncommutativity', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # (G) パラメータ情報
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    info_text = f"""
PAO Noncommutative Boundary
AdS/CFT Correspondence

Boundary Σ:
  Points: {len(boundary_points)}
  Length: {estimate_boundary_length(boundary_points):.1f}

Noncommutative Parameters:
  <|θ_eff|> = {np.mean(np.abs(theta_eff)):.3e}
  max|θ_eff| = {np.max(np.abs(theta_eff)):.3e}

NC Signature:
  Δ_NC = {nc_sig['Delta_NC']:.3e}
  |Δ_NC| > 1e-6: {"YES ✓" if np.abs(nc_sig['Delta_NC']) > 1e-6 else "NO"}

Ξ Packet (avg):
  <ω_Λ> = {np.mean(np.abs(Xi['omega_Lambda'])):.3e}
  <|∂_nΛ|> = {np.mean(Xi['grad_n_Lambda']):.3e}
  <κ> = {np.mean(curvature):.3e}
"""
    
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('PAO Noncommutative Boundary Diagnostics (環作成)',
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n非可換境界図（PAO版）を保存: {save_path}")
    
    return fig

# =============================================================================
# Section 8: 統合実行関数
# =============================================================================

def run_pao_noncommutative_analysis(
    params_opt: Dict,
    exp_data,
    boundary_points: List[Tuple[float, float]],
    local_width: float = 5.0,
    n_local: int = 20,
    save_results: bool = True,
):
    """
    ★PAO非可換境界解析の統合実行★
    
    Args:
        params_opt: 最適化されたパラメータ
        exp_data: 実験データ
        boundary_points: 境界Σの点列
        local_width: 近傍幅
        n_local: サンプル数
        save_results: 結果保存のON/OFF
    
    Returns:
        result: 診断結果
    """
    print("\n" + "="*60)
    print("PAO非可換境界解析（AdS/CFT対応）")
    print("環ちゃんが作ったよ！💕")
    print("="*60)
    
    # 診断実行
    result = diagnose_noncommutative_boundary_pao(
        params_opt,
        boundary_points,
        exp_data,
        local_width=local_width,
        n_local=n_local,
        verbose=True
    )
    
    if not result:
        print("診断失敗")
        return None
    
    # 可視化
    print("\n可視化中...")
    result['params'] = params_opt  # プロットで使うため
    fig = plot_noncommutative_boundary_pao(
        result, 
        exp_data,
        save_path='pao_noncommutative_boundary.png'
    )
    plt.show()
    
    # 結果保存
    if save_results:
        # JSON保存（NumPy配列を変換）
        save_dict = {}
        for key, value in result.items():
            if key == 'params':
                continue  # paramsは別ファイル
            elif key == 'Xi_packet':
                # Ξパケットは個別処理
                Xi_save = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        Xi_save[k] = v.tolist()
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], tuple):
                        Xi_save[k] = [list(t) for t in v]
                    else:
                        Xi_save[k] = v
                save_dict[key] = Xi_save
            elif isinstance(value, np.ndarray):
                save_dict[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], tuple):
                save_dict[key] = [list(t) for t in value]
            elif isinstance(value, dict):
                # nc_signature等
                sub_dict = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        sub_dict[k] = v.tolist()
                    else:
                        sub_dict[k] = v
                save_dict[key] = sub_dict
            else:
                save_dict[key] = value
        
        with open('pao_nc_boundary_result.json', 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print("\n結果保存: pao_nc_boundary_result.json")
    
    return result

# =============================================================================
# メイン実行（テスト用）
# =============================================================================

if __name__ == "__main__":
    print("PAO Noncommutative Boundary Module")
    print("使用方法:")
    print("  from pao_noncommutative_boundary import run_pao_noncommutative_analysis")
    print("  result = run_pao_noncommutative_analysis(params_opt, exp_data, boundary_points)")
