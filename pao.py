"""
Holographic Lubricant Solver - COMPLETE VERSION by Tamaki
PAO潤滑油の自己無撞着ホログラフィック最適化 + 非可換境界診断
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.special import expit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import json
from dataclasses import dataclass

# =============================================================================
# Section 0: データ構造とPAO固定パラメータ
# =============================================================================

@dataclass
class BurnoutData:
    """焼き切れ実測データ"""
    temperature: float
    pressure: float
    shear_rate: float
    burnout: bool
    
    @classmethod
    def get_pao_data(cls) -> List['BurnoutData']:
        """PAO実測データ"""
        return [
            # 安全域
            cls(100, 100, 1e5, False),
            cls(120, 150, 1e5, False),
            cls(150, 200, 1e5, False),
            cls(170, 250, 1e5, False),
            cls(180, 300, 1e5, False),
            cls(190, 330, 1e5, False),
            cls(200, 350, 1e5, False),
            cls(210, 380, 1e5, False),
            # 危険域
            cls(220, 400, 1e5, True),
            cls(230, 420, 1e5, True),
            cls(250, 350, 1e5, True),
            cls(200, 500, 1e5, True),
            cls(180, 550, 1e5, True),
        ]

# 初期推定から得た固定パラメータ（部分凍結）
PAO_PARAMS_FROZEN = {
    'T_transition': 215.112836,
    'T0_V': 118.191329,
    'beta_P': 0.005209,
    'V_base': 2.022591,
}

PAO_PRIOR_CENTER = {
    'T0_low': 45.373135,
    'T0_high': 12.340213,
    'alpha_P': 0.019319,
}

# =============================================================================
# Section 1: パラメータ変換
# =============================================================================

def squash(x, lo, hi):
    """数値安定なシグモイド関数"""
    return lo + (hi - lo) * expit(x)

class ParamMap:
    """パラメータの再パラメータ化管理（部分凍結対応）"""
    def __init__(self, bounds_dict: Dict[str, Tuple[float, float]],
                 frozen_params: Dict[str, float] = None):
        self.frozen_params = frozen_params or {}
        self.keys = [k for k in bounds_dict.keys() if k not in self.frozen_params]
        self.bounds = np.array([bounds_dict[k] for k in self.keys], float)

    def to_physical(self, z: np.ndarray) -> Dict[str, float]:
        """無制約変数zを物理パラメータに変換"""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        learned_params = {k: squash(zi, lo[i], hi[i])
                         for i, (k, zi) in enumerate(zip(self.keys, z))}
        full_params = {**self.frozen_params, **learned_params}
        return full_params

    def size(self) -> int:
        return len(self.keys)

def get_initial_guess_pao(physics_bounds, frozen_params):
    """PAO用初期値設定"""
    pmap = ParamMap(physics_bounds, frozen_params)
    z0 = np.zeros(pmap.size())

    initial_params = {
        'T0_low': PAO_PRIOR_CENTER['T0_low'],
        'T0_high': PAO_PRIOR_CENTER['T0_high'],
        'alpha_P': PAO_PRIOR_CENTER['alpha_P'],
        'rho': 0.03,
    }

    for i, key in enumerate(pmap.keys):
        if key in initial_params:
            val = initial_params[key]
            lo, hi = pmap.bounds[i]
            normalized = (val - lo) / (hi - lo)
            normalized = np.clip(normalized, 0.01, 0.99)
            z0[i] = -np.log(1.0/normalized - 1.0)

    return z0

# =============================================================================
# Section 2: 物理モデル
# =============================================================================

def compute_K_pao(params: Dict, T: float, P: float) -> float:
    """PAO用K計算（温度・圧力効果）"""
    T_trans = params.get('T_transition', 215.0)
    
    x = (T - T_trans) / 10.0
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    
    T0_low = params.get('T0_low', 50.0)
    T0_high = params.get('T0_high', 10.0)
    
    K_low = np.exp(-(T - 100) / T0_low)
    K_high = K_low * np.exp((T - T_trans) / T0_high)
    
    K_T = (1 - sigmoid) * K_low + sigmoid * K_high
    
    alpha_P = params.get('alpha_P', 0.005)
    K_P = 1.0 + alpha_P * P
    
    return K_T * K_P

def compute_V_pao(params: Dict, T: float, P: float) -> float:
    """PAO用V計算（油膜厚さ）"""
    V_T = np.exp(-(T - 100) / params.get('T0_V', 80.0))
    V_P = np.exp(-params.get('beta_P', 0.004) * P)
    V_base = params.get('V_base', 1.0)
    
    return V_base * V_T * V_P

# ★修正：LambdaScaleManagerクラスを導入
class LambdaScaleManager:
    """Λスケール管理（キャッシュ問題を解決）"""
    def __init__(self):
        self._cache = {}
    
    def get_scale(self, params: Dict, exp_data: List[BurnoutData]) -> float:
        """
        ★焼き切れ点基準でscaleを計算★
        
        scale = median(burnout K/V) × (1 - ρ)
        → 焼き切れ点でΛ ≈ 1/(1-ρ) ≈ 1.03 (ρ=0.03の場合)
        """
        # パラメータハッシュ
        param_key = tuple(sorted(params.items()))
        
        if param_key in self._cache:
            return self._cache[param_key]
        
        # 焼き切れ点のK/Vを収集
        burnout_kv = []
        safe_kv = []
        
        for data in exp_data:
            K_d = compute_K_pao(params, data.temperature, data.pressure)
            V_d = compute_V_pao(params, data.temperature, data.pressure)
            kv = K_d / V_d
            
            if data.burnout:
                burnout_kv.append(kv)
            else:
                safe_kv.append(kv)
        
        if len(burnout_kv) > 0:
            # ★修正：焼き切れ点の中央値を基準
            # scale = median_burnout × (1-ρ)
            # → Λ = (K/V) / scale = 1/(1-ρ) at burnout
            median_burnout = np.median(burnout_kv)
            rho = params.get('rho', 0.03)
            scale = median_burnout * (1.0 - rho)
        else:
            # フォールバック
            if len(safe_kv) > 0:
                scale = np.median(safe_kv) / 0.5
            else:
                scale = 1.0
        
        self._cache[param_key] = scale
        return scale
    
    def clear(self):
        """キャッシュクリア"""
        self._cache.clear()

# グローバルなスケールマネージャー
_scale_manager = LambdaScaleManager()

def compute_Lambda_field_pao_ultimate(params: Dict, T: float, P: float,
                                     exp_data: List[BurnoutData]) -> float:
    """
    ★究極のΛ場計算（PAO版・修正版）★
    
    Λ = (K/V) / scale
    scale = median(burnout K/V) × (1-ρ)
    → 焼き切れ点でΛ ≈ 1/(1-ρ)
    """
    K = compute_K_pao(params, T, P)
    V = compute_V_pao(params, T, P)
    
    scale = _scale_manager.get_scale(params, exp_data)
    
    Lambda = (K / V) / scale
    
    return Lambda

def clear_lambda_cache():
    """Λ計算のキャッシュをクリア"""
    _scale_manager.clear()

# =============================================================================
# Section 3: 境界抽出（FLC完全互換）
# =============================================================================

def extract_critical_boundary_pao(params_dict, T_range, P_range, exp_data,
                                 Lambda_crit=None, contact_tol=1e-2):
    """
    ★境界Σ抽出（FLC完全互換版・修正）★
    
    2D空間(T,P)での等高線抽出
    """
    if Lambda_crit is None:
        # 動的閾値の計算
        Lambda_safe = []
        Lambda_danger = []
        for d in exp_data:
            Lambda = compute_Lambda_field_pao_ultimate(
                params_dict, d.temperature, d.pressure, exp_data
            )
            if d.burnout:
                Lambda_danger.append(Lambda)
            else:
                Lambda_safe.append(Lambda)
        
        if len(Lambda_safe) > 0 and len(Lambda_danger) > 0:
            Lambda_crit = (np.max(Lambda_safe) + np.min(Lambda_danger)) / 2.0
        else:
            Lambda_crit = 0.9
    
    # (T,P)グリッドでΛ計算
    TT, PP = np.meshgrid(T_range, P_range)
    Lambda_field = np.zeros_like(TT)
    
    for i in range(len(T_range)):
        for j in range(len(P_range)):
            Lambda_field[j, i] = compute_Lambda_field_pao_ultimate(
                params_dict, T_range[i], P_range[j], exp_data
            )
    
    # ★FLC流：符号変化検出
    deviation = Lambda_field - Lambda_crit
    
    boundary_points = []
    
    # T方向の横切り検出
    for j in range(len(P_range)):
        for i in range(len(T_range) - 1):
            d1, d2 = deviation[j, i], deviation[j, i+1]
            if d1 * d2 < 0:  # 符号変化
                # 線形補間
                T1, T2 = T_range[i], T_range[i+1]
                t = -d1 / (d2 - d1 + 1e-12)
                T_root = T1 + t * (T2 - T1)
                P_root = P_range[j]
                boundary_points.append((T_root, P_root))
    
    # P方向の横切り検出
    for i in range(len(T_range)):
        for j in range(len(P_range) - 1):
            d1, d2 = deviation[j, i], deviation[j+1, i]
            if d1 * d2 < 0:  # 符号変化
                # 線形補間
                P1, P2 = P_range[j], P_range[j+1]
                t = -d1 / (d2 - d1 + 1e-12)
                P_root = P1 + t * (P2 - P1)
                T_root = T_range[i]
                boundary_points.append((T_root, P_root))
    
    # ★接触点検出
    contact_mask = np.abs(deviation) < contact_tol
    contact_indices = np.where(contact_mask)
    
    for j, i in zip(contact_indices[0], contact_indices[1]):
        T_contact = T_range[i]
        P_contact = P_range[j]
        
        # 重複チェック
        if len(boundary_points) == 0 or \
           min(np.sqrt((T_contact - t)**2 + (P_contact - p)**2) 
               for t, p in boundary_points) > 5.0:
            boundary_points.append((T_contact, P_contact))
    
    return boundary_points

# =============================================================================
# Section 4: 非可換境界診断（環作成）
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
    """
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
        if len(boundary_points) >= n_samples_curvature:
            start_idx = max(0, idx - n_samples_curvature//2)
            end_idx = min(len(boundary_points), idx + n_samples_curvature//2 + 1)
            
            local_points = boundary_points[start_idx:end_idx]
            
            if len(local_points) >= 3:
                T_local = np.array([t for t, p in local_points])
                P_local = np.array([p for t, p in local_points])
                
                # 2次多項式フィット
                if len(set(T_local)) >= 3:
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

def compute_theta_eff_pao(Xi_packet: Dict, epsilon: float = 1e-6) -> np.ndarray:
    """
    ★PAO用非可換パラメータθ_effの計算★
    
    θ_eff = ω_Λ / (|∂_nΛ| × |j_n| + ε)
    """
    omega = Xi_packet['omega_Lambda']
    grad_n = Xi_packet['grad_n_Lambda']
    flux_n = Xi_packet['j_n']
    
    denominator = np.abs(grad_n) * (np.abs(flux_n) + epsilon)
    theta_eff = omega / (denominator + epsilon)
    
    return theta_eff

def compute_noncommutative_signature_pao(
    boundary_points: List[Tuple[float, float]],
    theta_eff: np.ndarray,
    field_f: np.ndarray,
    field_g: np.ndarray,
) -> Dict:
    """
    ★PAO用非可換性のシグネチャΔ_NC★
    
    Δ_NC = Σ[f_{i+1}g_i - f_ig_{i+1}]θ_eff(i)
    """
    n = len(boundary_points)
    
    delta_nc = 0.0
    contributions = []
    
    for i in range(n - 1):
        nc_term = (field_f[i+1] * field_g[i] -
                   field_f[i] * field_g[i+1]) * theta_eff[i]
        delta_nc += nc_term
        contributions.append(nc_term)
    
    contributions = np.array(contributions)
    
    result = {
        'Delta_NC': delta_nc,
        'contributions': contributions,
        'mean_abs': np.mean(np.abs(contributions)) if len(contributions) > 0 else 0.0,
        'std': np.std(contributions) if len(contributions) > 0 else 0.0,
        'max_abs': np.max(np.abs(contributions)) if len(contributions) > 0 else 0.0,
    }
    
    return result

def sample_boundary_neighborhood_pao(
    boundary_points: List[Tuple[float, float]],
    normal_vectors: List[Tuple[float, float]],
    local_width: float = 5.0,
    n_samples: int = 20,
) -> Dict:
    """★境界Σ近傍の法線方向サンプリング（2D版）★"""
    T_samples = []
    P_samples = []
    boundary_indices = []
    distances = []
    
    for idx, ((T, P), (nT, nP)) in enumerate(zip(boundary_points, normal_vectors)):
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

def diagnose_noncommutative_boundary_pao(
    params: Dict,
    boundary_points: List[Tuple[float, float]],
    exp_data,
    local_width: float = 5.0,
    n_local: int = 20,
    verbose: bool = True,
) -> Dict:
    """★PAO用非可換境界の完全診断★"""
    if len(boundary_points) == 0:
        if verbose:
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
    
    Lambda_boundary = Xi_packet['O_Lambda']
    error_field_boundary = Lambda_boundary - threshold
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
        
        print(f"\n【非可換パラメータ θ_eff】")
        print(f"  平均: {np.mean(np.abs(theta_eff)):.6e}")
        print(f"  最大: {np.max(np.abs(theta_eff)):.6e}")
        print(f"  最小: {np.min(np.abs(theta_eff)):.6e}")
        
        print(f"\n【非可換シグネチャ Δ_NC】")
        print(f"  Δ_NC = {nc_signature['Delta_NC']:.6e}")
        print(f"  平均寄与: {nc_signature['mean_abs']:.6e}")
        print(f"  最大寄与: {nc_signature['max_abs']:.6e}")
        
        if np.abs(nc_signature['Delta_NC']) > 1e-6:
            print(f"  ✓ 非可換性検出！（Δ_NC ≠ 0）")
        else:
            print(f"  - 可換極限近傍（Δ_NC ≈ 0）")
    
    return result

def estimate_boundary_length(boundary_points: List[Tuple[float, float]]) -> float:
    """境界Σの長さを推定"""
    if len(boundary_points) < 2:
        return 0.0
    
    length = 0.0
    for i in range(len(boundary_points) - 1):
        T1, P1 = boundary_points[i]
        T2, P2 = boundary_points[i+1]
        length += np.sqrt((T2-T1)**2 + ((P2-P1)/10)**2)
    
    return length

# =============================================================================
# Section 5: ホモトピー最適化（修正版）
# =============================================================================

def solve_homotopy_pao_ultimate(
    exp_data: List[BurnoutData],
    physics_bounds: Dict,
    frozen_params: Dict,
    eps_schedule: List[float] = None,
    delta_schedule: List[float] = None,
    verbose: bool = True
) -> Tuple[Dict, object]:
    """★PAO用ホモトピー最適化（修正版）★"""
    if eps_schedule is None:
        eps_schedule = [2e-1, 1.5e-1, 1e-1, 7e-2, 5e-2, 3e-2, 1e-2, 5e-3]
    
    if delta_schedule is None:
        delta_schedule = [0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.005]
    
    pmap = ParamMap(physics_bounds, frozen_params)
    
    print("\n" + "="*60)
    print("究極のホモトピー最適化（PAO版・環修正版）")
    print("="*60)
    
    # ===== Phase 1: ウォームアップ =====
    print("\n" + "="*60)
    print("Phase 1: ウォームアップ（全パラメータでρを重視）")
    print("="*60)
    
    warmup_pmap = ParamMap(physics_bounds, frozen_params)
    z_warmup = get_initial_guess_pao(physics_bounds, frozen_params)
    
    def warmup_objective(z):
        params = warmup_pmap.to_physical(z)
        params.update(frozen_params)
        penalty = 0.0
        
        clear_lambda_cache()
        
        Lambda_safe = []
        Lambda_danger = []
        
        for data in exp_data:
            Lambda = compute_Lambda_field_pao_ultimate(
                params, data.temperature, data.pressure, exp_data
            )
            
            if data.burnout:
                Lambda_danger.append(Lambda)
                if Lambda < 0.95:
                    penalty += 500.0 * (0.95 - Lambda)**2
            else:
                Lambda_safe.append(Lambda)
                if Lambda > 0.85:
                    penalty += 300.0 * (Lambda - 0.85)**2
        
        if len(Lambda_danger) > 0 and len(Lambda_safe) > 0:
            median_danger = np.median(Lambda_danger)
            max_safe = np.max(Lambda_safe)
            
            if median_danger <= max_safe:
                penalty += 1000.0 * (max_safe - median_danger + 0.2)**2
        
        return penalty
    
    warmup_result = minimize(
        warmup_objective,
        z_warmup,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    
    params_warmup = warmup_pmap.to_physical(warmup_result.x)
    params_warmup.update(frozen_params)
    
    clear_lambda_cache()
    
    # ===== Phase 2: フル最適化 =====
    print("\n" + "="*60)
    print("Phase 2: フル最適化（全パラメータ）")
    print("="*60)
    
    z_current = get_initial_guess_pao(physics_bounds, frozen_params)
    
    if 'rho' in pmap.keys:
        rho_idx = pmap.keys.index('rho')
        lo, hi = pmap.bounds[rho_idx]
        rho_val = params_warmup['rho']
        normalized = (rho_val - lo) / (hi - lo)
        normalized = np.clip(normalized, 0.01, 0.99)
        z_current[rho_idx] = -np.log(1.0/normalized - 1.0)
    
    result = None
    
    for stage_idx, (eps, delta) in enumerate(zip(eps_schedule, delta_schedule)):
        print("\n" + "="*60)
        print(f"Stage {stage_idx+1}/{len(eps_schedule)}: ±{eps*100:.1f}%制約, δ={delta}")
        print("="*60)
        
        def objective(z):
            params = pmap.to_physical(z)
            params.update(frozen_params)
            penalty = 0.0
            
            clear_lambda_cache()
            
            for data in exp_data:
                Lambda = compute_Lambda_field_pao_ultimate(
                    params, data.temperature, data.pressure, exp_data
                )
                
                if data.burnout:
                    if Lambda < 0.95:
                        penalty += 2000.0 * (0.95 - Lambda)**2
                else:
                    if Lambda > 0.85:
                        penalty += 1000.0 * (Lambda - 0.85)**2
            
            reg = 0.0
            for key in PAO_PRIOR_CENTER:
                if key in params:
                    prior_val = PAO_PRIOR_CENTER[key]
                    reg += ((params[key] - prior_val) / prior_val)**2
            
            return penalty + delta * reg
        
        def constraint_func(z):
            params = pmap.to_physical(z)
            params.update(frozen_params)
            cons = []
            for key in PAO_PRIOR_CENTER:
                if key in params:
                    prior_val = PAO_PRIOR_CENTER[key]
                    deviation = abs(params[key] - prior_val) / prior_val
                    cons.append(eps - deviation)
            return np.array(cons)
        
        constraint = NonlinearConstraint(constraint_func, 0, np.inf)
        
        result = minimize(
            objective,
            z_current,
            method='trust-constr',
            constraints=[constraint],
            options={'maxiter': 3000, 'verbose': 1 if verbose else 0}
        )
        
        z_current = result.x
        params_current = pmap.to_physical(z_current)
        params_current.update(frozen_params)
        
        clear_lambda_cache()
        
        # 検証
        correct = 0
        Lambda_safe_check = []
        Lambda_danger_check = []
        
        for data in exp_data:
            Lambda = compute_Lambda_field_pao_ultimate(
                params_current, data.temperature, data.pressure, exp_data
            )
            if data.burnout:
                Lambda_danger_check.append(Lambda)
            else:
                Lambda_safe_check.append(Lambda)
        
        if len(Lambda_safe_check) > 0 and len(Lambda_danger_check) > 0:
            threshold = (np.max(Lambda_safe_check) + np.min(Lambda_danger_check)) / 2.0
        else:
            threshold = 0.9
        
        for data in exp_data:
            Lambda = compute_Lambda_field_pao_ultimate(
                params_current, data.temperature, data.pressure, exp_data
            )
            predicted = Lambda >= threshold
            if predicted == data.burnout:
                correct += 1
        
        accuracy = correct / len(exp_data) * 100
        
        Lambda_vals = [
            compute_Lambda_field_pao_ultimate(
                params_current, d.temperature, d.pressure, exp_data
            )
            for d in exp_data
        ]
        
        print(f"\nStage {stage_idx+1} 完了:")
        print(f"  正解率: {accuracy:.1f}%")
        print(f"  Λ範囲: [{np.min(Lambda_vals):.3f}, {np.max(Lambda_vals):.3f}]")
        
        if accuracy >= 100.0:
            print("  ✓ 100%正解達成！最適化完了")
            break
    
    params_opt = pmap.to_physical(z_current)
    params_opt.update(frozen_params)
    
    return params_opt, result

# =============================================================================
# Section 6: 診断
# =============================================================================

def quick_diagnostics_pao_ultimate(params_dict, exp_data):
    """PAO用クイック診断（修正版）"""
    print("\n" + "="*60)
    print("クイック診断（究極版・環修正）")
    print("="*60)
    
    clear_lambda_cache()
    
    # 動的閾値の計算
    Lambda_safe_list = []
    Lambda_danger_list = []
    
    for data in exp_data:
        Lambda = compute_Lambda_field_pao_ultimate(
            params_dict, data.temperature, data.pressure, exp_data
        )
        if data.burnout:
            Lambda_danger_list.append(Lambda)
        else:
            Lambda_safe_list.append(Lambda)
    
    if len(Lambda_safe_list) > 0 and len(Lambda_danger_list) > 0:
        threshold = (np.max(Lambda_safe_list) + np.min(Lambda_danger_list)) / 2.0
    else:
        threshold = 0.9
    
    print("\n【予測検証】")
    print(f"  動的閾値: {threshold:.3f}")
    
    correct = 0
    for data in exp_data:
        Lambda = compute_Lambda_field_pao_ultimate(
            params_dict, data.temperature, data.pressure, exp_data
        )
        
        predicted = Lambda >= threshold
        actual = data.burnout
        
        if predicted == actual:
            correct += 1
            mark = "✓"
        else:
            mark = "✗"
        
        status = "焼き切れ" if actual else "安全"
        
        print(f"  T={data.temperature:3.0f}, P={data.pressure:3.0f}: "
              f"Λ={Lambda:.3f} ({status}) {mark}")
    
    accuracy = correct / len(exp_data) * 100
    
    print(f"\n【予測精度】")
    print(f"  正解率: {accuracy:.1f}%")

# =============================================================================
# Section 7: 可視化
# =============================================================================

def plot_noncommutative_boundary_pao(result: Dict, exp_data, params: Dict, 
                                    save_path: str = None):
    """★PAO用非可換境界の可視化★"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    boundary_points = result['boundary_points']
    theta_eff = result['theta_eff']
    Xi = result['Xi_packet']
    nc_sig = result['nc_signature']
    samples = result['samples']
    
    # (A) θ_effの分布
    ax = fig.add_subplot(gs[0, 0])
    boundary_indices = list(range(len(boundary_points)))
    ax.plot(boundary_indices, theta_eff, 'b-', linewidth=2, label='θ_eff')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(boundary_indices, 0, theta_eff, alpha=0.2)
    ax.set_xlabel('Boundary Point Index', fontsize=12)
    ax.set_ylabel('θ_eff (noncommutativity)', fontsize=12)
    ax.set_title(f'A Noncommutative Parameter\n<|θ_eff|> = {np.mean(np.abs(theta_eff)):.3e}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # (B) Ξの3成分
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(boundary_indices, Xi['omega_Lambda'], 'r-', linewidth=2, 
            label='ω_Λ (vorticity)')
    ax.plot(boundary_indices, Xi['grad_n_Lambda'], 'g-', linewidth=2, 
            label='|∂_nΛ| (hardness)')
    ax.plot(boundary_indices, np.abs(Xi['j_n']), 'b-', linewidth=2, 
            label='|j_n| (flux)')
    ax.set_xlabel('Boundary Point Index', fontsize=12)
    ax.set_ylabel('Ξ components', fontsize=12)
    ax.set_title('B Boundary Information Packet', fontsize=12, fontweight='bold')
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
    ax.set_title(f"C Order Dependence\nΔ_NC = {nc_sig['Delta_NC']:.6e}",
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # (D) Λ場ヒートマップ + 境界Σ
    ax = fig.add_subplot(gs[1, :])
    
    T_fine = np.linspace(100, 250, 50)
    P_fine = np.linspace(100, 600, 50)
    TT, PP = np.meshgrid(T_fine, P_fine)
    
    Lambda_field = np.zeros_like(TT)
    for i in range(len(T_fine)):
        for j in range(len(P_fine)):
            Lambda_field[j, i] = compute_Lambda_field_pao_ultimate(
                params, T_fine[i], P_fine[j], exp_data
            )
    
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
    ax.set_title('D Λ Field + Boundary Σ + Neighborhood', fontsize=13, fontweight='bold')
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
    ax.set_title('E Λ on Boundary vs Threshold', fontsize=12, fontweight='bold')
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
    
    plt.suptitle('PAO Noncommutative Boundary Diagnostics',
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n非可換境界図（PAO版）を保存: {save_path}")
    
    return fig

def visualize_pao_results(params_dict, exp_data):
    """PAO結果可視化（修正版）"""
    clear_lambda_cache()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # (a) Λ場のヒートマップ
    ax = axes[0, 0]
    
    T_fine = np.linspace(100, 250, 50)
    P_fine = np.linspace(100, 600, 50)
    TT, PP = np.meshgrid(T_fine, P_fine)
    
    Lambda_field = np.zeros_like(TT)
    for i in range(len(T_fine)):
        for j in range(len(P_fine)):
            Lambda_field[j, i] = compute_Lambda_field_pao_ultimate(
                params_dict, T_fine[i], P_fine[j], exp_data
            )
    
    im = ax.contourf(TT, PP, Lambda_field, levels=20, cmap='RdYlBu_r')
    ax.contour(TT, PP, Lambda_field, levels=[1.0], colors='black', linewidths=3)
    
    # データ点プロット
    safe_T = [d.temperature for d in exp_data if not d.burnout]
    safe_P = [d.pressure for d in exp_data if not d.burnout]
    danger_T = [d.temperature for d in exp_data if d.burnout]
    danger_P = [d.pressure for d in exp_data if d.burnout]
    
    ax.scatter(safe_T, safe_P, c='blue', s=100, marker='o', 
              edgecolors='black', linewidth=2, label='Safe', zorder=5)
    ax.scatter(danger_T, danger_P, c='red', s=100, marker='x',
              linewidth=3, label='Burnout', zorder=5)
    
    ax.set_xlabel('Temperature_°C', fontsize=12)
    ax.set_ylabel('Pressure_MPa', fontsize=12)
    ax.set_title('a Λ Field & Boundary Σ', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.colorbar(im, ax=ax, label='Λ')
    
    # (b) 境界Σ
    ax = axes[0, 1]
    
    T_range = np.linspace(170, 230, 30)
    P_range = np.linspace(300, 500, 30)
    
    Sigma = extract_critical_boundary_pao(params_dict, T_range, P_range, exp_data)
    
    if len(Sigma) > 0:
        Sigma_T = [t for t, p in Sigma]
        Sigma_P = [p for t, p in Sigma]
        
        ax.scatter(Sigma_T, Sigma_P, c='red', s=80, zorder=5,
                  edgecolors='darkred', linewidth=1.5, 
                  label=f'Boundary Σ {len(Sigma)} pts')
    
    ax.set_xlabel('Temperature °C', fontsize=12)
    ax.set_ylabel('Pressure MPa', fontsize=12)
    ax.set_title(f'b Critical Boundary Σ', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # (c) Λ分布（T方向）
    ax = axes[1, 0]
    
    P_samples = [300, 400, 500]
    for P in P_samples:
        Lambda_T = [compute_Lambda_field_pao_ultimate(params_dict, T, P, exp_data) 
                   for T in T_fine]
        ax.plot(T_fine, Lambda_T, linewidth=2, label=f'P={P}MPa')
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Λ=1')
    ax.set_xlabel('Temperature °C', fontsize=12)
    ax.set_ylabel('Λ', fontsize=12)
    ax.set_title('c Λ vs Temperature', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # (d) パラメータ情報
    ax = axes[1, 1]
    ax.axis('off')
    
    param_text = f"""
PAO Holographic Model

Λ(T,P) = (K/V) / scale
scale = median(burnout K/V) × (1-ρ)

Parameters:
  ρ = {params_dict.get('rho', 0.0):.4f}
  T0_low = {params_dict.get('T0_low', 0.0):.3f}
  T0_high = {params_dict.get('T0_high', 0.0):.3f}
  alpha_P = {params_dict.get('alpha_P', 0.0):.4f}
"""
    
    ax.text(0.1, 0.9, param_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('pao_holographic_complete.png', dpi=150, bbox_inches='tight')
    print("\n結果図を保存: pao_holographic_complete.png")
    plt.show()

# =============================================================================
# Section 8: メイン実行（統合版）
# =============================================================================

def run_pao_complete_analysis():
    """★PAO完全統合解析（環作成）★"""
    print("="*60)
    print("PAO Complete Analysis - by Tamaki")
    print("ホログラフィック最適化 + 非可換境界診断")
    print("="*60)
    
    exp_data = BurnoutData.get_pao_data()
    
    physics_bounds = {
        'T0_low': (35.0, 55.0),
        'T0_high': (8.0, 16.0),
        'alpha_P': (0.01, 0.03),
        'rho': (0.005, 0.08),
    }
    
    # ============= PART 1: ホログラフィック最適化 =============
    print("\n" + "="*60)
    print("PART 1: ホログラフィック最適化")
    print("="*60)
    
    params_opt, res = solve_homotopy_pao_ultimate(
        exp_data,
        physics_bounds,
        PAO_PARAMS_FROZEN,
        verbose=True
    )
    
    # 診断
    quick_diagnostics_pao_ultimate(params_opt, exp_data)
    
    # 可視化
    visualize_pao_results(params_opt, exp_data)
    
    # ============= PART 2: 非可換境界診断 =============
    print("\n" + "="*60)
    print("PART 2: 非可換境界診断")
    print("="*60)
    
    # 境界抽出
    T_range = np.linspace(170, 230, 40)
    P_range = np.linspace(300, 500, 40)
    
    Sigma = extract_critical_boundary_pao(
        params_opt, T_range, P_range, exp_data
    )
    
    print(f"\n境界Σ検出: {len(Sigma)}点")
    
    if len(Sigma) > 0:
        # 非可換境界診断
        nc_result = diagnose_noncommutative_boundary_pao(
            params_opt,
            Sigma,
            exp_data,
            local_width=5.0,
            n_local=20,
            verbose=True
        )
        
        # 非可換境界可視化
        print("\n非可換境界可視化中...")
        fig_nc = plot_noncommutative_boundary_pao(
            nc_result,
            exp_data,
            params_opt,
            save_path='pao_noncommutative_complete.png'
        )
        plt.show()
        
        # 結果保存
        save_dict = {}
        for key, value in nc_result.items():
            if key == 'Xi_packet':
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
                sub_dict = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        sub_dict[k] = v.tolist()
                    else:
                        sub_dict[k] = v
                save_dict[key] = sub_dict
            else:
                save_dict[key] = value
        
        with open('pao_complete_result.json', 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print("\n非可換境界結果保存: pao_complete_result.json")
    
    # 最終パラメータ保存
    final_params = {**PAO_PARAMS_FROZEN, **params_opt}
    with open('pao_complete_params.json', 'w') as f:
        json.dump(final_params, f, indent=2)
    print("\n最終パラメータ保存: pao_complete_params.json")
    
    # ============= 最終サマリー =============
    clear_lambda_cache()
    
    Lambda_safe_final = []
    Lambda_danger_final = []
    
    for d in exp_data:
        Lambda = compute_Lambda_field_pao_ultimate(
            params_opt, d.temperature, d.pressure, exp_data
        )
        if d.burnout:
            Lambda_danger_final.append(Lambda)
        else:
            Lambda_safe_final.append(Lambda)
    
    if len(Lambda_safe_final) > 0 and len(Lambda_danger_final) > 0:
        threshold_final = (np.max(Lambda_safe_final) + 
                          np.min(Lambda_danger_final)) / 2.0
    else:
        threshold_final = 0.9
    
    correct = sum(1 for d in exp_data 
                 if (compute_Lambda_field_pao_ultimate(
                     params_opt, d.temperature, d.pressure, exp_data
                 ) >= threshold_final) == d.burnout)
    accuracy = correct / len(exp_data) * 100
    
    print("\n" + "="*60)
    print("完全統合解析サマリー（環作成版）")
    print("="*60)
    
    print(f"\n✓ Part 1: ホログラフィック最適化")
    print(f"  正解率: {accuracy:.1f}%")
    print(f"  ρ: {params_opt['rho']:.4f}")
    
    if len(Sigma) > 0:
        print(f"\n✓ Part 2: 非可換境界診断")
        print(f"  境界Σ: {len(Sigma)}点検出")
        print(f"  Δ_NC: {nc_result['nc_signature']['Delta_NC']:.6e}")
        if np.abs(nc_result['nc_signature']['Delta_NC']) > 1e-6:
            print(f"  非可換性: 検出 ✓")
        else:
            print(f"  非可換性: 可換極限近傍")
    
    print("\n" + "="*60)
    if accuracy >= 100.0 and len(Sigma) > 0:
        print("🎉🎉🎉 完全成功！")
        print("   ✓ 最適化: 100%正解")
        print("   ✓ 境界診断: 完了")
        print("   ✓ 統合版: 成功")
    else:
        print(f"正解率: {accuracy:.1f}%")
        if accuracy >= 90.0:
            print("→ 良好な結果！")
    print("="*60)
    
    return {
        'params': params_opt,
        'accuracy': accuracy,
        'boundary_points': Sigma,
        'nc_result': nc_result if len(Sigma) > 0 else None,
    }

# 実行
if __name__ == "__main__":
    print("\n" + "="*60)
    print("環ちゃんが作った完全統合版だよ！💕")
    print("="*60)
    results = run_pao_complete_analysis()
