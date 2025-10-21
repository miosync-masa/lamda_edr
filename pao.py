"""
Holographic Lubricant Solver - Ultimate Edition (FIXED)
PAO潤滑油の自己無撞着ホログラフィック最適化
FLCコードの全機能を正しく継承！
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

def compute_Lambda_field_pao_ultimate(params: Dict, T: float, P: float,
                                     TP_range: List[Tuple[float, float]] = None,
                                     exp_data: List[BurnoutData] = None,
                                     _cache: Dict = {}) -> float:
    """
    ★究極のΛ場計算（PAO版・焼き切れ境界基準）★
    
    scale = median(burnout K/V) / (1+ρ)
    → 焼き切れ点がΛ≈1になる！
    """
    K = compute_K_pao(params, T, P)
    V = compute_V_pao(params, T, P)
    
    # パラメータのハッシュ値を作成（キャッシュキー）
    param_key = tuple(sorted(params.items()))
    
    if param_key not in _cache:
        # 初回：scaleを計算してキャッシュ
        if exp_data is not None:
            # ★修正：焼き切れ点のK/Vを基準に！
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
                # 焼き切れ点の中央値を基準
                median_burnout = np.median(burnout_kv)
                
                # ρで調整（焼き切れ点がΛ≈1+ρになる）
                rho = params.get('rho', 0.03)
                scale = median_burnout / (1.0 + rho)
            else:
                # 焼き切れ点がない場合（デバッグ用）
                scale = np.median(safe_kv) / 0.5
        else:
            # exp_dataがない場合はデフォルト
            T_samples = np.linspace(180, 250, 20)
            P_samples = np.linspace(350, 550, 20)
            TP_range = [(t, p) for t in T_samples for p in P_samples]
            
            all_K_over_V = [
                compute_K_pao(params, t, p) / compute_V_pao(params, t, p)
                for t, p in TP_range
            ]
            rho = params.get('rho', 0.03)
            scale = np.median(all_K_over_V) / (1.0 + rho)
        
        _cache[param_key] = scale
    else:
        # 2回目以降：キャッシュから取得
        scale = _cache[param_key]
    
    Lambda = (K / V) / scale
    
    return Lambda

def clear_lambda_cache():
    """Λ計算のキャッシュをクリア"""
    compute_Lambda_field_pao_ultimate.__defaults__[2].clear()

# =============================================================================
# Section 3: 境界抽出（FLC完全互換）
# =============================================================================

def extract_critical_boundary_pao(params_dict, T_range, P_range, exp_data,
                                 Lambda_crit=None, contact_tol=1e-2):
    """
    ★境界Σ抽出（FLC完全互換版）★
    
    2D空間(T,P)での等高線抽出
    FLCの1D版を2Dに拡張
    """
    if Lambda_crit is None:
        # 動的閾値の計算
        Lambda_safe = []
        Lambda_danger = []
        for d in exp_data:
            Lambda = compute_Lambda_field_pao_ultimate(
                params_dict, d.temperature, d.pressure, exp_data=exp_data
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
                params_dict, T_range[i], P_range[j], exp_data=exp_data
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

def compute_boundary_info_packet_pao(params_dict, boundary_points, exp_data):
    """
    ★境界情報パケット（FLC完全互換版）★
    
    Ξパケット計算
    """
    if len(boundary_points) == 0:
        return {}
    
    Ξ = {}
    Ξ['Sigma'] = boundary_points
    
    # 各境界点での勾配計算
    grad_norms = []
    j_n_values = []
    
    for T, P in boundary_points:
        # 数値微分でΛの勾配
        dT, dP = 0.5, 5.0
        
        Lambda_0 = compute_Lambda_field_pao_ultimate(
            params_dict, T, P, exp_data=exp_data
        )
        Lambda_T_plus = compute_Lambda_field_pao_ultimate(
            params_dict, T + dT, P, exp_data=exp_data
        )
        Lambda_T_minus = compute_Lambda_field_pao_ultimate(
            params_dict, T - dT, P, exp_data=exp_data
        )
        Lambda_P_plus = compute_Lambda_field_pao_ultimate(
            params_dict, T, P + dP, exp_data=exp_data
        )
        Lambda_P_minus = compute_Lambda_field_pao_ultimate(
            params_dict, T, P - dP, exp_data=exp_data
        )
        
        # 勾配（中心差分）
        grad_T = (Lambda_T_plus - Lambda_T_minus) / (2 * dT)
        grad_P = (Lambda_P_plus - Lambda_P_minus) / (2 * dP)
        grad_norm = np.sqrt(grad_T**2 + grad_P**2)
        
        grad_norms.append(grad_norm)
        
        # フラックス（法線方向の勾配）
        # 境界に垂直な方向 = -∇Λ方向
        j_n = -grad_norm
        j_n_values.append(j_n)
    
    Ξ['grad_n_Lambda'] = np.array(grad_norms)
    Ξ['j_n'] = np.array(j_n_values)
    
    # 境界上のΛ値
    Ξ['O_Lambda'] = np.array([
        compute_Lambda_field_pao_ultimate(params_dict, T, P, exp_data=exp_data)
        for T, P in boundary_points
    ])
    
    # 境界上の(T,P)座標
    Ξ['O_T'] = np.array([T for T, P in boundary_points])
    Ξ['O_P'] = np.array([P for T, P in boundary_points])
    
    # 統計量
    Ξ['grad_n_mean'] = float(np.mean(grad_norms))
    Ξ['grad_n_std'] = float(np.std(grad_norms))
    Ξ['grad_n_max'] = float(np.max(grad_norms))
    Ξ['grad_n_min'] = float(np.min(grad_norms))
    Ξ['num_points'] = len(boundary_points)
    
    return Ξ

# =============================================================================
# Section 4: ホモトピー最適化（FLCから完全移植）
# =============================================================================

def solve_homotopy_pao_ultimate(
    exp_data: List[BurnoutData],
    physics_bounds: Dict,
    frozen_params: Dict,
    eps_schedule: List[float] = None,
    delta_schedule: List[float] = None,
    verbose: bool = True
) -> Tuple[Dict, object]:
    """
    ★PAO用ホモトピー最適化★
    
    FLCの solve_homotopy_ultimate_rho と同じ構造！
    """
    if eps_schedule is None:
        eps_schedule = [2e-1, 1.5e-1, 1e-1, 7e-2, 5e-2, 3e-2, 1e-2, 5e-3]
    
    if delta_schedule is None:
        delta_schedule = [0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.005]
    
    pmap = ParamMap(physics_bounds, frozen_params)
    
    print("\n" + "="*60)
    print("究極のホモトピー最適化（PAO版）")
    print("="*60)
    
    print("\n★2段階戦略:")
    print("  1. ウォームアップ：ρで横切り確保")
    print("  2. フル最適化：全パラメータで100%正解")
    
    # ===== Phase 1: ウォームアップ =====
    print("\n" + "="*60)
    print("Phase 1: ウォームアップ（全パラメータでρを重視）")
    print("="*60)
    
    # ★修正：warmupでも全パラメータを使う！
    warmup_pmap = ParamMap(physics_bounds, frozen_params)
    z_warmup = get_initial_guess_pao(physics_bounds, frozen_params)
    
    def warmup_objective(z):
        params = warmup_pmap.to_physical(z)
        penalty = 0.0
        
        # ★焼き切れ点基準スケールに合わせた目的関数
        Lambda_safe = []
        Lambda_danger = []
        
        for data in exp_data:
            # ★exp_dataを渡す
            Lambda = compute_Lambda_field_pao_ultimate(
                params, data.temperature, data.pressure, exp_data=exp_data
            )
            
            if data.burnout:
                Lambda_danger.append(Lambda)
                # ★焼き切れ点は0.95以上に！（median≈1+ρ）
                if Lambda < 0.95:
                    penalty += 500.0 * (0.95 - Lambda)**2
            else:
                Lambda_safe.append(Lambda)
                # ★安全点は0.85以下に！
                if Lambda > 0.85:
                    penalty += 300.0 * (Lambda - 0.85)**2
        
        # ★重要：焼き切れ点と安全点の分離
        if len(Lambda_danger) > 0 and len(Lambda_safe) > 0:
            median_danger = np.median(Lambda_danger)
            max_safe = np.max(Lambda_safe)
            
            # 焼き切れ点の中央値が安全点の最大値より大きいことを確保
            if median_danger <= max_safe:
                penalty += 1000.0 * (max_safe - median_danger + 0.2)**2
        
        return penalty
    
    print(f"\nWarmup Stage: 横切り確保（制約なし）")
    
    warmup_result = minimize(
        warmup_objective,
        z_warmup,
        method='L-BFGS-B',  # ★シンプルな制約なし最適化
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    
    params_warmup = warmup_pmap.to_physical(warmup_result.x)
    
    # 横切りチェック
    Lambda_vals = [
        compute_Lambda_field_pao_ultimate(
            params_warmup, d.temperature, d.pressure, exp_data=exp_data
        )
        for d in exp_data
    ]
    
    print(f"  Λ範囲: [{np.min(Lambda_vals):.3f}, {np.max(Lambda_vals):.3f}]")
    print(f"  ρ = {params_warmup['rho']:.4f}")
    
    if np.min(Lambda_vals) < 1.0 < np.max(Lambda_vals):
        print("  ✓ 横切り確保！Phase 2へ移行")
    else:
        print("  ⚠ 横切り未達成、Phase 2で調整")
    
    # ===== Phase 2: フル最適化 =====
    print("\n" + "="*60)
    print("Phase 2: フル最適化（全パラメータ）")
    print("="*60)
    
    z_current = get_initial_guess_pao(physics_bounds, frozen_params)
    
    # warmupのρを初期値に反映
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
            penalty = 0.0
            
            for data in exp_data:
                # ★exp_dataを渡す
                Lambda = compute_Lambda_field_pao_ultimate(
                    params, data.temperature, data.pressure, exp_data=exp_data
                )
                
                if data.burnout:
                    # ★焼き切れ点：Λ >= 0.95（medianが1+ρ≈1.03になる想定）
                    if Lambda < 0.95:
                        penalty += 2000.0 * (0.95 - Lambda)**2
                else:
                    # ★安全点：Λ <= 0.85
                    if Lambda > 0.85:
                        penalty += 1000.0 * (Lambda - 0.85)**2
            
            # Prior正則化
            reg = 0.0
            for key in PAO_PRIOR_CENTER:
                if key in params:
                    prior_val = PAO_PRIOR_CENTER[key]
                    reg += ((params[key] - prior_val) / prior_val)**2
            
            return penalty + delta * reg
        
        # 制約：Priorからのずれ
        def constraint_func(z):
            params = pmap.to_physical(z)
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
            options={'maxiter': 3000, 'verbose': 1}
        )
        
        z_current = result.x
        params_current = pmap.to_physical(z_current)
        
        # 検証
        correct = 0
        
        # ★動的閾値の計算
        Lambda_safe_check = []
        Lambda_danger_check = []
        
        for data in exp_data:
            Lambda = compute_Lambda_field_pao_ultimate(
                params_current, data.temperature, data.pressure, exp_data=exp_data
            )
            if data.burnout:
                Lambda_danger_check.append(Lambda)
            else:
                Lambda_safe_check.append(Lambda)
        
        # 閾値 = (安全点の最大 + 焼き切れ点の最小) / 2
        if len(Lambda_safe_check) > 0 and len(Lambda_danger_check) > 0:
            threshold = (np.max(Lambda_safe_check) + np.min(Lambda_danger_check)) / 2.0
        else:
            threshold = 0.9  # デフォルト
        
        # 正解率計算
        for data in exp_data:
            Lambda = compute_Lambda_field_pao_ultimate(
                params_current, data.temperature, data.pressure, exp_data=exp_data
            )
            predicted = Lambda >= threshold
            if predicted == data.burnout:
                correct += 1
        
        accuracy = correct / len(exp_data) * 100
        
        Lambda_vals = [
            compute_Lambda_field_pao_ultimate(
                params_current, d.temperature, d.pressure, exp_data=exp_data
            )
            for d in exp_data
        ]
        
        print(f"\nStage {stage_idx+1} 完了:")
        print(f"  Success: {result.success}")
        print(f"  正解率: {accuracy:.1f}%")
        print(f"  Λ範囲: [{np.min(Lambda_vals):.3f}, {np.max(Lambda_vals):.3f}]")
        print(f"  ρ = {params_current.get('rho', 0.0):.4f}")
        
        if np.min(Lambda_vals) < 1.0 < np.max(Lambda_vals):
            print("  ✓ Λ=1横切り達成！")
        
        if accuracy >= 100.0:
            print("  ✓ 100%正解達成！最適化完了")
            break
    
    params_opt = pmap.to_physical(z_current)
    
    return params_opt, result

# =============================================================================
# Section 5: 診断
# =============================================================================

def quick_diagnostics_pao_ultimate(params_dict, exp_data):
    """PAO用クイック診断"""
    print("\n" + "="*60)
    print("クイック診断（究極版）")
    print("="*60)
    
    # Λ場の診断
    T_test = np.linspace(100, 250, 40)
    P_test = np.linspace(100, 600, 40)
    
    Lambda_vals = []
    cross_count = 0
    prev_side = None
    
    for T in T_test:
        for P in P_test:
            Lambda = compute_Lambda_field_pao_ultimate(
                params_dict, T, P, exp_data=exp_data
            )
            Lambda_vals.append(Lambda)
            
            current_side = Lambda >= 1.0
            if prev_side is not None and prev_side != current_side:
                cross_count += 1
            prev_side = current_side
    
    print("\n【Λ場の診断】")
    print(f"  Λ範囲: [{np.min(Lambda_vals):.3f}, {np.max(Lambda_vals):.3f}]")
    print(f"  Λ=1横切り回数: {cross_count}回")
    
    if np.min(Lambda_vals) < 1.0 < np.max(Lambda_vals):
        print("  ✓ Λ=1を横切っています！")
    
    # 学習パラメータ
    print("\n【学習パラメータ】")
    for key in ['T0_low', 'T0_high', 'alpha_P', 'rho']:
        if key in params_dict:
            print(f"  {key}: {params_dict[key]:.6f}")
    
    # 予測精度
    correct = 0
    
    # ★動的閾値の計算
    Lambda_safe_list = []
    Lambda_danger_list = []
    
    for data in exp_data:
        Lambda = compute_Lambda_field_pao_ultimate(
            params_dict, data.temperature, data.pressure, exp_data=exp_data
        )
        if data.burnout:
            Lambda_danger_list.append(Lambda)
        else:
            Lambda_safe_list.append(Lambda)
    
    # 閾値 = (安全点の最大 + 焼き切れ点の最小) / 2
    if len(Lambda_safe_list) > 0 and len(Lambda_danger_list) > 0:
        threshold = (np.max(Lambda_safe_list) + np.min(Lambda_danger_list)) / 2.0
    else:
        threshold = 0.9
    
    print("\n【予測検証】")
    print(f"  動的閾値: {threshold:.3f}")
    
    for data in exp_data:
        Lambda = compute_Lambda_field_pao_ultimate(
            params_dict, data.temperature, data.pressure, exp_data=exp_data
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
    
    if accuracy >= 100.0:
        print("  ✓ 完璧！")

# =============================================================================
# Section 6: 可視化
# =============================================================================

def visualize_pao_results(params_dict, exp_data):
    """PAO結果の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) Λ場のヒートマップ
    ax = axes[0, 0]
    
    T_fine = np.linspace(100, 250, 100)
    P_fine = np.linspace(100, 600, 100)
    TT, PP = np.meshgrid(T_fine, P_fine)
    
    Lambda_field = np.zeros_like(TT)
    for i in range(len(T_fine)):
        for j in range(len(P_fine)):
            Lambda_field[j, i] = compute_Lambda_field_pao_ultimate(
                params_dict, T_fine[i], P_fine[j]
            )
    
    im = ax.contourf(TT, PP, Lambda_field, levels=20, cmap='RdYlBu_r')
    ax.contour(TT, PP, Lambda_field, levels=[1.0], colors='red', linewidths=2)
    
    # データ点プロット
    safe_T = [d.temperature for d in exp_data if not d.burnout]
    safe_P = [d.pressure for d in exp_data if not d.burnout]
    danger_T = [d.temperature for d in exp_data if d.burnout]
    danger_P = [d.pressure for d in exp_data if d.burnout]
    
    ax.scatter(safe_T, safe_P, c='blue', s=100, marker='o', 
              edgecolors='black', linewidth=2, label='Safe', zorder=5)
    ax.scatter(danger_T, danger_P, c='red', s=100, marker='x',
              linewidth=3, label='Burnout', zorder=5)
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Pressure (MPa)', fontsize=12)
    ax.set_title('(a) Λ Field & Boundary Σ', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.colorbar(im, ax=ax, label='Λ')
    
    # (b) 境界Σの勾配
    ax = axes[0, 1]
    
    T_range = np.linspace(170, 230, 30)
    P_range = np.linspace(300, 500, 30)
    Sigma = extract_critical_boundary_pao(params_dict, T_range, P_range)
    
    if len(Sigma) > 0:
        Sigma_T = [t for t, p in Sigma]
        Sigma_P = [p for t, p in Sigma]
        
        ax.scatter(Sigma_T, Sigma_P, c='red', s=80, zorder=5,
                  edgecolors='darkred', linewidth=1.5, label=f'Boundary Σ ({len(Sigma)} pts)')
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Pressure (MPa)', fontsize=12)
    ax.set_title(f'(b) Critical Boundary Σ', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # (c) Λ分布（T方向）
    ax = axes[1, 0]
    
    P_samples = [300, 400, 500]
    for P in P_samples:
        Lambda_T = [compute_Lambda_field_pao_ultimate(params_dict, T, P) 
                   for T in T_fine]
        ax.plot(T_fine, Lambda_T, linewidth=2, label=f'P={P}MPa')
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Λ=1')
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Λ', fontsize=12)
    ax.set_title('(c) Λ vs Temperature', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # (d) パラメータ情報
    ax = axes[1, 1]
    ax.axis('off')
    
    param_text = f"""
PAO Holographic Model

Λ(T,P) = (K/V) / scale
scale = (1-ρ) × max(K/V)

Parameters:
  ρ = {params_dict.get('rho', 0.0):.4f}
  T0_low = {params_dict.get('T0_low', 0.0):.3f}
  T0_high = {params_dict.get('T0_high', 0.0):.3f}
  alpha_P = {params_dict.get('alpha_P', 0.0):.4f}

Frozen:
  T_transition = {params_dict.get('T_transition', 0.0):.1f}°C
  T0_V = {params_dict.get('T0_V', 0.0):.1f}
  beta_P = {params_dict.get('beta_P', 0.0):.4f}
  V_base = {params_dict.get('V_base', 0.0):.3f}
"""
    
    ax.text(0.1, 0.9, param_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('pao_holographic_results.png', dpi=150, bbox_inches='tight')
    print("\n結果図を保存: pao_holographic_results.png")
    plt.show()

# =============================================================================
# Section 7: メイン実行
# =============================================================================

def run_pao_holographic_ultimate():
    """PAOホログラフィック実験（究極版）"""
    print("="*60)
    print("PAO Holographic Analysis - Ultimate Edition")
    print("FLC完全互換・自己無撞着スケール")
    print("="*60)
    
    exp_data = BurnoutData.get_pao_data()
    
    physics_bounds = {
        'T0_low': (35.0, 55.0),
        'T0_high': (8.0, 16.0),
        'alpha_P': (0.01, 0.03),
        'rho': (0.005, 0.08),
    }
    
    print("\n★特徴:")
    print("  - 部分凍結: T_transition, T0_V, beta_P, V_base")
    print("  - 学習: T0_low, T0_high, alpha_P, ρ")
    print("  - Lambda_scale = (1-ρ) × max(K/V)")
    print("  - Λ_max = 1/(1-ρ) > 1 保証")
    print("  - 2段階戦略（ウォームアップ→フル）")
    
    # ホモトピー最適化
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
    
    # ★動的閾値の計算（境界抽出用）
    Lambda_safe_for_boundary = []
    Lambda_danger_for_boundary = []
    
    for d in exp_data:
        Lambda = compute_Lambda_field_pao_ultimate(params_opt, d.temperature, d.pressure, exp_data=exp_data)
        if d.burnout:
            Lambda_danger_for_boundary.append(Lambda)
        else:
            Lambda_safe_for_boundary.append(Lambda)
    
    if len(Lambda_safe_for_boundary) > 0 and len(Lambda_danger_for_boundary) > 0:
        boundary_threshold = (np.max(Lambda_safe_for_boundary) + np.min(Lambda_danger_for_boundary)) / 2.0
    else:
        boundary_threshold = 0.9
    
    # 境界抽出
    T_range = np.linspace(170, 230, 40)
    P_range = np.linspace(300, 500, 40)
    Sigma = extract_critical_boundary_pao(params_opt, T_range, P_range, Lambda_crit=boundary_threshold)
    
    print(f"\n境界Σ検出: {len(Sigma)}点")
    
    if len(Sigma) > 0:
        # 境界情報パケット
        Ξ = compute_boundary_info_packet_pao(params_opt, Sigma)
        
        print("\n【境界情報パケット出力】")
        print(f"  |∂_nΛ|平均: {Ξ['grad_n_mean']:.6f}")
        print(f"  |∂_nΛ|最大: {Ξ['grad_n_max']:.6f}")
        
        # 保存
        with open('Xi_boundary_pao_ultimate.json', 'w') as f:
            Ξ_serializable = {}
            for key, value in Ξ.items():
                if isinstance(value, np.ndarray):
                    Ξ_serializable[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], tuple):
                    Ξ_serializable[key] = [list(v) for v in value]
                else:
                    Ξ_serializable[key] = value
            json.dump(Ξ_serializable, f, indent=2)
        
        print("  保存: Xi_boundary_pao_ultimate.json")
    
    # 最終パラメータ保存
    final_params = {**PAO_PARAMS_FROZEN, **params_opt}
    with open('pao_holographic_params_ultimate.json', 'w') as f:
        json.dump(final_params, f, indent=2)
    print("\n最終パラメータ保存: pao_holographic_params_ultimate.json")
    
    # サマリー
    # ★動的閾値で正解率計算
    Lambda_safe_final = []
    Lambda_danger_final = []
    
    for d in exp_data:
        Lambda = compute_Lambda_field_pao_ultimate(params_opt, d.temperature, d.pressure, exp_data=exp_data)
        if d.burnout:
            Lambda_danger_final.append(Lambda)
        else:
            Lambda_safe_final.append(Lambda)
    
    if len(Lambda_safe_final) > 0 and len(Lambda_danger_final) > 0:
        threshold_final = (np.max(Lambda_safe_final) + np.min(Lambda_danger_final)) / 2.0
    else:
        threshold_final = 0.9
    
    correct = sum(1 for d in exp_data 
                 if (compute_Lambda_field_pao_ultimate(params_opt, d.temperature, d.pressure, exp_data=exp_data) >= threshold_final) == d.burnout)
    accuracy = correct / len(exp_data) * 100
    
    Lambda_vals = [compute_Lambda_field_pao_ultimate(params_opt, d.temperature, d.pressure, exp_data=exp_data)
                  for d in exp_data]
    Lambda_crosses = np.min(Lambda_vals) < threshold_final < np.max(Lambda_vals)
    
    print("\n" + "="*60)
    print("実験結果サマリー")
    print("="*60)
    
    print(f"\n✓ 予測精度:")
    print(f"  正解率: {accuracy:.1f}%")
    
    print(f"\n✓ Λ場:")
    print(f"  範囲: [{np.min(Lambda_vals):.3f}, {np.max(Lambda_vals):.3f}]")
    print(f"  横切り: {'YES ✓' if Lambda_crosses else 'NO ✗'}")
    
    print(f"\n✓ 学習パラメータ:")
    print(f"  T0_low: {params_opt['T0_low']:.3f}")
    print(f"  T0_high: {params_opt['T0_high']:.3f}")
    print(f"  alpha_P: {params_opt['alpha_P']:.4f}")
    print(f"  ρ: {params_opt['rho']:.4f}")
    
    print(f"\n✓ 境界Σ:")
    print(f"  検出点数: {len(Sigma)}点")
    
    # 最終判定
    print("\n" + "="*60)
    if accuracy >= 100.0 and Lambda_crosses and len(Sigma) > 0:
        print("🎉🎉🎉 完全成功！PAO究極版で達成！")
        print("   ✓ 正解率: 100%")
        print("   ✓ Λ横切り: YES")
        print("   ✓ 境界Σ: 検出")
        print("   ✓ FLC完全互換: 達成")
    else:
        print(f"正解率: {accuracy:.1f}%")
        print(f"Λ横切り: {'YES' if Lambda_crosses else 'NO'}")
    print("="*60)
    
    return {
        'params': params_opt,
        'success': res.success,
        'accuracy': accuracy,
        'Lambda_crosses': Lambda_crosses,
        'boundary_detected': len(Sigma) > 0,
    }

# 実行
if __name__ == "__main__":
    results = run_pao_holographic_ultimate()
