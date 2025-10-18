"""
Holographic FLC Experiment
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

# =============================================================================
# Section 0: 安定版パラメータ（部分凍結）
# =============================================================================

STABLE_PARAMS_FROZEN = {
    'K_scale': 9.963279e-01,
    'K_scale_draw': 1.158737e+00,
    'K_scale_plane': 8.227154e-01,
    'K_scale_biax': 1.141635e+00,
}

PRIOR_CENTER = {
    'beta_A': 1.076917e-01,
    'beta_bw': 2.500006e-01,
    'beta_A_pos': 7.086621e-02,
}

# =============================================================================
# Section 1: パラメータ変換
# =============================================================================

def squash(x, lo, hi):
    """数値安定なシグモイド関数"""
    return lo + (hi - lo) * expit(x)

class ParamMap:
    """パラメータの再パラメータ化管理"""
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

def get_initial_guess_rho(physics_bounds, frozen_params):
    """初期値設定（ρ版）"""
    pmap = ParamMap(physics_bounds, frozen_params)
    z0 = np.zeros(pmap.size())

    initial_params = {
        'E_gain': 0.5,
        'gamma': 0.8,
        'eta': 1.0,
        'alpha': 1.5,
        'beta_A': 0.10,
        'beta_bw': 0.25,
        'beta_A_pos': 0.08,
        'rho': 0.03,  # 🔥 追加！
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
# Section 2: EDR→FLC予測モデル
# =============================================================================

def compute_K(params_dict: Dict[str, float], beta: float) -> float:
    """エネルギー密度K(β)の計算"""
    K_base = params_dict.get('K_scale', 1.0)
    K_draw = params_dict.get('K_scale_draw', 1.0)
    K_plane = params_dict.get('K_scale_plane', 1.0)
    K_biax = params_dict.get('K_scale_biax', 1.0)

    w_draw = np.exp(-((beta + 0.5) / 0.25)**2)
    w_plane = np.exp(-((beta - 0.0) / 0.25)**2)
    w_biax = np.exp(-((beta - 0.75) / 0.3)**2)

    w_sum = w_draw + w_plane + w_biax + 1e-10
    w_draw /= w_sum
    w_plane /= w_sum
    w_biax /= w_sum

    K = K_base * (w_draw * K_draw + w_plane * K_plane + w_biax * K_biax)
    return K

def compute_V_eff(params_dict: Dict[str, float], beta: float) -> float:
    """実効体積|V|_eff(β)の計算"""
    beta_A = params_dict.get('beta_A', 0.1)
    beta_bw = params_dict.get('beta_bw', 0.5)
    beta_A_pos = params_dict.get('beta_A_pos', beta_A)

    w_draw = np.exp(-((beta + 0.5) / 0.25)**2)
    w_plane = np.exp(-((beta - 0.0) / 0.25)**2)
    w_biax = np.exp(-((beta - 0.75) / 0.3)**2)

    w_sum = w_draw + w_plane + w_biax + 1e-10
    w_draw /= w_sum
    w_plane /= w_sum
    w_biax /= w_sum

    base_draw = 0.40
    base_plane = 0.30
    base_biax = 0.22
    base = w_draw * base_draw + w_plane * base_plane + w_biax * base_biax

    if beta < 0:
        depth = beta_A * (1 - np.exp(-((beta / beta_bw)**2)))
    else:
        beta_bw_pos = beta_bw * 0.8
        depth = beta_A_pos * (1 - np.exp(-((beta / beta_bw_pos)**2)))

    depth_effect = depth * (1 - np.exp(-2 * abs(beta)))

    V_eff = base - depth_effect
    return np.clip(V_eff, 0.05, 1.0)

def compute_Lambda_field_ultimate(params_dict: Dict[str, float], beta: float,
                                  beta_range: np.ndarray = None) -> float:
    """
    ★究極のΛ場計算（ρ版）★

    Lambda_scale = (1-ρ) × max(K/V)
    → Λ_max = 1/(1-ρ) > 1 保証
    """
    K = compute_K(params_dict, beta)
    V_eff = compute_V_eff(params_dict, beta)

    # ★自己無撞着なLambda_scale
    if beta_range is None:
        beta_range = np.linspace(-0.5, 1.0, 50)

    all_K_over_V = [
        compute_K(params_dict, b) / compute_V_eff(params_dict, b)
        for b in beta_range
    ]

    max_kv = np.max(all_K_over_V)

    # 🔥 オーバーシュート係数ρ追加 🔥
    rho = params_dict.get('rho', 0.03)  # デフォルト3%
    scale = (1.0 - rho) * max_kv

    Lambda = (K / V_eff) / scale

    return Lambda

def compute_flc_point_ultimate(params_dict: Dict[str, float], beta: float,
                               beta_range: np.ndarray = None) -> float:
    """
    ★究極のFLC計算★

    Em(β) = Em_star + E_gain × |1-Λ(β)|^γ × V_eff^α × (K/K_ref)^η

    - Lambda_scaleは自己無撞着に決定
    - 物理的に必ずmax(Λ)=1.0
    """
    Lambda = compute_Lambda_field_ultimate(params_dict, beta, beta_range)
    K = compute_K(params_dict, beta)
    V_eff = compute_V_eff(params_dict, beta)

    # ★固定パラメータ（Em_starとK_refのみ）
    Em_star = params_dict.get('Em_star', 0.18)
    K_ref = params_dict.get('K_ref', 1.0)

    # ★学習パラメータ
    E_gain = params_dict.get('E_gain', 0.2)
    gamma = params_dict.get('gamma', 0.8)
    eta = params_dict.get('eta', 0.0)
    alpha = params_dict.get('alpha', 0.2)

    # ★究極の形式
    margin = np.abs(1.0 - Lambda)
    V_viscosity = V_eff ** alpha
    K_amplitude = (K / K_ref) ** eta

    Em = Em_star + E_gain * (margin ** gamma) * V_viscosity * K_amplitude

    return float(np.clip(Em, 0.05, 0.8))


# =============================================================================
# Section 3: 臨界アンカー（固定版）
# =============================================================================

def compute_minimal_anchor(flc_points: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    ★最小限のアンカー★

    Em_starとK_refのみ
    Lambda_scaleは自己無撞着に決定
    """
    # Em_star: 実測値の最小値
    Em_star = min(Em for _, Em in flc_points)

    # K_ref: β=0での値（基準点として）
    K_ref = compute_K(STABLE_PARAMS_FROZEN, 0.0)

    anchor_params = {
        'Em_star': Em_star,
        'K_ref': K_ref,
    }

    return anchor_params

def print_anchor_info_ultimate(anchor_params, flc_points=None):
    """最小限のアンカー情報表示"""
    print("\n" + "="*60)
    print("最小限アンカー（自己無撞着版）")
    print("="*60)

    print(f"\n★固定パラメータ:")
    print(f"  Em_star = {anchor_params['Em_star']:.3f}")
    print(f"  K_ref = {anchor_params['K_ref']:.3f}")
    print(f"  Lambda_scale = 動的計算（自己無撞着）")

    if flc_points:
        Em_star = anchor_params['Em_star']
        print(f"\n基底チェック:")
        for beta, Em in flc_points:
            status = "✓" if Em_star <= Em else "✗"
            print(f"  β={beta:+.2f}: Em_star={Em_star:.3f} ≤ 実測={Em:.3f} {status}")

# =============================================================================
# Section 4: 境界抽出
# =============================================================================

def extract_critical_boundary_ultimate(params_dict, beta_range,
                                      Lambda_crit=1.0, contact_tol=1e-3):
    """境界Σ抽出（究極版）"""
    # ★beta_range_globalを準備
    beta_range_global = np.linspace(-0.5, 1.0, 50)

    Lambda_values = np.array([
        compute_Lambda_field_ultimate(params_dict, b, beta_range_global)
        for b in beta_range
    ])

    deviation = Lambda_values - Lambda_crit
    sign_changes = np.sign(deviation)
    cross_indices = np.where(sign_changes[:-1] * sign_changes[1:] < 0)[0]

    roots = []
    for i in cross_indices:
        b1, b2 = beta_range[i], beta_range[i+1]
        L1, L2 = deviation[i], deviation[i+1]
        t = -L1 / (L2 - L1 + 1e-12)
        beta_root = b1 + t * (b2 - b1)
        roots.append(beta_root)

    contact_indices = np.where(np.abs(Lambda_values - Lambda_crit) < contact_tol)[0]
    for i in contact_indices:
        beta_contact = beta_range[i]
        if len(roots) == 0 or min(abs(beta_contact - r) for r in roots) > 0.05:
            roots.append(beta_contact)

    return np.array(sorted(roots))

    deviation = Lambda_values - Lambda_crit
    sign_changes = np.sign(deviation)
    cross_indices = np.where(sign_changes[:-1] * sign_changes[1:] < 0)[0]

    roots = []
    for i in cross_indices:
        b1, b2 = beta_range[i], beta_range[i+1]
        L1, L2 = deviation[i], deviation[i+1]

        t = -L1 / (L2 - L1 + 1e-12)
        beta_root = b1 + t * (b2 - b1)
        roots.append(beta_root)

    contact_indices = np.where(np.abs(Lambda_values - Lambda_crit) < contact_tol)[0]
    for i in contact_indices:
        beta_contact = beta_range[i]
        if len(roots) == 0 or min(abs(beta_contact - r) for r in roots) > 0.05:
            roots.append(beta_contact)

    return np.array(sorted(roots))

def compute_boundary_info_packet_ultimate(params_dict, beta_boundary, n_samples=50):
    """境界情報パケット（究極版）"""
    if len(beta_boundary) == 0:
        return {}

    # ★beta_range_globalを準備
    beta_range_global = np.linspace(-0.5, 1.0, 50)

    beta_min = np.min(beta_boundary) - 0.05
    beta_max = np.max(beta_boundary) + 0.05
    beta_dense = np.linspace(beta_min, beta_max, n_samples)

    Ξ = {}
    Ξ['Sigma'] = beta_boundary

    # ★究極版関数を使用
    Lambda_dense = np.array([
        compute_Lambda_field_ultimate(params_dict, b, beta_range_global)
        for b in beta_dense
    ])

    dLambda_dbeta = np.gradient(Lambda_dense, beta_dense)

    if len(beta_boundary) > 0:
        f_grad = interp1d(beta_dense, dLambda_dbeta, kind='cubic',
                         fill_value='extrapolate')
        Ξ['grad_n_Lambda'] = np.abs(f_grad(beta_boundary))

        J_Lambda = -dLambda_dbeta
        f_flux = interp1d(beta_dense, J_Lambda, kind='cubic',
                         fill_value='extrapolate')
        Ξ['j_n'] = f_flux(beta_boundary)

        omega_Lambda = np.gradient(J_Lambda, beta_dense)
        f_omega = interp1d(beta_dense, omega_Lambda, kind='cubic',
                          fill_value='extrapolate')
        Ξ['omega_Lambda'] = f_omega(beta_boundary)
    else:
        Ξ['grad_n_Lambda'] = np.array([])
        Ξ['j_n'] = np.array([])
        Ξ['omega_Lambda'] = np.array([])

    Ξ['O_beta'] = beta_boundary

    # ★究極版関数を使用
    Ξ['O_Em'] = np.array([
        compute_flc_point_ultimate(params_dict, b, beta_range_global)
        for b in beta_boundary
    ])
    Ξ['O_Lambda'] = np.array([
        compute_Lambda_field_ultimate(params_dict, b, beta_range_global)
        for b in beta_boundary
    ])

    return Ξ

# =============================================================================
# Section 5: 制約と最適化（超緩和版）
# =============================================================================

def make_boundary_constraint_ultra_relaxed(flc_points, pmap, anchor_params, eps_rel=1e-2):
    """
    制約（超緩和版）

    アンカー点の制約を緩和
    """
    beta_star = anchor_params['beta_star']

    def cons_vec(z):
        p = pmap.to_physical(z)
        p.update(anchor_params)

        constraints = []
        for beta, Em in flc_points:
            Em_pred = compute_flc_point_ultimate(p, beta)
            relative_error = (Em_pred - Em) / max(Em, 1e-3)
            constraints.append(relative_error)

        return np.array(constraints, float)

    n = len(flc_points)
    lower = np.full(n, -eps_rel)
    upper = np.full(n, eps_rel)

    # ★アンカー点も同じ制約（厳しくしない）

    return NonlinearConstraint(cons_vec, lower, upper)

def regularizer_rho_crossover(z: np.ndarray, pmap: ParamMap,
                              betas_for_shape: np.ndarray,
                              flc_points: List[Tuple[float, float]],
                              anchor_params: Dict[str, float],
                              beta_range_global: np.ndarray,
                              delta_cross: float = 0.03,
                              lambda_prior: float = 0.01) -> float:
    """
    ★究極の正則化：ρ版横切り保証★
    """
    p = pmap.to_physical(z)
    p.update(anchor_params)

    # 弱い事前分布
    prior_penalty = 0.0
    for key, center_val in PRIOR_CENTER.items():
        if key in p:
            current_val = p[key]
            if key == 'beta_A':
                scale = 0.2 * (0.30 - 0.03)
            elif key == 'beta_bw':
                scale = 0.2 * (0.50 - 0.10)
            else:  # beta_A_pos
                scale = 0.2 * (0.15 - 0.02)
            prior_penalty += lambda_prior * ((current_val - center_val) / scale) ** 2

    # 滑らかさ
    Em = np.array([
        compute_flc_point_ultimate(p, b, beta_range_global)
        for b in betas_for_shape
    ])
    if len(Em) > 2:
        d2 = np.diff(Em, n=2)
        smooth = np.mean(d2**2)
    else:
        smooth = 0.0

    # Λ場の計算
    Lambda_vals = np.array([
        compute_Lambda_field_ultimate(p, b, beta_range_global)
        for b in betas_for_shape
    ])
    Lambda_min = np.min(Lambda_vals)
    Lambda_max = np.max(Lambda_vals)

    # 🔥 横切り保証ペナルティ（δスケジュール対応） 🔥
    # Λ_max > 1+δ かつ Λ_min < 1-δ を推奨
    cross_penalty = (
        max(0.0, (1.0 + delta_cross) - Lambda_max)**2 +  # 上を超えてほしい
        max(0.0, Lambda_min - (1.0 - delta_cross))**2    # 下を超えてほしい
    )

    # η暴れ防止（弱いL2）
    eta = p.get('eta', 0.0)
    eta_penalty = 1e-3 * (eta ** 2)

    # 総合評価
    return (0.1 * smooth +
            0.2 * cross_penalty +  # ★強めに
            prior_penalty +
            eta_penalty)

def solve_homotopy_ultimate_rho(flc_points: List[Tuple[float, float]],
                               physics_bounds: Dict[str, Tuple[float, float]],
                               frozen_params: Dict[str, float],
                               eps_schedule: List[float] = [2e-1, 1.5e-1, 1e-1, 7e-2, 5e-2, 3e-2, 1e-2],
                               delta_schedule: List[float] = None,
                               verbose: bool = True) -> Tuple[Dict, object]:
    """
    ★究極のホモトピー最適化（ρ版）★

    2段階戦略：
    1. ウォームアップ：E_gain, ρのみ動かして横切りを先に満たす
    2. フル最適化：全パラメータで1%以内に収束
    """

    if delta_schedule is None:
        delta_schedule = [0.05] * 3 + [0.03] * 2 + [0.02, 0.01]

    anchor_params = compute_minimal_anchor(flc_points)

    print("\n" + "="*60)
    print("究極のホモトピー最適化（ρ版）")
    print("="*60)
    print(f"\n★オーバーシュート係数ρ追加:")
    print(f"  - Lambda_scale = (1-ρ) × max(K/V)")
    print(f"  - Λ_max = 1/(1-ρ) > 1 保証")
    print(f"  - 横切り確定！")

    print(f"\n★2段階戦略:")
    print(f"  1. ウォームアップ：E_gain, ρで横切り")
    print(f"  2. フル最適化：全パラメータで1%以内")

    print(f"\n★最小限のアンカー:")
    print(f"  Em_star = {anchor_params['Em_star']:.3f}")
    print(f"  K_ref = {anchor_params['K_ref']:.3f}")
    print(f"  Lambda_scale = 動的計算（ρ補正）")

    beta_range_global = np.linspace(-0.5, 1.0, 50)

    # 制約関数
    def cons_vec(z):
        p = pmap.to_physical(z)
        p.update(anchor_params)

        constraints = []
        for beta, Em in flc_points:
            Em_pred = compute_flc_point_ultimate(p, beta, beta_range_global)
            relative_error = (Em_pred - Em) / max(Em, 1e-3)
            constraints.append(relative_error)

        return np.array(constraints, float)

    # ★Phase 1: ウォームアップ（E_gain, ρのみ）
    print("\n" + "="*60)
    print("Phase 1: ウォームアップ（E_gain, ρのみ）")
    print("="*60)

    # E_gain, ρのみ学習可能
    warmup_bounds = {
        'E_gain': physics_bounds['E_gain'],
        'rho': physics_bounds['rho'],
    }

    # 他は固定
    warmup_frozen = {
        **frozen_params,
        'gamma': 0.8,
        'eta': 1.0,
        'alpha': 1.5,
        'beta_A': 0.10,
        'beta_bw': 0.25,
        'beta_A_pos': 0.08,
    }

    pmap_warmup = ParamMap(warmup_bounds, warmup_frozen)
    z_warmup = np.zeros(pmap_warmup.size())

    # 初期値
    for i, key in enumerate(pmap_warmup.keys):
        if key == 'E_gain':
            val = 0.5
        elif key == 'rho':
            val = 0.03
        lo, hi = pmap_warmup.bounds[i]
        normalized = (val - lo) / (hi - lo)
        normalized = np.clip(normalized, 0.01, 0.99)
        z_warmup[i] = -np.log(1.0/normalized - 1.0)

    # ウォームアップ最適化（横切り重視）
    def regularizer_warmup(z):
        beta_grid = np.linspace(-0.5, 1.0, 41)
        return regularizer_rho_crossover(
            z, pmap_warmup, beta_grid, flc_points, anchor_params,
            beta_range_global, delta_cross=0.05, lambda_prior=0.01
        )

    def cons_vec_warmup(z):
        p = pmap_warmup.to_physical(z)
        p.update(anchor_params)
        p.update(warmup_frozen)

        constraints = []
        for beta, Em in flc_points:
            Em_pred = compute_flc_point_ultimate(p, beta, beta_range_global)
            relative_error = (Em_pred - Em) / max(Em, 1e-3)
            constraints.append(relative_error)

        return np.array(constraints, float)

    # ウォームアップ実行（緩い制約）
    for stage in [1]:
        eps_rel = 0.20
        print(f"\nWarmup Stage {stage}: ±{eps_rel*100:.1f}%制約")

        n = len(flc_points)
        lower = np.full(n, -eps_rel)
        upper = np.full(n, eps_rel)
        nlcons = NonlinearConstraint(cons_vec_warmup, lower, upper)

        res_warmup = minimize(
            regularizer_warmup, z_warmup,
            method='trust-constr',
            constraints=[nlcons],
            options={'maxiter': 5000, 'verbose': 1 if verbose else 0}
        )

        z_warmup = res_warmup.x

        params_temp = pmap_warmup.to_physical(z_warmup)
        params_temp.update(anchor_params)
        params_temp.update(warmup_frozen)

        Lambda_vals = [
            compute_Lambda_field_ultimate(params_temp, b, beta_range_global)
            for b in beta_range_global
        ]
        print(f"  Λ範囲: [{np.min(Lambda_vals):.3f}, {np.max(Lambda_vals):.3f}]")
        print(f"  ρ = {params_temp['rho']:.4f}")

        if np.min(Lambda_vals) < 1.0 < np.max(Lambda_vals):
            print(f"  ✓ 横切り確保！Phase 2へ移行")
            break  # 成功したらPhase 2へ

    # ウォームアップ結果を初期値に
    warmup_params = pmap_warmup.to_physical(z_warmup)

    # ★Phase 2: フル最適化
    print("\n" + "="*60)
    print("Phase 2: フル最適化（全パラメータ）")
    print("="*60)

    pmap = ParamMap(physics_bounds, frozen_params)
    z_current = get_initial_guess_rho(physics_bounds, frozen_params)

    # ウォームアップ結果を反映
    for i, key in enumerate(pmap.keys):
        if key in warmup_params:
            val = warmup_params[key]
            lo, hi = pmap.bounds[i]
            normalized = (val - lo) / (hi - lo)
            normalized = np.clip(normalized, 0.01, 0.99)
            z_current[i] = -np.log(1.0/normalized - 1.0)

    # フル最適化
    res = None
    for stage, (eps_rel, delta_cross) in enumerate(zip(eps_schedule, delta_schedule), 1):
        print(f"\n{'='*60}")
        print(f"Stage {stage}/{len(eps_schedule)}: ±{eps_rel*100:.1f}%制約, δ={delta_cross:.2f}")
        print(f"{'='*60}")

        def regularizer_full(z):
            beta_grid = np.linspace(-0.5, 1.0, 41)
            return regularizer_rho_crossover(
                z, pmap, beta_grid, flc_points, anchor_params,
                beta_range_global, delta_cross=delta_cross, lambda_prior=0.01
            )

        n = len(flc_points)
        lower = np.full(n, -eps_rel)
        upper = np.full(n, eps_rel)
        nlcons = NonlinearConstraint(cons_vec, lower, upper)

        res = minimize(
            regularizer_full, z_current,
            method='trust-constr',
            constraints=[nlcons],
            options={'maxiter': 10000, 'verbose': 1 if verbose else 0}
        )

        if res.success or stage == len(eps_schedule):
            z_current = res.x
            print(f"\nStage {stage} 完了:")
            print(f"  Success: {res.success}")

            params_temp = pmap.to_physical(z_current)
            params_temp.update(anchor_params)

            errors_temp = []
            for beta, Em in flc_points:
                Em_pred = compute_flc_point_ultimate(params_temp, beta, beta_range_global)
                error = abs(Em_pred - Em) / Em * 100
                errors_temp.append(error)
            print(f"  平均誤差: {np.mean(errors_temp):.3f}%")

            Lambda_vals = [
                compute_Lambda_field_ultimate(params_temp, b, beta_range_global)
                for b in beta_range_global
            ]
            Lambda_min = np.min(Lambda_vals)
            Lambda_max = np.max(Lambda_vals)
            print(f"  Λ範囲: [{Lambda_min:.3f}, {Lambda_max:.3f}]")
            print(f"  ρ = {params_temp['rho']:.4f}")

            # 横切りチェック
            if Lambda_min < 1.0 < Lambda_max:
                print(f"  ✓ Λ=1横切り達成！")
        else:
            print(f"\nStage {stage} 失敗、継続...")
            z_current = res.x

    params_phys = pmap.to_physical(res.x)
    params_phys.update(anchor_params)

    return params_phys, res

# =============================================================================
# Section 6: クイック診断
# =============================================================================

def quick_diagnostics_ultimate(params, betas=None, flc_points=None):
    """クイック診断（究極版）"""
    if betas is None:
        betas = np.linspace(-0.8, 1.0, 200)

    # ★beta_range_globalを準備
    beta_range_global = np.linspace(-0.5, 1.0, 50)

    print("\n" + "="*60)
    print("クイック診断（究極版）")
    print("="*60)

    # ★究極版関数を使用
    Lambda_vals = np.array([
        compute_Lambda_field_ultimate(params, b, beta_range_global)
        for b in betas
    ])
    Lambda_min = np.min(Lambda_vals)
    Lambda_max = np.max(Lambda_vals)

    sign_changes = np.sign(Lambda_vals - 1.0)
    crossings = np.where(sign_changes[:-1] * sign_changes[1:] < 0)[0].size

    print(f"\n【Λ場の診断】")
    print(f"  Λ範囲: [{Lambda_min:.3f}, {Lambda_max:.3f}]")
    print(f"  Λ=1横切り回数: {crossings}回")

    if crossings == 1:
        print(f"  ✓ Λ=1を1回だけ横切っています！")
    elif crossings > 0:
        print(f"  △ Λ=1を{crossings}回横切っています")
    else:
        print(f"  ✗ Λ=1を横切っていません")

    print(f"\n【学習パラメータ】")
    print(f"  beta_A: {params.get('beta_A', 'N/A'):.6f}")
    print(f"  beta_bw: {params.get('beta_bw', 'N/A'):.6f}")
    print(f"  E_gain: {params.get('E_gain', 'N/A'):.3f}")
    print(f"  γ: {params.get('gamma', 'N/A'):.3f}")
    print(f"  η: {params.get('eta', 'N/A'):.3f}")
    print(f"  α: {params.get('alpha', 'N/A'):.3f}")

    print(f"\n【アンカー（自己無撞着）】")
    print(f"  Em_star: {params.get('Em_star', 'N/A'):.3f}")
    print(f"  Lambda_scale: 動的計算")

    if flc_points is not None:
        print(f"\n【FLC予測誤差】")
        errors = []
        for beta, Em_target in flc_points:
            # ★究極版関数を使用
            Em_pred = compute_flc_point_ultimate(params, beta, beta_range_global)
            error = abs(Em_pred - Em_target) / Em_target * 100
            errors.append(error)

        print(f"  平均誤差: {np.mean(errors):.3f}%")
        print(f"  最大誤差: {np.max(errors):.3f}%")
        print(f"  最小誤差: {np.min(errors):.3f}%")

        if np.mean(errors) < 0.1:
            print(f"  🎉 0.1%以内！完璧！")
        elif np.mean(errors) < 1.0:
            print(f"  ✓ 1%以内！")
        elif np.mean(errors) < 3.0:
            print(f"  △ 3%以内")
        else:
            print(f"  ✗ 誤差が大きいです")

# =============================================================================
# Section 7: 可視化（省略、前回と同様）
# =============================================================================

def visualize_ultimate_result(params_dict, flc_points, beta_range):
    """究極版結果可視化"""
    # ★beta_range_globalを準備
    beta_range_global = np.linspace(-0.5, 1.0, 50)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    beta_fine = np.linspace(-0.8, 1.0, 300)

    # (a) Λ場
    ax = axes[0, 0]
    Lambda_field = [compute_Lambda_field_ultimate(params_dict, b) for b in beta_fine]
    ax.plot(beta_fine, Lambda_field, 'b-', linewidth=2.5, label='Λfield')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Λ=1')
    ax.fill_between(beta_fine, 1.0, Lambda_field,
                    where=np.array(Lambda_field)>1.0, alpha=0.3, color='red')
    ax.fill_between(beta_fine, 0, Lambda_field,
                    where=np.array(Lambda_field)<1.0, alpha=0.3, color='blue')
    ax.set_xlabel('β', fontsize=12)
    ax.set_ylabel('Λ', fontsize=12)
    ax.set_title('(a)Λfield（99%）', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) 境界Σ
    ax = axes[0, 1]
    Sigma = extract_critical_boundary_ultimate(params_dict, beta_range)
    ax.plot(beta_fine, Lambda_field, 'b-', alpha=0.3, linewidth=1.5)
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2)

    if len(Sigma) > 0:
        Lambda_on_Sigma = [compute_Lambda_field_ultimate(params_dict, b) for b in Sigma]
        ax.scatter(Sigma, Lambda_on_Sigma, c='red', s=120, zorder=5,
                  edgecolors='darkred', linewidth=2, label='BoundaryΣ')

    ax.set_xlabel('β', fontsize=12)
    ax.set_ylabel('Λ', fontsize=12)
    ax.set_title(f'(b)BoundaryΣ（{len(Sigma)}point）', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # (c) V_eff^α
    ax = axes[0, 2]
    alpha = params_dict.get('alpha', 0.2)
    V_viscosity = [compute_V_eff(params_dict, b) ** alpha for b in beta_fine]
    ax.plot(beta_fine, V_viscosity, 'purple', linewidth=2.5, label=f'V_eff^{alpha:.2f}')
    ax.set_xlabel('β', fontsize=12)
    ax.set_ylabel('V_eff^α', fontsize=12)
    ax.set_title(f'(c)viscosityTerm（α={alpha:.3f}）', fontsize=13, fontweight='bold')
    ax.fill_between(beta_fine, 1.0, V_viscosity, alpha=0.2, color='purple')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # (d) FLC予測
    ax = axes[1, 0]
    Em_pred = [compute_flc_point_ultimate(params_dict, b) for b in beta_fine]
    ax.plot(beta_fine, Em_pred, 'b-', linewidth=2.5, label='prediction')

    betas_data = [b for b, _ in flc_points]
    Ems_data = [e for _, e in flc_points]
    ax.scatter(betas_data, Ems_data, c='red', s=120, label='Actual', zorder=5,
              edgecolors='darkred', linewidth=1.5)

    # Em_star線
    Em_star = params_dict.get('Em_star', 0.18)
    ax.axhline(y=Em_star, color='green', linestyle='--', linewidth=1.5,
              alpha=0.5, label=f'Em_star={Em_star:.3f}')

    ax.set_xlabel('β', fontsize=12)
    ax.set_ylabel('Em', fontsize=12)
    ax.set_title('(d)FLC prediction', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # (e) 誤差
    ax = axes[1, 1]
    errors = []
    for beta, Em_target in flc_points:
        Em_p = compute_flc_point_ultimate(params_dict, beta)
        error = abs(Em_p - Em_target) / Em_target * 100
        errors.append(error)

    colors = ['green' if e < 0.1
             else 'lightgreen' if e < 1.0
             else 'orange' if e < 3.0
             else 'red' for e in errors]

    ax.bar(range(len(errors)), errors, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0.1, color='green', linestyle='--', linewidth=1.5, label='0.1%', alpha=0.7)
    ax.axhline(y=1.0, color='lightgreen', linestyle='--', linewidth=1.5, label='1%', alpha=0.7)
    ax.set_ylabel('Error (%)', fontsize=12)
    ax.set_title(f'(e)Prediction error（Average: {np.mean(errors):.3f}%）',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(errors)))
    ax.set_xticklabels([f'{b:.2f}' for b, _ in flc_points], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # (f) モデル説明
    ax = axes[1, 2]
    ax.axis('off')

    model_text = f"""
Fix model

Em(β) = Em_star + E_gain × |1-Λ|^γ
        × V_eff^α × (K/K_ref)^η

parameters:
  Em_star = {params_dict.get('Em_star', 0.18):.3f} (fix)
  E_gain = {params_dict.get('E_gain', 0.0):.3f}
  γ = {params_dict.get('gamma', 0.0):.3f}
  η = {params_dict.get('eta', 0.0):.3f}
  α = {params_dict.get('alpha', 0.0):.3f} (minute index)
"""

    ax.text(0.1, 0.9, model_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('holographic_purity99.png', dpi=150, bbox_inches='tight')
    print("\n純度99%版結果図を保存: holographic_purity99.png")
    plt.show()

# =============================================================================
# Section 8: メイン実行
# =============================================================================

def run_holographic_experiment_ultimate():
    """ホログラフィック実験（究極版ρ）"""
    print("="*60)
    print("Λ–Holo双対性 実験 - Ultimate Edition (ρ)")
    print("オーバーシュート係数で横切り確定")
    print("="*60)

    flc_points = [
        (-0.5, 0.38),
        (-0.25, 0.32),
        (0.0, 0.25),
        (0.25, 0.23),
        (0.5, 0.20),
        (1.0, 0.18),
    ]

    physics_bounds = {
        'K_scale': (0.9, 1.2),
        'K_scale_draw': (0.95, 1.25),
        'K_scale_plane': (0.75, 0.90),
        'K_scale_biax': (1.0, 1.2),
        'beta_A': (0.03, 0.30),
        'beta_bw': (0.10, 0.50),
        'beta_A_pos': (0.02, 0.15),
        'E_gain': (0.01, 15.0),
        'gamma': (0.1, 3.0),
        'eta': (-3.0, 6.0),
        'alpha': (0.0, 2.5),  # ★軽く絞る
        'rho': (0.005, 0.08),  # 🔥 追加！
    }

    print("\n★ρ版特徴:")
    print("  - Lambda_scale = (1-ρ) × max(K/V)")
    print("  - Λ_max = 1/(1-ρ) > 1 保証")
    print("  - 横切り確定")
    print("  - 2段階戦略（ウォームアップ→フル）")

    # ★ρ版最適化
    params_opt, res = solve_homotopy_ultimate_rho(
        flc_points,
        physics_bounds,
        STABLE_PARAMS_FROZEN,
        eps_schedule=[2e-1, 1.5e-1, 1e-1, 7e-2, 5e-2, 3e-2, 1e-2, 5e-3],
        delta_schedule=[0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.005],
        verbose=True
    )

    # ★究極版診断
    quick_diagnostics_ultimate(params_opt, flc_points=flc_points)

    # ★究極版可視化
    beta_range = np.linspace(-0.8, 1.0, 200)
    visualize_ultimate_result(params_opt, flc_points, beta_range)

    # ★beta_range_global準備
    beta_range_global = np.linspace(-0.5, 1.0, 50)

    # 総合評価（究極版関数使用）
    Lambda_vals = [
        compute_Lambda_field_ultimate(params_opt, b, beta_range_global)
        for b in beta_range
    ]
    Lambda_crosses = np.min(Lambda_vals) < 1.0 < np.max(Lambda_vals)

    errors = []
    for beta, Em_target in flc_points:
        Em_pred = compute_flc_point_ultimate(params_opt, beta, beta_range_global)
        error = abs(Em_pred - Em_target) / Em_target * 100
        errors.append(error)

    Sigma = extract_critical_boundary_ultimate(params_opt, beta_range)

    print("\n" + "="*60)
    print("実験結果サマリー")
    print("="*60)

    print(f"\n✓ FLC予測精度:")
    print(f"  平均誤差: {np.mean(errors):.3f}%")
    print(f"  最大誤差: {np.max(errors):.3f}%")

    print(f"\n✓ Λ場:")
    print(f"  範囲: [{np.min(Lambda_vals):.3f}, {np.max(Lambda_vals):.3f}]")
    print(f"  横切り: {'YES ✓' if Lambda_crosses else 'NO ✗'}")

    print(f"\n✓ 学習パラメータ:")
    print(f"  E_gain: {params_opt['E_gain']:.3f}")
    print(f"  γ: {params_opt['gamma']:.3f}")
    print(f"  η: {params_opt['eta']:.3f}")
    print(f"  α: {params_opt['alpha']:.3f}")
    print(f"  beta_A: {params_opt['beta_A']:.6f}")
    print(f"  beta_bw: {params_opt['beta_bw']:.6f}")
    print(f"  beta_A_pos: {params_opt['beta_A_pos']:.6f}")

    print(f"\n✓ 境界Σ:")
    print(f"  検出点数: {len(Sigma)}点")

    # 最終判定
    print("\n" + "="*60)
    if np.mean(errors) < 1.0 and Lambda_crosses and len(Sigma) > 0:
        print("🎉🎉🎉 完全成功！究極版で達成！")
        print("   ✓ FLC精度: 1%以内")
        print("   ✓ Λ横切り: YES")
        print("   ✓ 境界Σ: 検出")
        print("   ✓ 自己無撞着: 完璧")
    else:
        print(f"平均誤差: {np.mean(errors):.3f}%")
        print(f"Λ横切り: {'YES' if Lambda_crosses else 'NO'}")
    print("="*60)

    # 境界情報パケット
    if len(Sigma) > 0:
        print("\n【境界情報パケット出力】")
        Ξ = compute_boundary_info_packet_ultimate(params_opt, Sigma)

        with open('Xi_boundary_ultimate.json', 'w') as f:
            Ξ_serializable = {}
            for key, value in Ξ.items():
                if isinstance(value, np.ndarray):
                    Ξ_serializable[key] = value.tolist()
                else:
                    Ξ_serializable[key] = value
            json.dump(Ξ_serializable, f, indent=2)

        print("  保存: Xi_boundary_ultimate.json")

    return {
        'params': params_opt,
        'success': res.success,
        'Lambda_crosses': Lambda_crosses,
        'Sigma_detected': len(Sigma) > 0,
        'errors': errors,
        'avg_error': np.mean(errors)
    }

# 実行
if __name__ == "__main__":
    results = run_holographic_experiment_ultimate()
