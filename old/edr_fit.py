"""
=============================================================================
EDRパラメータフィッティング統合版 v5.2 (JAX + CUDA)
+ Operation Marie Antoinette（逆問題データ生成）
Miosync, Inc. / Inverse-EDR Neural Calibration Engine (IENCE)

【概要】
板材成形における破壊予測のための統一理論（EDR理論）実装
- JAX版：メイン実装（CPU/GPU両対応、自動微分可能）
- CUDA版：大量並列評価用（オプション）
- Operation Marie Antoinette：逆問題×多様体学習によるデータ生成

【最適化戦略】
多段階Hybrid最適化：
  Phase 0: Unsupervised FLC Pretraining（物理制約のみ）
  Phase 1: FLC形状確立＋段階的バイナリ統合
  Phase 1.5B: 制約付き多様体最適化（NEW!）
  Phase 2: L-BFGS-B（局所精密化）- オプション

【著者】
飯泉 真道 (Masamichi Iizumi)
環 (Tamaki) - AI Co-Developer

【日付】
2025-01-19 (v5.2: Operation Marie Antoinette統合版)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Optional
from scipy.optimize import minimize, Bounds, differential_evolution
from scipy.signal import savgol_filter
from collections import deque
import time

# JAX関連（必須）
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    import optax
    JAX_AVAILABLE = True
    print(f"✓ JAX利用可能: バージョン {jax.__version__}")
except ImportError:
    raise ImportError("JAXが必要です: pip install jax jaxlib")

# CUDA関連（オプション）
CUDA_AVAILABLE = False
try:
    from numba import cuda, float64, int32
    import math
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"✓ CUDA利用可能: {cuda.get_current_device().name.decode()}")
    else:
        print("⚠️  CUDA無効: JAX modeで実行")
except ImportError:
    print("⚠️  Numba未インストール: JAX modeのみ")

# Operation Marie Antoinette
print("🎂 Operation Marie Antoinette: ロード中...")
try:
    from operation_marie_antoinette import (
        build_safe_manifold,
        compute_safety_score,
        loss_binary_manifold,
        phase_15b_manifold_optimization,
        analyze_safety_scores,
        visualize_safe_manifold
    )
    MARIE_ANTOINETTE_AVAILABLE = True
    print("✓ Operation Marie Antoinette: 利用可能")
except ImportError:
    MARIE_ANTOINETTE_AVAILABLE = False
    print("⚠️  Operation Marie Antoinette: 未インストール")

# =============================================================================
# Section 1: データ構造
# =============================================================================

@dataclass
class MaterialParams:
    """材料パラメータ"""
    rho: float = 7800.0      # 密度 [kg/m3]
    cp: float = 500.0        # 比熱 [J/kg/K]
    k: float = 40.0          # 熱伝導率 [W/m/K]
    thickness: float = 0.0008 # 板厚 [m]
    sigma0: float = 600e6    # 初期降伏応力 [Pa]
    n: float = 0.15          # 加工硬化指数
    m: float = 0.02          # 速度感受指数
    r_value: float = 1.0     # ランクフォード値

@dataclass
class EDRParams:
    """EDR理論パラメータ"""
    V0: float = 2e9            # 基準凝集エネルギー [Pa = J/m3]
    av: float = 3e4            # 空孔影響係数
    ad: float = 1e-7           # 転位影響係数
    chi: float = 0.1           # 摩擦発熱の内部分配率
    K_scale: float = 0.2       # K総量スケール
    triax_sens: float = 0.3    # 三軸度感度
    Lambda_crit: float = 1.0   # 臨界Λ
    # 経路別スケール係数
    K_scale_draw: float = 0.15   # 深絞り用
    K_scale_plane: float = 0.25  # 平面ひずみ用
    K_scale_biax: float = 0.20   # 等二軸用
    # FLC V字パラメータ
    beta_A: float = 0.35       # 谷の深さ
    beta_bw: float = 0.28      # 谷の幅
    # 非対称パラメータ（オプション）
    beta_A_pos: float = 0.50   # 等二軸側の深さ

@dataclass
class PressSchedule:
    """FEM or 実験ログの時系列データ"""
    t: np.ndarray                 # 時間 [s]
    eps_maj: np.ndarray           # 主ひずみ
    eps_min: np.ndarray           # 副ひずみ
    triax: np.ndarray             # 三軸度 σm/σeq
    mu: np.ndarray                # 摩擦係数
    pN: np.ndarray                # 接触圧 [Pa]
    vslip: np.ndarray             # すべり速度 [m/s]
    htc: np.ndarray               # 熱伝達係数 [W/m2/K]
    Tdie: np.ndarray              # 金型温度 [K]
    contact: np.ndarray           # 接触率 [0-1]
    T0: float = 293.15            # 板の初期温度 [K]

@dataclass
class ExpBinary:
    """破断/安全のラベル付き実験"""
    schedule: PressSchedule
    failed: int                   # 1:破断, 0:安全
    label: str = ""

@dataclass
class FLCPoint:
    """FLC: 経路比一定での限界点（実測）"""
    path_ratio: float            # β = eps_min/eps_maj
    major_limit: float           # 実測限界主ひずみ
    minor_limit: float           # 実測限界副ひずみ
    rate_major: float = 1.0      # 主ひずみ速度 [1/s]
    duration_max: float = 1.0    # 試験上限時間 [s]
    label: str = ""

# =============================================================================
# Section 2: JAXヘルパー関数
# =============================================================================

def schedule_to_jax_dict(schedule: PressSchedule):
    """PressSchedule → JAX用dict変換"""
    return {
        't': jnp.array(schedule.t),
        'eps_maj': jnp.array(schedule.eps_maj),
        'eps_min': jnp.array(schedule.eps_min),
        'triax': jnp.array(schedule.triax),
        'mu': jnp.array(schedule.mu),
        'pN': jnp.array(schedule.pN),
        'vslip': jnp.array(schedule.vslip),
        'htc': jnp.array(schedule.htc),
        'Tdie': jnp.array(schedule.Tdie),
        'contact': jnp.array(schedule.contact),
        'T0': schedule.T0
    }

def mat_to_jax_dict(mat: MaterialParams):
    """MaterialParams → JAX用dict変換"""
    return {
        'rho': mat.rho,
        'cp': mat.cp,
        'h0': mat.thickness,
        'sigma0': mat.sigma0,
        'n': mat.n,
        'm': mat.m,
        'r_value': mat.r_value
    }

@jit
def soft_clamp(x, min_val, max_val):
    """ソフト境界制約（微分可能）"""
    return min_val + (max_val - min_val) * jax.nn.sigmoid(x)

@jit
def triax_from_path_jax(beta):
    """ひずみ経路比βから三軸度ηを計算"""
    b = jnp.clip(beta, -0.95, 1.0)
    return (1.0 + b) / (jnp.sqrt(3.0) * jnp.sqrt(1.0 + b + b*b))

@jit
def equiv_strain_rate_jax(epsM_dot, epsm_dot):
    """相当ひずみ速度"""
    sqrt_2_3 = 0.8164965809277260
    return sqrt_2_3 * jnp.sqrt(
        (epsM_dot - epsm_dot)**2 + epsM_dot**2 + epsm_dot**2
    )

@jit
def flow_stress_jax(ep_eq, epdot_eq, sigma0, n, m, r_value, T):
    """温度依存の流動応力"""
    Tref = 293.15
    alpha = 3e-4
    rate_fac = jnp.power(jnp.maximum(epdot_eq, 1e-6), m)
    aniso = (2.0 + r_value) / 3.0
    temp_fac = 1.0 - alpha * jnp.maximum(T - Tref, 0.0)
    return sigma0 * temp_fac * jnp.power(1.0 + ep_eq, n) * rate_fac / aniso

@jit
def step_cv_jax(cv, T, rho_d, dt):
    """空孔濃度の時間発展"""
    kB_eV = 8.617e-5
    c0 = 1e-6; Ev_eV = 1.0; tau0 = 1e-3; Q_eV = 0.8
    k_ann = 1e6; k_sink = 1e-15
    
    cv_eq = c0 * jnp.exp(-Ev_eV / (kB_eV * T))
    tau = tau0 * jnp.exp(Q_eV / (kB_eV * T))
    dcv = (cv_eq - cv) / tau - k_ann * cv**2 - k_sink * cv * rho_d
    return cv + dcv * dt

@jit
def step_rho_jax(rho_d, epdot_eq, T, dt):
    """転位密度の時間発展"""
    A = 1e14; B = 1e-4; Qv_eV = 0.8; kB_eV = 8.617e-5
    Dv = 1e-6 * jnp.exp(-Qv_eV / (kB_eV * T))
    drho = A * jnp.maximum(epdot_eq, 0.0) - B * rho_d * Dv
    return jnp.maximum(rho_d + drho * dt, 1e10)

@jit
def get_k_scale_smooth_jax(beta, params):
    """滑らかなK_scale選択（分岐レス）"""
    w_draw = jnp.exp(-((beta + 0.5) / 0.1)**2)
    w_plane = jnp.exp(-(beta / 0.1)**2)
    w_biax = jnp.exp(-((beta - 0.5) / 0.2)**2)
    w_else = 1.0 - jnp.maximum(w_draw, jnp.maximum(w_plane, w_biax))
    
    w_sum = w_draw + w_plane + w_biax + w_else + 1e-8
    
    return (params["K_scale_draw"] * w_draw + 
            params["K_scale_plane"] * w_plane +
            params["K_scale_biax"] * w_biax +
            params["K_scale"] * w_else) / w_sum

@jit
def beta_multiplier_jax(beta, A, bw):
    """β依存ゲイン（V字形状）"""
    b = jnp.clip(beta, -0.95, 0.95)
    return 1.0 + A * jnp.exp(-(b / bw)**2)

@jit
def beta_multiplier_asymmetric_jax(beta, A_neg, A_pos, bw):
    """非対称β依存ゲイン"""
    b = jnp.clip(beta, -0.95, 0.95)
    A = jnp.where(b < 0, A_neg, A_pos)
    return 1.0 + A * jnp.exp(-(b / bw)**2)

@jit
def mu_effective_jax(mu0, T, pN, vslip):
    """温度・速度・荷重依存の有効摩擦係数（Stribeck風）"""
    s = (vslip * 1e3) / (pN / 1e6 + 1.0)
    stribeck = 0.7 + 0.3 / (1 + s)
    temp_reduction = 1.0 - 1e-4 * jnp.maximum(T - 293.15, 0)
    return mu0 * stribeck * temp_reduction

def smooth_signal_jax(x, window_size=11):
    """JAX版移動平均によるスムージング（スパイク除去）"""
    if window_size <= 1 or len(x) <= window_size:
        return x
    kernel = jnp.ones(window_size) / window_size
    # JAX版のパディングと畳み込み
    padded = jnp.pad(x, (window_size//2, window_size//2), mode='edge')
    smoothed = jnp.convolve(padded, kernel, mode='valid')
    return smoothed[:len(x)]

# =============================================================================
# Section 3: メインシミュレーション（JAX版）
# =============================================================================

@jit
def simulate_lambda_jax(schedule_dict, mat_dict, edr_dict):
    """JAX版シミュレーション（メイン実装）"""
    
    # データ取り出し
    t = schedule_dict['t']
    epsM = schedule_dict['eps_maj']
    epsm = schedule_dict['eps_min']
    triax = schedule_dict['triax']
    mu = schedule_dict['mu']
    pN = schedule_dict['pN']
    vslip = schedule_dict['vslip']
    htc = schedule_dict['htc']
    Tdie = schedule_dict['Tdie']
    contact = schedule_dict['contact']
    T0 = schedule_dict['T0']
    
    dt = (t[-1] - t[0]) / (len(t) - 1)
    
    # ひずみ速度計算
    epsM_dot = jnp.gradient(epsM, dt)
    epsm_dot = jnp.gradient(epsm, dt)
    
    # 経路平均β
    beta_avg = jnp.mean(epsm / (epsM + 1e-10))
    
    # scanで時間ループ
    def time_step(carry, inputs):
        T, cv, rho_d, ep_eq, h_eff, eps3, beta_hist = carry
        idx = inputs
        
        epsM_dot_t = epsM_dot[idx]
        epsm_dot_t = epsm_dot[idx]
        triax_t = triax[idx]
        mu_t = mu[idx]
        pN_t = pN[idx]
        vslip_t = vslip[idx]
        htc_t = htc[idx]
        Tdie_t = Tdie[idx]
        contact_t = contact[idx]
        
        # 相当ひずみ速度
        epdot_eq = equiv_strain_rate_jax(epsM_dot_t, epsm_dot_t)
        
        # 板厚更新
        d_eps3 = -(epsM_dot_t + epsm_dot_t) * dt
        eps3_new = eps3 + d_eps3
        h_eff_new = jnp.maximum(mat_dict['h0'] * jnp.exp(eps3_new), 0.2 * mat_dict['h0'])
        
        # 熱収支
        q_fric = mu_t * pN_t * vslip_t * contact_t
        dTdt = (2.0 * htc_t * (Tdie_t - T) + 2.0 * edr_dict['chi'] * q_fric) / \
               (mat_dict['rho'] * mat_dict['cp'] * h_eff_new)
        dTdt = jnp.clip(dTdt, -1000.0, 1000.0)
        T_new = jnp.clip(T + dTdt * dt, 200.0, 2000.0)
        
        # 欠陥更新
        rho_d_new = step_rho_jax(rho_d, epdot_eq, T, dt)
        cv_new = step_cv_jax(cv, T, rho_d_new, dt)
        
        # K計算（改善版：冷却は回復側）
        K_th = mat_dict['rho'] * mat_dict['cp'] * jnp.maximum(dTdt, 0.0)  # 加熱時のみカウント！
        
        # 温度依存の流動応力
        sigma_eq = flow_stress_jax(ep_eq, epdot_eq, mat_dict['sigma0'], 
                                   mat_dict['n'], mat_dict['m'], mat_dict['r_value'], T)
        K_pl = 0.9 * sigma_eq * epdot_eq
        
        # 温度・速度・荷重依存の摩擦係数（物理増強）
        mu_eff = mu_effective_jax(mu_t, T, pN_t, vslip_t)
        q_fric_eff = mu_eff * pN_t * vslip_t * contact_t
        K_fr = (2.0 * edr_dict['chi'] * q_fric_eff) / h_eff_new
        
        # K_scale選択
        k_scale_path = get_k_scale_smooth_jax(beta_avg, edr_dict)
        
        # β瞬間値とβ履歴の5点移動平均
        beta_inst = epsm_dot_t / (epsM_dot_t + 1e-8)
        
        # β履歴更新（5点移動平均）
        beta_hist_new = jnp.roll(beta_hist, -1).at[4].set(beta_inst)
        beta_smooth = jnp.mean(beta_hist_new)
        
        # K_total（非対称ゲイン使用）
        K_total = k_scale_path * (K_th + K_pl + K_fr)
        K_total *= beta_multiplier_asymmetric_jax(
            beta_smooth, 
            edr_dict['beta_A'], 
            edr_dict.get('beta_A_pos', edr_dict['beta_A']),
            edr_dict['beta_bw']
        )
        K_total = jnp.maximum(K_total, 0.0)
        K_total = jnp.where(jnp.isnan(K_total), 0.0, K_total)  # NaN対策
        
        # V_eff（温度依存性を強化）
        T_ratio = jnp.minimum((T - 273.15) / (1500.0 - 273.15), 1.0)
        temp_factor = 1.0 - 0.5 * T_ratio  # 温度が上がるとV_effが下がる
        V_eff = edr_dict['V0'] * temp_factor * \
                (1.0 - edr_dict['av'] * cv - edr_dict['ad'] * jnp.sqrt(jnp.maximum(rho_d, 1e10)))
        V_eff = jnp.maximum(V_eff, 0.01 * edr_dict['V0'])
        V_eff = jnp.where(jnp.isnan(V_eff), edr_dict['V0'], V_eff)  # NaN対策
        
        # 三軸度補正（感度を調整）
        D_triax = jnp.exp(-edr_dict['triax_sens'] * jnp.maximum(triax_t, 0.0))
        D_triax = jnp.where(jnp.isnan(D_triax), 1.0, D_triax)  # NaN対策
        
        # Λ計算
        Lambda = K_total / jnp.maximum(V_eff * D_triax, 1e7)
        Lambda = jnp.minimum(Lambda, 10.0)
        Lambda = jnp.where(jnp.isnan(Lambda), 0.0, Lambda)  # NaN対策
        
        # 相当塑性ひずみ更新
        ep_eq_new = ep_eq + epdot_eq * dt
        
        new_carry = (T_new, cv_new, rho_d_new, ep_eq_new, h_eff_new, eps3_new, beta_hist_new)
        return new_carry, Lambda
    
    # 初期状態（β履歴も初期化）
    init_beta_hist = jnp.zeros(5)  # 5点移動平均用
    init_carry = (T0, 1e-7, 1e11, 0.0, mat_dict['h0'], 0.0, init_beta_hist)
    
    # scan実行
    indices = jnp.arange(len(t) - 1)
    _, Lambdas = jax.lax.scan(time_step, init_carry, indices)
    
    # Damage積分
    Damage = jnp.cumsum(jnp.maximum(Lambdas - edr_dict['Lambda_crit'], 0.0) * dt)
    
    return {"Lambda": Lambdas, "Damage": Damage}

# =============================================================================
# Section 4: パラメータ管理と損失関数
# =============================================================================

def init_edr_params_jax():
    """JAX用パラメータ初期化（バランス調整版）"""
    return {
        'log_V0': jnp.log(1.5e9),
        'log_av': jnp.log(4e4),
        'log_ad': jnp.log(1e-7),
        'logit_chi': jnp.log(0.09 / (1 - 0.09)),
        'logit_K_scale': jnp.log(0.25 / (1 - 0.25)),
        'logit_K_scale_draw': jnp.log(0.18 / (1 - 0.18)),
        'logit_K_scale_plane': jnp.log(0.25 / (1 - 0.25)),
        'logit_K_scale_biax': jnp.log(0.22 / (1 - 0.22)),
        'logit_triax_sens': jnp.log(0.28 / (1 - 0.28)),
        'Lambda_crit': jnp.array(0.97),
        'logit_beta_A': jnp.log(0.32 / (1 - 0.32)),
        'logit_beta_bw': jnp.log(0.29 / (1 - 0.29)),
        'logit_beta_A_pos': jnp.log(0.48 / (1 - 0.48)),
    }

def transform_params_jax(raw_params):
    """制約付きパラメータ変換（soft bounds）"""
    return {
        'V0': jnp.exp(raw_params['log_V0']),
        'av': jnp.exp(raw_params['log_av']),
        'ad': jnp.exp(raw_params['log_ad']),
        'chi': soft_clamp(raw_params['logit_chi'], 0.05, 0.3),
        'K_scale': soft_clamp(raw_params['logit_K_scale'], 0.05, 1.0),
        'K_scale_draw': soft_clamp(raw_params['logit_K_scale_draw'], 0.05, 0.3),
        'K_scale_plane': soft_clamp(raw_params['logit_K_scale_plane'], 0.1, 0.4),
        'K_scale_biax': soft_clamp(raw_params['logit_K_scale_biax'], 0.05, 0.3),
        'triax_sens': soft_clamp(raw_params['logit_triax_sens'], 0.1, 0.5),
        'Lambda_crit': jnp.clip(raw_params['Lambda_crit'], 0.95, 1.05),
        'beta_A': soft_clamp(raw_params['logit_beta_A'], 0.2, 0.5),
        'beta_bw': soft_clamp(raw_params['logit_beta_bw'], 0.2, 0.35),
        'beta_A_pos': soft_clamp(raw_params['logit_beta_A_pos'], 0.3, 0.7),
    }

def edr_dict_to_dataclass(edr_dict):
    """dict → EDRParams変換（JAX値を取得）"""
    return EDRParams(
        V0=float(jax.device_get(edr_dict['V0'])),
        av=float(jax.device_get(edr_dict['av'])),
        ad=float(jax.device_get(edr_dict['ad'])),
        chi=float(jax.device_get(edr_dict['chi'])),
        K_scale=float(jax.device_get(edr_dict['K_scale'])),
        triax_sens=float(jax.device_get(edr_dict['triax_sens'])),
        Lambda_crit=float(jax.device_get(edr_dict['Lambda_crit'])),
        K_scale_draw=float(jax.device_get(edr_dict['K_scale_draw'])),
        K_scale_plane=float(jax.device_get(edr_dict['K_scale_plane'])),
        K_scale_biax=float(jax.device_get(edr_dict['K_scale_biax'])),
        beta_A=float(jax.device_get(edr_dict['beta_A'])),
        beta_bw=float(jax.device_get(edr_dict['beta_bw'])),
        beta_A_pos=float(jax.device_get(edr_dict.get('beta_A_pos', edr_dict['beta_A']))),
    )

def loss_single_exp_jax(schedule_dict, mat_dict, edr_dict, failed):
    """単一実験の損失（JAX版）"""
    res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
    
    # スムージングしてスパイクを除去（重要！）
    Lambda_smooth = smooth_signal_jax(res["Lambda"], window_size=11)
    
    peak = jnp.max(Lambda_smooth)
    D_end = res["Damage"][-1]
    
    margin = 0.08
    Dcrit = 0.01
    delta = 0.03
    
    # failed == 1の場合（破断）
    condition_met = (peak > edr_dict['Lambda_crit']) & (D_end > Dcrit)
    
    peak_penalty = jnp.maximum(0.0, edr_dict['Lambda_crit'] - peak)
    D_penalty = jnp.maximum(0.0, Dcrit - D_end)
    loss_fail_not_met = 10.0 * (peak_penalty**2 + D_penalty**2)
    
    margin_loss = jnp.where(
        peak < edr_dict['Lambda_crit'] + margin,
        (edr_dict['Lambda_crit'] + margin - peak)**2,
        0.0
    )
    D_loss = jnp.where(
        D_end < 2*Dcrit,
        (2*Dcrit - D_end)**2,
        0.0
    )
    loss_fail_met = margin_loss + D_loss
    
    loss_fail = jnp.where(condition_met, loss_fail_met, loss_fail_not_met)
    
    # failed == 0の場合（安全）
    loss_safe_peak = jnp.where(
        peak > edr_dict['Lambda_crit'] - delta,
        (peak - (edr_dict['Lambda_crit'] - delta))**2 * 3.0,
        0.0
    )
    loss_safe_D = jnp.where(
        D_end >= 0.5*Dcrit,
        10.0 * (D_end - 0.5*Dcrit)**2,
        0.0
    )
    loss_safe = loss_safe_peak + loss_safe_D
    
    return jnp.where(failed == 1, loss_fail, loss_safe)

def loss_fn_jax(raw_params, exps, mat):
    """バッチ損失関数"""
    edr_dict = transform_params_jax(raw_params)
    
    # データ変換
    mat_dict = mat_to_jax_dict(mat)
    
    total_loss = 0.0
    for exp in exps:
        schedule_dict = schedule_to_jax_dict(exp.schedule)
        loss = loss_single_exp_jax(schedule_dict, mat_dict, edr_dict, exp.failed)
        total_loss += loss
    
    return total_loss / len(exps)

def predict_flc_from_sim_jax(beta, mat_dict, edr_dict, 
                              major_rate=0.6, duration=1.0):
    """Λ(t)シミュレーションからFLC限界点を微分可能に抽出"""
    dt = 1e-3
    N = int(duration/dt) + 1
    t = jnp.linspace(0, duration, N)
    epsM = major_rate * t
    epsm = beta * epsM
    
    schedule_dict = {
        't': t,
        'eps_maj': epsM,
        'eps_min': epsm,
        'triax': jnp.full(N, triax_from_path_jax(beta)),
        'mu': jnp.full(N, 0.08),
        'pN': jnp.full(N, 200e6),
        'vslip': jnp.full(N, 0.02),
        'htc': jnp.full(N, 8000.0),
        'Tdie': jnp.full(N, 293.15),
        'contact': jnp.full(N, 1.0),
        'T0': 293.15
    }
    
    res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
    Lambda_raw = res["Lambda"]  # 長さはN-1
    
    # スムージング
    Lambda_smooth = smooth_signal_jax(Lambda_raw, window_size=11)
    
    # epsM配列をLambdaの長さに合わせる（最後の要素を除く）
    epsM_trimmed = epsM[:-1]
    
    # 微分可能な限界点検出
    exceed = jnp.maximum(Lambda_smooth - edr_dict['Lambda_crit'], 0.0)
    w = jnp.exp(jnp.minimum(10.0 * exceed, 10.0))
    w = w / (jnp.sum(w) + 1e-12)
    Em = jnp.sum(w * epsM_trimmed)
    Em = jnp.where(jnp.isnan(Em), epsM_trimmed[-1], Em)
    em = beta * Em
    
    return Em, em, Lambda_smooth

def loss_flc_true_jax(raw_params, flc_pts_data, mat_dict):
    """真のΛシミュレーションベースFLC損失"""
    edr_dict = transform_params_jax(raw_params)
    
    # FLC点ごとの誤差を順次計算
    flc_err = 0.0
    for i in range(len(flc_pts_data['path_ratios'])):
        beta = flc_pts_data['path_ratios'][i]
        Em_gt = flc_pts_data['major_limits'][i]
        em_gt = flc_pts_data['minor_limits'][i]
        
        Em_pred, em_pred, _ = predict_flc_from_sim_jax(beta, mat_dict, edr_dict)
        
        # β依存の重み付け（等二軸を最重視）
        w = jnp.where(jnp.abs(beta - 0.5) < 0.1, 6.0,
                      jnp.where(jnp.abs(beta) < 0.1, 1.8, 1.0))
        
        flc_err += w * ((Em_pred - Em_gt)**2 + (em_pred - em_gt)**2)
    
    flc_err = flc_err / len(flc_pts_data['path_ratios'])
    
    # V字形状と滑らかさの正則化
    beta_batch = jnp.linspace(-0.6, 0.6, 21)
    
    # 正則化用のFLC曲線を計算
    Em_curve = []
    for beta in beta_batch:
        Em, _, _ = predict_flc_from_sim_jax(beta, mat_dict, edr_dict)
        Em_curve.append(Em)
    Em_curve = jnp.array(Em_curve)
    
    # V字形状の正則化
    center_idx = len(beta_batch) // 2
    center = Em_curve[center_idx]
    valley_loss = 0.1 * jnp.mean(jnp.maximum(0.0, Em_curve - center))
    
    # 曲率の正則化（滑らかさ）
    grad2 = jnp.diff(jnp.diff(Em_curve))
    smoothness_loss = 0.05 * jnp.mean(grad2**2) + 0.02 * jnp.mean(jnp.abs(grad2))
    
    # 動的重み付け（分散に応じて正則化を調整）
    valley_weight = jnp.clip(jnp.var(Em_curve), 0.05, 0.3)
    valley_loss = valley_weight * valley_loss
    
    total_loss = flc_err + valley_loss + smoothness_loss
    
    return total_loss

# =============================================================================
# Section 5: 多段階Hybrid最適化
# =============================================================================

def hybrid_staged_optimization(
    exps: List[ExpBinary],
    flc_pts: List[FLCPoint],
    mat: MaterialParams,
    initial_edr: Optional[EDRParams] = None,
    verbose: bool = True
) -> Tuple[EDRParams, Dict]:
    """
    多段階Hybrid最適化
    Phase 0: Unsupervised FLC Pretraining（物理制約のみ）
    Phase 1: FLC形状確立＋段階的バイナリ統合
    Phase 2: オプション（必要に応じてL-BFGS-B）
    """
    
    if initial_edr is None:
        initial_edr = EDRParams()
    
    mat_dict = mat_to_jax_dict(mat)
    
    # ===========================
    # Phase 0: Unsupervised FLC Pretraining
    # ===========================
    if verbose:
        print("\n" + "="*60)
        print(" Phase 0: Unsupervised FLC Manifold Learning")
        print("="*60)
        print("  物理制約のみでFLC面を事前学習")
    
    # Phase 0用の安定版FLC予測
    def predict_flc_jax_stable(path_ratio, edr_dict, mat_dict):
        """Phase 0用の安定版FLC予測（argmax使用）"""
        duration = 1.0
        major_rate = 0.6
        dt = 1e-3
        N = int(duration/dt) + 1
        t = jnp.linspace(0, duration, N)
        epsM = major_rate * t
        epsm = path_ratio * epsM
        
        schedule_dict = {
            't': t,
            'eps_maj': epsM,
            'eps_min': epsm,
            'triax': jnp.full(N, triax_from_path_jax(path_ratio)),
            'mu': jnp.full(N, 0.08),
            'pN': jnp.full(N, 200e6),
            'vslip': jnp.full(N, 0.02),
            'htc': jnp.full(N, 8000.0),
            'Tdie': jnp.full(N, 293.15),
            'contact': jnp.full(N, 1.0),
            'T0': 293.15
        }
        
        res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
        Lambda_smooth = smooth_signal_jax(res["Lambda"], window_size=11)
        
        # 安定版：argmax使用
        exceed_mask = Lambda_smooth > edr_dict['Lambda_crit']
        first_exceed = jnp.argmax(exceed_mask)
        has_exceeded = jnp.any(exceed_mask)
        
        epsM_trimmed = epsM[:-1]
        Em = jnp.where(has_exceeded, epsM_trimmed[first_exceed], epsM_trimmed[-1])
        em = path_ratio * Em
        
        return Em, em
    
    # Phase 0: 教師なしFLC面学習
    def loss_phase0(raw_params):
        """物理制約のみでFLC面を学習"""
        edr_dict = transform_params_jax(raw_params)
        
        beta_grid = jnp.linspace(-0.8, 0.8, 13)
        
        # 各βでの仮想FLC限界を計算
        Em_grid = []
        for beta in beta_grid:
            Em, _ = predict_flc_jax_stable(beta, edr_dict, mat_dict)
            Em = jnp.where(jnp.isnan(Em), 0.3, Em)
            Em = jnp.clip(Em, 0.1, 0.8)
            Em_grid.append(Em)
        
        Em_array = jnp.array(Em_grid)
        Em_array = jnp.where(jnp.isnan(Em_array), 0.3, Em_array)
        
        # 物理制約1: 単調性
        monotonicity_loss = jnp.mean(jnp.maximum(0, -jnp.diff(jnp.abs(Em_array))))
        monotonicity_loss = jnp.where(jnp.isnan(monotonicity_loss), 0.0, monotonicity_loss)
        
        # 物理制約2: 凸性（V字形状）
        center = len(beta_grid) // 2
        left_branch = Em_array[:center]
        right_branch = Em_array[center:]
        
        convexity_loss = jnp.mean(jnp.maximum(0, jnp.diff(left_branch))) + \
                         jnp.mean(jnp.maximum(0, -jnp.diff(right_branch)))
        convexity_loss = jnp.where(jnp.isnan(convexity_loss), 0.0, convexity_loss)
        
        # 物理制約3: 対称性（破れを許容）
        asymmetry_factor = jnp.clip(edr_dict['beta_A_pos'] / (edr_dict['beta_A'] + 1e-8), 0.5, 2.0)
        symmetry_target = Em_array[::-1] * asymmetry_factor
        symmetry_loss = 0.1 * jnp.mean((Em_array - symmetry_target)**2)
        symmetry_loss = jnp.where(jnp.isnan(symmetry_loss), 0.0, symmetry_loss)
        
        # 物理制約4: 平滑性
        grad2 = jnp.diff(jnp.diff(Em_array))
        smoothness_loss = 0.05 * jnp.mean(grad2**2)
        smoothness_loss = jnp.where(jnp.isnan(smoothness_loss), 0.0, smoothness_loss)
        
        # 物理制約5: 合理的な範囲
        range_loss = jnp.mean(jnp.maximum(0, 0.1 - Em_array)**2) + \
                     jnp.mean(jnp.maximum(0, Em_array - 1.0)**2)
        range_loss = jnp.where(jnp.isnan(range_loss), 0.0, range_loss)
        
        total_loss = monotonicity_loss + convexity_loss + symmetry_loss + \
                    smoothness_loss + range_loss
        
        total_loss = jnp.where(jnp.isnan(total_loss), 1e10, total_loss)
        
        return total_loss
    
    # Phase 0の初期化
    params_phase0 = init_edr_params_jax()
    
    # Phase 0最適化
    schedule_phase0 = optax.exponential_decay(
        init_value=3e-3,
        transition_steps=50,
        decay_rate=0.92
    )
    
    optimizer_phase0 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_phase0)
    )
    
    opt_state_phase0 = optimizer_phase0.init(params_phase0)
    grad_fn_phase0 = jax.grad(loss_phase0)
    
    for step in range(300):
        grads = grad_fn_phase0(params_phase0)
        updates, opt_state_phase0 = optimizer_phase0.update(grads, opt_state_phase0, params_phase0)
        params_phase0 = optax.apply_updates(params_phase0, updates)
        
        if step % 100 == 0 and verbose:
            loss = loss_phase0(params_phase0)
            print(f"  Step {step:3d}: Physics Loss = {loss:.6f}")
    
    if verbose:
        final_loss_phase0 = loss_phase0(params_phase0)
        print(f"\n  Phase 0完了: Physics Loss = {final_loss_phase0:.6f}")
        print("  物理的に妥当なFLC面の初期化完了")
    
    # ===========================
    # Phase 1: FLC形状確立＋段階的バイナリ統合
    # ===========================
    if verbose:
        print("\n" + "="*60)
        print(" Phase 1: FLC形状確立＋段階的バイナリ統合")
        print("="*60)
        print("  FLCを基礎として段階的にバイナリ性能を積み上げ")
    
    # Phase 0の結果を使用
    params_main = params_phase0.copy()
    
    # FLCデータ準備
    if flc_pts:
        # 仮想FLC点を追加（データ増強）
        virtual_flc_pts = [
            FLCPoint(-0.3, 0.32, -0.096, 0.6, 1.0, "virtual_1"),
            FLCPoint(0.3, 0.25, 0.075, 0.6, 1.0, "virtual_2"),
            FLCPoint(-0.7, 0.38, -0.266, 0.6, 1.0, "virtual_3"),
            FLCPoint(0.7, 0.18, 0.126, 0.6, 1.0, "virtual_4"),
        ]
        
        all_flc_pts = flc_pts + virtual_flc_pts
        flc_pts_data = {
            'path_ratios': jnp.array([p.path_ratio for p in all_flc_pts]),
            'major_limits': jnp.array([p.major_limit for p in all_flc_pts]),
            'minor_limits': jnp.array([p.minor_limit for p in all_flc_pts])
        }
    
    # Step 1.1: β系確立（FLC 100%）
    if verbose:
        print("\n--- Step 1.1: β系確立（FLC形状の基礎） ---")
        print("  最適化: beta_A, beta_bw, beta_A_pos")
        print("  損失: FLC 100%")
    
    beta_keys = ['logit_beta_A', 'logit_beta_bw', 'logit_beta_A_pos']
    
    def loss_step1(params):
        # β系のみ最適化、他は固定
        active_params = {k: params[k] for k in beta_keys}
        params_temp = params_main.copy()
        params_temp.update(active_params)
        
        # FLC損失のみ
        return loss_flc_true_jax(params_temp, flc_pts_data, mat_dict)
    
    # AdamWで最適化
    beta_params = {k: params_main[k] for k in beta_keys}
    optimizer = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
    opt_state = optimizer.init(beta_params)
    grad_fn = jax.grad(loss_step1)
    
    for step in range(300):
        grads = grad_fn(beta_params)
        updates, opt_state = optimizer.update(grads, opt_state, beta_params)
        beta_params = optax.apply_updates(beta_params, updates)
        
        if step % 100 == 0 and verbose:
            loss = loss_step1(beta_params)
            print(f"    Step {step}: FLC Loss = {loss:.6f}")
    
    params_main.update(beta_params)
    
    # Step 1.2: K_scale系追加（FLC 90% + Binary 10%）
    if verbose:
        print("\n--- Step 1.2: K_scale系追加 ---")
        print("  最適化: K_scale_draw/plane/biax, triax_sens")
        print("  損失: FLC 90% + Binary 10%")
    
    k_keys = ['logit_K_scale', 'logit_K_scale_draw', 
              'logit_K_scale_plane', 'logit_K_scale_biax', 'logit_triax_sens']
    
    def loss_step2(params):
        # K系のみ最適化、β系は固定
        active_params = {k: params[k] for k in k_keys}
        params_temp = params_main.copy()
        params_temp.update(active_params)
        
        # FLC 90% + Binary 10%
        flc_loss = loss_flc_true_jax(params_temp, flc_pts_data, mat_dict)
        bin_loss = loss_fn_jax(params_temp, exps, mat)
        return 0.9 * flc_loss + 0.1 * bin_loss
    
    k_params = {k: params_main[k] for k in k_keys}
    optimizer = optax.adamw(learning_rate=8e-3, weight_decay=1e-3)
    opt_state = optimizer.init(k_params)
    grad_fn = jax.grad(loss_step2)
    
    for step in range(300):
        grads = grad_fn(k_params)
        updates, opt_state = optimizer.update(grads, opt_state, k_params)
        k_params = optax.apply_updates(k_params, updates)
        
        if step % 100 == 0 and verbose:
            loss = loss_step2(k_params)
            print(f"    Step {step}: Mixed Loss = {loss:.6f}")
    
    params_main.update(k_params)
    
    # Step 1.3: V系追加（FLC 70% + Binary 30%）
    if verbose:
        print("\n--- Step 1.3: V系追加 ---")
        print("  最適化: V0, av, ad, chi")
        print("  損失: FLC 70% + Binary 30%")
    
    v_keys = ['log_V0', 'log_av', 'log_ad', 'logit_chi']
    
    def loss_step3(params):
        # V系のみ最適化、β系・K系は固定
        active_params = {k: params[k] for k in v_keys}
        params_temp = params_main.copy()
        params_temp.update(active_params)
        
        # FLC 70% + Binary 30%
        flc_loss = loss_flc_true_jax(params_temp, flc_pts_data, mat_dict)
        bin_loss = loss_fn_jax(params_temp, exps, mat)
        return 0.7 * flc_loss + 0.3 * bin_loss
    
    # L-BFGS-Bで精密最適化
    def loss_v_numpy(v_array):
        v_dict = {v_keys[i]: jnp.array(v_array[i]) for i in range(len(v_keys))}
        return float(loss_step3(v_dict))
    
    v_init = np.array([float(params_main[k]) for k in v_keys])
    res_v = minimize(loss_v_numpy, v_init, method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-8})
    
    if verbose:
        print(f"    L-BFGS-B: Mixed Loss = {res_v.fun:.6f}, iterations = {res_v.nit}")
    
    for i, key in enumerate(v_keys):
        params_main[key] = jnp.array(res_v.x[i])
    
    # Step 1.4: Lambda_crit調整（FLC 50% + Binary 50%）
    if verbose:
        print("\n--- Step 1.4: Lambda_crit調整 ---")
        print("  最適化: Lambda_crit")
        print("  損失: FLC 50% + Binary 50%")
    
    def loss_step4(lam):
        params_temp = params_main.copy()
        params_temp['Lambda_crit'] = lam
        
        # FLC 50% + Binary 50%
        flc_loss = loss_flc_true_jax(params_temp, flc_pts_data, mat_dict)
        bin_loss = loss_fn_jax(params_temp, exps, mat)
        return 0.5 * flc_loss + 0.5 * bin_loss
    
    # 勾配降下で微調整
    lam = params_main['Lambda_crit']
    learning_rate = 2e-3
    
    for step in range(50):
        grad = jax.grad(loss_step4)(lam)
        lam = lam - learning_rate * grad
        lam = jnp.clip(lam, 0.9, 1.1)
        
        if step % 25 == 0 and verbose:
            loss = loss_step4(lam)
            print(f"    Step {step}: Mixed Loss = {loss:.6f}, Lambda_crit = {float(lam):.4f}")
    
    params_main['Lambda_crit'] = lam
    
    # Step 1.5: 全体微調整（制約付きBinary最適化）
    if verbose:
        print("\n--- Step 1.5: FLC Shaping + 制約付きBinary最適化 ---")
        print("  Step 0-1000: FLC最適化（目標: < 0.025）")
        print("  Step 1000-1500: Binary最適化（制約: FLC < 0.026, 3%許容）")
    
    # 制約付き損失関数（動的重み付け版）
    def loss_constrained(params, flc_pts_data, exps, mat_dict, flc_target):
        flc_loss = loss_flc_true_jax(params, flc_pts_data, mat_dict)
        bin_loss = loss_fn_jax(params, exps, mat)
        
        # FLC閾値（3%許容）
        flc_threshold = flc_target * 1.03
        
        # FLCが閾値からどれだけ離れているか
        flc_margin = (flc_loss - flc_threshold) / (flc_threshold * 0.01)
        
        # sigmoid的に重みを切り替え
        # flc_margin < 0（閾値内）→ w_flc ≈ 0.1（Binary重視）
        # flc_margin > 0（閾値超）→ w_flc ≈ 0.9（FLC重視）
        w_transition = jax.nn.sigmoid(flc_margin * 5.0)
        w_flc = 0.1 + 0.8 * w_transition
        w_bin = 1.0 - w_flc
        
        return w_flc * flc_loss + w_bin * bin_loss, flc_loss, bin_loss, w_flc, w_bin
    
    # warmup付きcosineスケジューラー
    schedule_flc = optax.warmup_cosine_decay_schedule(
        init_value=2e-3,
        peak_value=5e-3,
        warmup_steps=200,
        decay_steps=1300,
        end_value=5e-4
    )
    
    optimizer_final = optax.chain(
        optax.clip_by_global_norm(0.3),
        optax.adamw(learning_rate=schedule_flc, weight_decay=5e-5)
    )
    opt_state_final = optimizer_final.init(params_main)
    grad_fn_flc = jax.grad(loss_flc_true_jax)
    
    # FLC目標値（Phase 1.5A後に設定）
    flc_target_value = None
    
    for step in range(1500):
        # Step 0～1000: FLC最適化
        if step < 1000:
            grads = grad_fn_flc(params_main, flc_pts_data, mat_dict)
        # Step 1000～1500: 制約付きBinary最適化
        else:
            # Step 1000でFLC目標値を確定
            if flc_target_value is None:
                flc_target_value = float(loss_flc_true_jax(params_main, flc_pts_data, mat_dict))
                if verbose:
                    print(f"\n    >>> FLC目標値確定: {flc_target_value:.6f}")
                    print(f"    >>> FLC閾値（3%許容）: {flc_target_value * 1.03:.6f}")
                    print(f"    >>> 制約付きBinary最適化開始！\n")
            
            # 制約付き損失関数で勾配計算
            grad_fn_constrained = jax.grad(
                lambda p: loss_constrained(p, flc_pts_data, exps, mat_dict, flc_target_value)[0]
            )
            grads = grad_fn_constrained(params_main)
        
        updates, opt_state_final = optimizer_final.update(grads, opt_state_final, params_main)
        params_main = optax.apply_updates(params_main, updates)
        
        if step % 300 == 0 and verbose:
            flc_loss = loss_flc_true_jax(params_main, flc_pts_data, mat_dict)
            bin_loss = loss_fn_jax(params_main, exps, mat)
            
            if step >= 1000:
                # 制約付き損失を計算
                total_loss, _, _, w_flc, w_bin = loss_constrained(
                    params_main, flc_pts_data, exps, mat_dict, flc_target_value
                )
                flc_violation = max(0.0, float(flc_loss - flc_target_value * 1.03))
                
                print(f"    Step {step:4d}: Total = {float(total_loss):.6f} "
                      f"(FLC: {flc_loss:.6f}, Binary: {bin_loss:.6f})")
                print(f"              重み(FLC: {float(w_flc):.2f}, Binary: {float(w_bin):.2f}), "
                      f"制約違反: {flc_violation:.6f}")
            else:
                print(f"    Step {step:4d}: FLC Loss = {flc_loss:.6f} "
                      f"(Binary参考: {bin_loss:.6f})")
    
    # 最終結果
    edr_dict_final = transform_params_jax(params_main)
    edr_final = edr_dict_to_dataclass(edr_dict_final)
    
    if verbose:
        # 最終損失を計算
        final_flc_loss = loss_flc_true_jax(params_main, flc_pts_data, mat_dict)
        final_bin_loss = loss_fn_jax(params_main, exps, mat)
        final_loss = 0.9 * final_flc_loss + 0.1 * final_bin_loss
        
        print(f"\n  Phase 1完了: 最終Loss = {final_loss:.6f}")
        print(f"    FLC Loss = {final_flc_loss:.6f}")
        print(f"    Binary Loss = {final_bin_loss:.6f}")
    
    # ===========================
    # 最終検証
    # ===========================
    if verbose:
        print("\n" + "="*60)
        print(" 最適化完了！")
        print("="*60)
        
        # Final Validation
        print("\n=== Final Validation ===")
        correct = 0
        
        edr_dict_for_validation = transform_params_jax(params_main)
        
        for i, exp in enumerate(exps):
            schedule_dict = schedule_to_jax_dict(exp.schedule)
            res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict_for_validation)
            Lambda_smooth = smooth_signal_jax(res["Lambda"], window_size=11)
            peak = float(jnp.max(Lambda_smooth))
            D_end = float(res["Damage"][-1])
            
            Dcrit = 0.01
            if exp.failed == 1:
                passed = (peak > edr_final.Lambda_crit and D_end > Dcrit)
            else:
                passed = (peak < edr_final.Lambda_crit - 0.03)
            
            if passed:
                correct += 1
                status = "✓"
            else:
                status = "✗"
                
            print(f"Exp{i}({exp.label}): Λ_max={peak:.3f}, D={D_end:.4f}, "
                  f"failed={exp.failed}, {status}")
        
        accuracy = correct / len(exps) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        
        final_binary_loss = loss_fn_jax(params_main, exps, mat)
        print(f"Final binary loss: {final_binary_loss:.4f}")
    
    info = {
        'success': True,
        'final_loss': float(final_loss),
        'final_flc_loss': float(final_flc_loss),
        'final_bin_loss': float(final_bin_loss),
        'phase0_loss': float(final_loss_phase0),
        'phase1_loss': float(final_loss),
    }
    
    return edr_final, info

# =============================================================================
# Section 6: ヘルパー関数
# =============================================================================

def predict_FLC_point(path_ratio: float, major_rate: float, duration_max: float,
                     mat: MaterialParams, edr: EDRParams,
                     base_contact: float=1.0, base_mu: float=0.08,
                     base_pN: float=200e6, base_vslip: float=0.02,
                     base_htc: float=8000.0, Tdie: float=293.15,
                     T0: float=293.15) -> Tuple[float, float]:
    """FLC点予測"""
    dt = 1e-3
    N = int(duration_max/dt) + 1
    t = np.linspace(0, duration_max, N)
    epsM = major_rate * t
    epsm = path_ratio * major_rate * t
    
    schedule = PressSchedule(
        t=t, eps_maj=epsM, eps_min=epsm,
        triax=np.full(N, float(triax_from_path_jax(path_ratio))),
        mu=np.full(N, base_mu), pN=np.full(N, base_pN),
        vslip=np.full(N, base_vslip), htc=np.full(N, base_htc),
        Tdie=np.full(N, Tdie), contact=np.full(N, base_contact), T0=T0
    )
    
    # JAX版で実行
    schedule_dict = schedule_to_jax_dict(schedule)
    mat_dict = mat_to_jax_dict(mat)
    edr_dict = {
        'V0': edr.V0, 'av': edr.av, 'ad': edr.ad, 'chi': edr.chi,
        'K_scale': edr.K_scale, 'triax_sens': edr.triax_sens,
        'Lambda_crit': edr.Lambda_crit,
        'K_scale_draw': edr.K_scale_draw,
        'K_scale_plane': edr.K_scale_plane,
        'K_scale_biax': edr.K_scale_biax,
        'beta_A': edr.beta_A, 'beta_bw': edr.beta_bw,
        'beta_A_pos': edr.beta_A_pos
    }
    
    res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
    Lambda = np.array(res["Lambda"])
    
    # スムージング処理
    Lambda_smooth = np.array(smooth_signal_jax(jnp.array(Lambda), window_size=11))
    
    # 限界点を探す
    idx = np.where(Lambda_smooth > edr.Lambda_crit)[0]
    if len(idx) > 0:
        k = idx[0]
        return float(epsM[k]), float(epsm[k])
    else:
        return float(epsM[-1]), float(epsm[-1])

def evaluate_flc_fit(experimental: List[FLCPoint],
                    predicted: List[Tuple[float, float]]) -> float:
    """FLC適合度評価"""
    errors = []
    for exp, pred in zip(experimental, predicted):
        deM = pred[0] - exp.major_limit
        dem = pred[1] - exp.minor_limit
        err = np.sqrt(deM**2 + dem**2)
        errors.append(err)
        print(f"  β={exp.path_ratio:+.1f}: 誤差={err:.4f} (ΔMaj={deM:+.3f}, ΔMin={dem:+.3f})")
    
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    
    print(f"\nFLC適合度評価:")
    print(f"  平均誤差: {mean_err:.4f}")
    print(f"  最大誤差: {max_err:.4f}")
    print(f"  精度評価: ", end="")
    
    if mean_err < 0.05:
        print("✅ 優秀（<5%）")
    elif mean_err < 0.10:
        print("🟡 良好（<10%）")
    elif mean_err < 0.20:
        print("🟠 要改善（<20%）")
    else:
        print("🔴 不良（>20%）")
    
    return mean_err

# =============================================================================
# Section 7: デモデータ生成
# =============================================================================

def generate_demo_experiments() -> List[ExpBinary]:
    """デモ実験データ生成"""
    def mk_schedule(beta, mu_base, mu_jump=False, high_stress=False):
        dt = 1e-3
        T = 0.6
        t = np.arange(0, T+dt, dt)
        
        if high_stress:
            epsM = 0.5 * (t/T)**0.8
        else:
            epsM = 0.35 * (t/T)
        epsm = beta * epsM
        
        mu = np.full_like(t, mu_base)
        if mu_jump:
            j = int(0.25/dt)
            mu[j:] += 0.06
        
        triax_val = float(triax_from_path_jax(beta))
        
        return PressSchedule(
            t=t, eps_maj=epsM, eps_min=epsm,
            triax=np.full_like(t, triax_val), mu=mu,
            pN=np.full_like(t, 250e6 if high_stress else 200e6),
            vslip=np.full_like(t, 0.03), htc=np.full_like(t, 8000.0),
            Tdie=np.full_like(t, 293.15), contact=np.full_like(t, 1.0), T0=293.15
        )
    
    exps = [
        ExpBinary(mk_schedule(-0.5, 0.08, False, False), failed=0, label="safe_draw"),
        ExpBinary(mk_schedule(-0.5, 0.08, True, True), failed=1, label="draw_fail"),
        ExpBinary(mk_schedule(0.0, 0.08, False, False), failed=0, label="safe_plane"),
        ExpBinary(mk_schedule(0.0, 0.08, True, True), failed=1, label="plane_fail"),
        ExpBinary(mk_schedule(0.5, 0.10, False, False), failed=0, label="safe_biax"),
        ExpBinary(mk_schedule(0.5, 0.10, True, True), failed=1, label="biax_fail"),
    ]
    return exps

def generate_demo_flc() -> List[FLCPoint]:
    """デモFLCデータ生成"""
    return [
        FLCPoint(-0.5, 0.35, -0.175, 0.6, 1.0, "draw"),
        FLCPoint(0.0, 0.28, 0.0, 0.6, 1.0, "plane"),
        FLCPoint(0.5, 0.22, 0.11, 0.6, 1.0, "biax"),
    ]

# =============================================================================
# Section 8: メイン実行
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" EDRパラメータフィッティング統合版 v5.2")
    print(" + Operation Marie Antoinette（逆問題データ生成）")
    print(" Inverse-EDR Neural Calibration Engine (IENCE)")
    print("="*80)
    
    # 材料パラメータ
    mat = MaterialParams()
    
    # デモデータ生成
    print("\n[デモデータ生成]")
    exps = generate_demo_experiments()
    flc_data = generate_demo_flc()
    print(f"  実験数: {len(exps)}")
    print(f"  FLC点数: {len(flc_data)}")
    
    # 多段階Hybrid最適化実行
    edr_fit, info = hybrid_staged_optimization(
        exps, flc_data, mat,
        verbose=True
    )
    
    # 結果表示
    print("\n" + "="*60)
    print(" 最終結果")
    print("="*60)
    print(f"\n最終Loss: {info['final_loss']:.6f}")
    print(f"  FLC Loss: {info['final_flc_loss']:.6f}")
    print(f"  Binary Loss: {info['final_bin_loss']:.6f}")
    print(f"Phase0 Loss: {info['phase0_loss']:.6f}")
    print(f"Phase1 Loss: {info['phase1_loss']:.6f}")
    
    print(f"\nEDR Parameters:")
    print(f"  V0: {edr_fit.V0:.2e} Pa")
    print(f"  av: {edr_fit.av:.2e}")
    print(f"  ad: {edr_fit.ad:.2e}")
    print(f"  chi: {edr_fit.chi:.3f}")
    print(f"  K_scale: {edr_fit.K_scale:.3f}")
    print(f"  triax_sens: {edr_fit.triax_sens:.3f}")
    print(f"  Lambda_crit: {edr_fit.Lambda_crit:.3f}")
    print(f"  K_scale_draw: {edr_fit.K_scale_draw:.3f}")
    print(f"  K_scale_plane: {edr_fit.K_scale_plane:.3f}")
    print(f"  K_scale_biax: {edr_fit.K_scale_biax:.3f}")
    print(f"  beta_A: {edr_fit.beta_A:.3f}")
    print(f"  beta_bw: {edr_fit.beta_bw:.3f}")
    print(f"  beta_A_pos: {edr_fit.beta_A_pos:.3f} (非対称)")
    
    # FLC予測
    print("\n[FLC予測]")
    preds = []
    for p in flc_data:
        Em, em = predict_FLC_point(p.path_ratio, p.rate_major, p.duration_max, mat, edr_fit)
        preds.append((Em, em))
        print(f"  β={p.path_ratio:+.1f}: 実測({p.major_limit:.3f}, {p.minor_limit:.3f}) "
              f"→ 予測({Em:.3f}, {em:.3f})")
    
    flc_error = evaluate_flc_fit(flc_data, preds)
    
    # 🎂 Operation Marie Antoinette（オプション）
    if MARIE_ANTOINETTE_AVAILABLE:
        print("\n" + "="*80)
        print(" 🎂 Operation Marie Antoinette: 安全多様体分析")
        print("="*80)
        
        use_marie = input("\n安全多様体を構築しますか？ (y/n): ").lower().strip()
        
        if use_marie == 'y':
            # 材料・EDRパラメータをJAX形式に
            mat_dict = mat_to_jax_dict(mat)
            edr_dict = {
                'V0': edr_fit.V0, 'av': edr_fit.av, 'ad': edr_fit.ad,
                'chi': edr_fit.chi, 'K_scale': edr_fit.K_scale,
                'triax_sens': edr_fit.triax_sens,
                'Lambda_crit': edr_fit.Lambda_crit,
                'K_scale_draw': edr_fit.K_scale_draw,
                'K_scale_plane': edr_fit.K_scale_plane,
                'K_scale_biax': edr_fit.K_scale_biax,
                'beta_A': edr_fit.beta_A, 'beta_bw': edr_fit.beta_bw,
                'beta_A_pos': edr_fit.beta_A_pos
            }
            
            # 安全多様体構築
            safe_manifold = build_safe_manifold(
                mat_dict, edr_dict, simulate_lambda_jax,
                n_beta=15, n_mu=5, n_pN=5,
                verbose=True
            )
            
            # 可視化
            visualize_safe_manifold(safe_manifold, 
                                   output_path='/home/claude/safe_manifold.png')
            
            # 安全スコア分析
            manifold_weights = {
                'tv': 0.1,
                'jump': 0.5,
                'topo': 0.1,
                'l1': 1e-3
            }
            
            analyze_safety_scores(
                exps, mat_dict, edr_dict, safe_manifold,
                simulate_lambda_jax, manifold_weights
            )
            
            print("\n✅ Operation Marie Antoinette完了！")
            print("  安全多様体を使ったバイナリ判定が可能になりました")
    
    print("\n" + "="*80)
    print(" 実行完了！")
    print(" 非対称FLC対応・多段階最適化完成 ✅")
    print(" 🎂 Operation Marie Antoinette統合版")
    print("="*80)
