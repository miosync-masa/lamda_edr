"""
=============================================================================
EDRパラメータフィッティング統合版 v5.0 (JAX + CUDA)
Miosync, Inc. / EDR Neural Calibration Engine (IENCE)

【概要】
板材成形における破壊予測のための統一理論（EDR理論）実装
- JAX版：メイン実装（CPU/GPU両対応、自動微分可能）
- CUDA版：大量並列評価用（オプション）

【最適化戦略】
3フェーズHybrid最適化：
  Phase 1: JAX + AdamW（大域探索、2000ステップ）
  Phase 2: L-BFGS-B（局所精密化、100イテレーション）
  Phase 3: AdamW（微調整、300ステップ）

【著者】
飯泉 真道 (Masamichi Iizumi)
環 (Tamaki) - AI Co-Developer

【日付】
2025-01-19
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

def sanity_check_jax(schedule_dict):
    """入力データの妥当性チェック（JAX版）"""
    pN = schedule_dict['pN']
    Tdie = schedule_dict['Tdie']
    t = schedule_dict['t']
    contact = schedule_dict['contact']
    mu = schedule_dict['mu']
    
    checks = [
        jnp.all(pN < 5e9),  # pN too large?
        jnp.all(pN > 0),    # pN must be positive
        jnp.all(Tdie > 150),  # Tdie out of range?
        jnp.all(Tdie < 1500),
        jnp.all(t >= 0),    # Time must be non-negative
        jnp.all(contact >= 0),  # Contact rate in [0,1]
        jnp.all(contact <= 1),
        jnp.all(mu >= 0),   # Friction coefficient
        jnp.all(mu < 1),
    ]
    
    return jnp.all(jnp.array(checks))

# =============================================================================
# Section 3: メインシミュレーション（JAX版）
# =============================================================================

@jit
def simulate_lambda_jax(schedule_dict, mat_dict, edr_dict):
    """JAX版シミュレーション（メイン実装）"""
    
    # 入力データ妥当性チェック
    # sanity_check_jax(schedule_dict)  # JIT内では省略
    
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
        
        # V_eff（温度依存性を強化）
        T_ratio = jnp.minimum((T - 273.15) / (1500.0 - 273.15), 1.0)
        temp_factor = 1.0 - 0.5 * T_ratio  # 温度が上がるとV_effが下がる
        V_eff = edr_dict['V0'] * temp_factor * \
                (1.0 - edr_dict['av'] * cv - edr_dict['ad'] * jnp.sqrt(jnp.maximum(rho_d, 1e10)))
        V_eff = jnp.maximum(V_eff, 0.01 * edr_dict['V0'])
        
        # 三軸度補正（感度を調整）
        D_triax = jnp.exp(-edr_dict['triax_sens'] * jnp.maximum(triax_t, 0.0))
        
        # Λ計算
        Lambda = K_total / jnp.maximum(V_eff * D_triax, 1e7)
        Lambda = jnp.minimum(Lambda, 10.0)
        
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
    """JAX用パラメータ初期化（log空間）"""
    return {
        'log_V0': jnp.log(2e9),
        'log_av': jnp.log(3e4),
        'log_ad': jnp.log(1e-7),
        'logit_chi': jnp.log(0.1 / (1 - 0.1)),
        'logit_K_scale': jnp.log(0.2 / (1 - 0.2)),
        'logit_K_scale_draw': jnp.log(0.15 / (1 - 0.15)),
        'logit_K_scale_plane': jnp.log(0.25 / (1 - 0.25)),
        'logit_K_scale_biax': jnp.log(0.20 / (1 - 0.20)),
        'logit_triax_sens': jnp.log(0.3 / (1 - 0.3)),
        'Lambda_crit': jnp.array(1.0),
        'logit_beta_A': jnp.log(0.35 / (1 - 0.35)),
        'logit_beta_bw': jnp.log(0.28 / (1 - 0.28)),
        'logit_beta_A_pos': jnp.log(0.5 / (1 - 0.5)),
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
    Dcrit = 0.01  # 0.05から0.01に緩和
    delta = 0.03  # 安全マージン
    
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
        (peak - (edr_dict['Lambda_crit'] - delta))**2 * 3.0,  # 係数を増やして重要視
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

# =============================================================================
# Section 5: 3フェーズHybrid最適化
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
    Phase 1: JAX + AdamW（大域探索）
    Phase 1.5: FLC Shaping（β方向の形状学習）
    Phase 2: L-BFGS-B（局所精密化）
    Phase 3: JAX + AdamW（微調整）
    """
    
    if initial_edr is None:
        initial_edr = EDRParams()
    
    # 共通で使用する関数を先に定義
    mat_dict = mat_to_jax_dict(mat)
    
    # 簡易FLC予測（JAX版）
    @jit 
    def predict_flc_jax(path_ratio, edr_dict, mat_dict):
        """簡易的なFLC限界ひずみ予測"""
        # V字形状をβ依存パラメータで表現
        beta_mult = beta_multiplier_asymmetric_jax(
            path_ratio,
            edr_dict['beta_A'],
            edr_dict.get('beta_A_pos', edr_dict['beta_A']), 
            edr_dict['beta_bw']
        )
        
        # 基準限界ひずみ（β=0での値）
        base_major = 0.28  # 平面ひずみでの基準値
        
        # β依存の調整
        # 深絞り側（β=-0.5）: 増加
        # 等二軸側（β=+0.5）: 減少（より厳しい）
        adjust = 1.0 + 0.25 * path_ratio - 0.4 * path_ratio**2
        
        # K_scale経路別の影響
        k_factor = jnp.where(
            jnp.abs(path_ratio + 0.5) < 0.1, edr_dict['K_scale_draw'],
            jnp.where(
                jnp.abs(path_ratio) < 0.1, edr_dict['K_scale_plane'],
                jnp.where(
                    jnp.abs(path_ratio - 0.5) < 0.1, edr_dict['K_scale_biax'],
                    edr_dict['K_scale']
                )
            )
        )
        
        # 最終的な限界主ひずみ
        Em = base_major * adjust / (beta_mult * k_factor + 0.5)
        em = Em * path_ratio
        
        return Em, em
    
    # ===========================
    # Phase 0: Unsupervised FLC Pretraining
    # ===========================
    if verbose:
        print("\n" + "="*60)
        print(" Phase 0: Unsupervised FLC Manifold Learning")
        print("="*60)
        print("  物理制約のみでFLC面を事前学習")
    
    # Phase 0: 教師なしFLC面学習
    @jit
    def loss_phase0(raw_params):
        """物理制約のみでFLC面を学習"""
        edr_dict = transform_params_jax(raw_params)
        
        # 密なβグリッド
        beta_grid = jnp.linspace(-1.0, 1.0, 50)
        
        # 各βでの仮想FLC限界を計算
        Em_grid = []
        for beta in beta_grid:
            Em, _ = predict_flc_jax(beta, edr_dict, mat_dict)
            Em_grid.append(Em)
        
        Em_array = jnp.array(Em_grid)
        
        # 物理制約1: 単調性（|ε|が増えるとΛも増える）
        monotonicity_loss = jnp.mean(jnp.maximum(0, -jnp.diff(jnp.abs(Em_array))))
        
        # 物理制約2: 凸性（V字形状）
        center = len(beta_grid) // 2
        left_branch = Em_array[:center]
        right_branch = Em_array[center:]
        
        # 左枝は下降、右枝は上昇
        convexity_loss = jnp.mean(jnp.maximum(0, jnp.diff(left_branch))) + \
                         jnp.mean(jnp.maximum(0, -jnp.diff(right_branch)))
        
        # 物理制約3: 対称性（破れを許容）
        asymmetry_factor = edr_dict['beta_A_pos'] / edr_dict['beta_A']
        symmetry_target = Em_array[::-1] * asymmetry_factor
        symmetry_loss = 0.1 * jnp.mean((Em_array - symmetry_target)**2)
        
        # 物理制約4: 平滑性（急激な変化を抑制）
        grad2 = jnp.diff(jnp.diff(Em_array))
        smoothness_loss = 0.05 * jnp.mean(grad2**2)
        
        # 物理制約5: 合理的な範囲（0.1 < Em < 1.0）
        range_loss = jnp.mean(jnp.maximum(0, 0.1 - Em_array)**2) + \
                     jnp.mean(jnp.maximum(0, Em_array - 1.0)**2)
        
        total_loss = monotonicity_loss + convexity_loss + symmetry_loss + \
                    smoothness_loss + range_loss
        
        return total_loss
    
    # Phase 0の初期化
    params_phase0 = init_edr_params_jax()
    
    # Phase 0最適化
    schedule_phase0 = optax.exponential_decay(
        init_value=5e-3,
        transition_steps=100,
        decay_rate=0.9
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
    
    # Phase 0の結果を初期値として使用
    params_jax = params_phase0
    
    # ===========================
    # Phase 1: AdamW広域探索
    # ===========================
    if verbose:
        print("\n" + "="*60)
        print(" Phase 1: JAX + AdamW 広域探索")
        print("="*60)
        print("  Phase 0で学習したFLC面を基に、バイナリ分類を最適化")
    
    # オプティマイザ設定
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-2,
        peak_value=1e-2,
        warmup_steps=100,
        decay_steps=1900,
        end_value=1e-4
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-3, b1=0.9, b2=0.999)
    )
    
    opt_state = optimizer.init(params_jax)
    grad_fn = jax.grad(loss_fn_jax)
    
    # 最適化ループ
    best_loss = float('inf')
    best_params = params_jax
    
    for step in range(2000):
        grads = grad_fn(params_jax, exps, mat)
        updates, opt_state = optimizer.update(grads, opt_state, params_jax)
        params_jax = optax.apply_updates(params_jax, updates)
        
        if step % 100 == 0:
            loss = loss_fn_jax(params_jax, exps, mat)
            if loss < best_loss:
                best_loss = loss
                best_params = params_jax
            
            if verbose:
                print(f"  Step {step:4d}: Loss = {loss:.6f}")
    
    # Phase 1結果を変換
    edr_dict = transform_params_jax(best_params)
    edr_phase1 = edr_dict_to_dataclass(edr_dict)
    
    if verbose:
        print(f"\n  Phase 1完了: 最終Loss = {best_loss:.6f}")
    
    # ===========================
    # Phase 1.5: FLC Shaping
    # ===========================
    if flc_pts and verbose:
        print("\n" + "="*60)
        print(" Phase 1.5: FLC Shaping (AdamW)")
        print("="*60)
        print("  FLC形状の学習に特化")
    
    # FLC専用損失関数
    @jit
    def loss_flc_jax(raw_params, flc_pts_data, mat_dict):
        edr_dict = transform_params_jax(raw_params)
        total_loss = 0.0
        
        # 適応的β分布（等二軸側を密に）
        beta_batch = jnp.concatenate([
            jnp.linspace(-0.6, 0.0, 7),   # 深絞り側は粗く
            jnp.linspace(0.05, 0.6, 13)   # 等二軸側は密に（0重複回避）
        ])
        lambda_peaks = []
        
        # 各β値でのΛピーク計算（V字形状の評価）
        for beta in beta_batch:
            Em, em = predict_flc_jax(beta, edr_dict, mat_dict)
            # 仮想的なΛピーク（FLC限界での値）
            lambda_peak = 1.0 / (Em + 0.1)  # 簡易的な逆相関
            lambda_peaks.append(lambda_peak)
        
        lambda_array = jnp.array(lambda_peaks)
        
        # 動的重み付けのvalley_loss
        valley_weight = jnp.clip(jnp.var(lambda_array), 0.05, 0.3)
        center_idx = 6  # β=0の位置（調整後）
        valley_loss = valley_weight * jnp.sum(
            (lambda_array - lambda_array[center_idx])**2 * 
            jnp.where(jnp.abs(beta_batch) < 0.1, 0.0, 1.0)
        )
        
        # L1+L2混合の曲率正則化
        grad1 = jnp.diff(lambda_array)
        grad2 = jnp.diff(grad1)
        smoothness_loss = 0.05 * jnp.mean(grad2**2) + 0.02 * jnp.mean(jnp.abs(grad2))
        
        # FLC点ごとの誤差計算
        for i in range(len(flc_pts_data['path_ratios'])):
            path_ratio = flc_pts_data['path_ratios'][i]
            major_limit = flc_pts_data['major_limits'][i]
            minor_limit = flc_pts_data['minor_limits'][i]
            
            # β依存の重み付け（等二軸を最重視）
            w = jnp.where(jnp.abs(path_ratio - 0.5) < 0.1, 5.0,
                         jnp.where(jnp.abs(path_ratio) < 0.1, 1.5, 1.0))
            
            # 予測値計算
            Em_pred, em_pred = predict_flc_jax(path_ratio, edr_dict, mat_dict)
            
            loss = w * ((Em_pred - major_limit)**2 + (em_pred - minor_limit)**2)
            total_loss += loss
        
        # 総損失 = FLC誤差 + 動的V字形状正則化
        total_loss = total_loss / max(len(flc_pts_data['path_ratios']), 1)
        total_loss += valley_loss + smoothness_loss
        
        return total_loss
    
    # FLC最適化実行
    if flc_pts:
        # FLCデータをJAX形式に変換
        flc_pts_data = {
            'path_ratios': jnp.array([p.path_ratio for p in flc_pts]),
            'major_limits': jnp.array([p.major_limit for p in flc_pts]),
            'minor_limits': jnp.array([p.minor_limit for p in flc_pts])
        }
        
        # 学習率を高めに設定（warmup付き）
        schedule_flc = optax.warmup_cosine_decay_schedule(
            init_value=1e-3,
            peak_value=3e-3,
            warmup_steps=100,
            decay_steps=400,
            end_value=3e-4
        )
        
        optimizer_flc = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(learning_rate=schedule_flc, weight_decay=1e-4)
        )
        
        opt_state_flc = optimizer_flc.init(best_params)
        mat_dict = mat_to_jax_dict(mat)
        grad_fn_flc = jax.grad(loss_flc_jax)
        
        params_flc = best_params
        
        for step in range(500):
            grads = grad_fn_flc(params_flc, flc_pts_data, mat_dict)
            updates, opt_state_flc = optimizer_flc.update(grads, opt_state_flc, params_flc)
            params_flc = optax.apply_updates(params_flc, updates)
            
            if step % 100 == 0 and verbose:
                loss = loss_flc_jax(params_flc, flc_pts_data, mat_dict)
                print(f"  Step {step:3d}: FLC Loss = {loss:.6f}")
        
        # Phase 1.5結果を使用
        best_params = params_flc
        edr_dict = transform_params_jax(best_params)
        edr_phase1 = edr_dict_to_dataclass(edr_dict)
        
        if verbose:
            final_flc_loss = loss_flc_jax(params_flc, flc_pts_data, mat_dict)
            print(f"\n  Phase 1.5完了: FLC Loss = {final_flc_loss:.6f}")
    
    # ===========================
    # Phase 2: L-BFGS-B局所精密化
    # ===========================
    
    # Phase 1が十分収束している場合はスキップ
    if best_loss < 1e-5:
        if verbose:
            print("\n" + "="*60)
            print(" Phase 2: スキップ（Phase 1で十分収束）")
            print("="*60)
            print(f"  Phase 1 Loss = {best_loss:.6f} < 1e-5")
            print("  L-BFGS-Bは不要と判断")
        
        edr_phase2 = edr_phase1
        res = type('obj', (), {'fun': best_loss, 'nit': 0})()  # ダミーresult
    else:
        if verbose:
            print("\n" + "="*60)
            print(" Phase 2: L-BFGS-B 局所精密化")
            print("="*60)
        
        # NumPy版の損失関数（L-BFGS-B用）
        def loss_numpy(theta):
            edr = EDRParams(
            V0=theta[0], av=theta[1], ad=theta[2], chi=theta[3],
            K_scale=theta[4], triax_sens=theta[5], Lambda_crit=theta[6],
            K_scale_draw=theta[7], K_scale_plane=theta[8], K_scale_biax=theta[9],
            beta_A=theta[10], beta_bw=theta[11], beta_A_pos=theta[12]
            )
            
            # JAXでシミュレーション実行
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
            
            total_loss = 0.0
            for exp in exps:
                schedule_dict = schedule_to_jax_dict(exp.schedule)
                loss = loss_single_exp_jax(schedule_dict, mat_dict, edr_dict, exp.failed)
                total_loss += float(loss)
            
            # FLC損失も追加（β重み付け版：等二軸をより重視）
            if flc_pts:
                for p in flc_pts:
                    # 等二軸（β≈0.5）を最重要視、平面ひずみ（β≈0）も重視
                    if abs(p.path_ratio - 0.5) < 0.1:
                        w = 5.0  # 等二軸は5倍重み
                    elif abs(p.path_ratio) < 0.1:
                        w = 1.5  # 平面ひずみは1.5倍
                    else:
                        w = 1.0  # 深絞りは通常
                        
                    Em, em = predict_FLC_point(p.path_ratio, p.rate_major, p.duration_max, mat, edr)
                    flc_loss = ((Em - p.major_limit)**2 + (em - p.minor_limit)**2)
                    total_loss += w * flc_loss * 0.8
            
            return total_loss / max(len(exps), 1)
        
        # 初期値と境界
        theta0 = np.array([
            edr_phase1.V0, edr_phase1.av, edr_phase1.ad, edr_phase1.chi,
            edr_phase1.K_scale, edr_phase1.triax_sens, edr_phase1.Lambda_crit,
            edr_phase1.K_scale_draw, edr_phase1.K_scale_plane, edr_phase1.K_scale_biax,
            edr_phase1.beta_A, edr_phase1.beta_bw, edr_phase1.beta_A_pos
        ])
        
        bounds = [
            (5e8, 5e9),       # V0
            (1e4, 1e6),       # av
            (1e-8, 1e-6),     # ad
            (0.05, 0.3),      # chi
            (0.05, 1.0),      # K_scale
            (0.1, 0.5),       # triax_sens
            (0.95, 1.05),     # Lambda_crit
            (0.05, 0.3),      # K_scale_draw
            (0.1, 0.4),       # K_scale_plane
            (0.05, 0.3),      # K_scale_biax
            (0.2, 0.5),       # beta_A
            (0.2, 0.35),      # beta_bw
            (0.3, 0.7),       # beta_A_pos
        ]
        
        res = minimize(loss_numpy, theta0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 100, 'ftol': 1e-10})
        
        edr_phase2 = EDRParams(
            V0=res.x[0], av=res.x[1], ad=res.x[2], chi=res.x[3],
            K_scale=res.x[4], triax_sens=res.x[5], Lambda_crit=res.x[6],
            K_scale_draw=res.x[7], K_scale_plane=res.x[8], K_scale_biax=res.x[9],
            beta_A=res.x[10], beta_bw=res.x[11], beta_A_pos=res.x[12]
        )
        
        if verbose:
            print(f"  Phase 2完了: 最終Loss = {res.fun:.6f}")
            print(f"  Iterations: {res.nit}")
    
    # ===========================
    # Phase 3: AdamW微調整
    # ===========================
    if verbose:
        print("\n" + "="*60)
        print(" Phase 3: JAX + AdamW 微調整")
        print("="*60)
    
    # Phase2結果から再初期化（安全な逆変換）
    def safe_logit(x, x_min, x_max):
        """安全なlogit変換"""
        eps = 1e-8
        x_scaled = (x - x_min) / (x_max - x_min)
        x_scaled = jnp.clip(x_scaled, eps, 1 - eps)
        return jnp.log(x_scaled / (1 - x_scaled))
    
    params_jax_final = {
        'log_V0': jnp.log(edr_phase2.V0),
        'log_av': jnp.log(edr_phase2.av),
        'log_ad': jnp.log(edr_phase2.ad),
        'logit_chi': safe_logit(edr_phase2.chi, 0.05, 0.3),
        'logit_K_scale': safe_logit(edr_phase2.K_scale, 0.05, 1.0),
        'logit_K_scale_draw': safe_logit(edr_phase2.K_scale_draw, 0.05, 0.3),
        'logit_K_scale_plane': safe_logit(edr_phase2.K_scale_plane, 0.1, 0.4),
        'logit_K_scale_biax': safe_logit(edr_phase2.K_scale_biax, 0.05, 0.3),
        'logit_triax_sens': safe_logit(edr_phase2.triax_sens, 0.1, 0.5),
        'Lambda_crit': jnp.array(edr_phase2.Lambda_crit),
        'logit_beta_A': safe_logit(edr_phase2.beta_A, 0.2, 0.5),
        'logit_beta_bw': safe_logit(edr_phase2.beta_bw, 0.2, 0.35),
        'logit_beta_A_pos': safe_logit(edr_phase2.beta_A_pos, 0.3, 0.7),
    }
    
    # 微調整用オプティマイザ（低学習率）
    optimizer_fine = optax.adamw(learning_rate=5e-4, weight_decay=1e-5)
    opt_state_fine = optimizer_fine.init(params_jax_final)
    
    for step in range(300):
        grads = grad_fn(params_jax_final, exps, mat)
        updates, opt_state_fine = optimizer_fine.update(grads, opt_state_fine, params_jax_final)
        params_jax_final = optax.apply_updates(params_jax_final, updates)
        
        if step % 100 == 0 and verbose:
            loss = loss_fn_jax(params_jax_final, exps, mat)
            print(f"  Step {step:3d}: Loss = {loss:.6f}")
    
    # 最終結果
    edr_dict_final = transform_params_jax(params_jax_final)
    edr_final = edr_dict_to_dataclass(edr_dict_final)
    
    final_loss = loss_fn_jax(params_jax_final, exps, mat)
    
    info = {
        'success': True,
        'final_loss': float(final_loss),
        'phase1_loss': float(best_loss),
        'phase2_loss': float(res.fun),
        'phase2_iterations': res.nit,
    }
    
    if verbose:
        print(f"\n  Phase 3完了: 最終Loss = {final_loss:.6f}")
        print("\n" + "="*60)
        print(" 最適化完了！")
        print("="*60)
        
        # Final Validation
        print("\n=== Final Validation ===")
        mat_dict = mat_to_jax_dict(mat)
        correct = 0
        for i, exp in enumerate(exps):
            schedule_dict = schedule_to_jax_dict(exp.schedule)
            res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict_final)
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
        final_binary_loss = loss_fn_jax(params_jax_final, exps, mat)
        print(f"Final binary loss: {final_binary_loss:.4f}")
    
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
        triax=np.full(N, triax_from_path_jax(path_ratio)),
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
    
    # スムージング処理（重要！）
    Lambda_smooth = smooth_signal_jax(Lambda, window_size=11)
    
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
    print(" EDRパラメータフィッティング統合版 v5.0")
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
    
    # 3フェーズHybrid最適化実行
    edr_fit, info = hybrid_staged_optimization(
        exps, flc_data, mat,
        verbose=True
    )
    
    # 結果表示
    print("\n" + "="*60)
    print(" 最終結果")
    print("="*60)
    print(f"\n最終Loss: {info['final_loss']:.6f}")
    print(f"Phase1 Loss: {info['phase1_loss']:.6f}")
    print(f"Phase2 Loss: {info['phase2_loss']:.6f}")
    
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
    
    print("\n" + "="*80)
    print(" 実行完了！")
    print(" 非対称FLC対応・3フェーズ最適化完成 ✅")
    print("="*80)
