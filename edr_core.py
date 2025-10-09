"""
=============================================================================
EDR Core Library v1.0
Energy Dissipation & Recovery理論のコア実装

板材成形における破壊予測のための基盤ライブラリ
- データ構造定義
- 物理計算ヘルパー
- メインシミュレーションエンジン
- パラメータ管理

【特徴】
- JAXによる自動微分対応
- 純粋な物理シミュレーション機能のみ
- フィッティング機能は別モジュール（marie_fitting.py）に分離

Author: 飯泉真道 + 環
Date: 2025-01-19
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional 
import jax
import jax.numpy as jnp
from jax import jit

# JAX確認
try:
    print(f"✓ JAX version: {jax.__version__}")
except ImportError:
    raise ImportError("JAXが必要です: pip install jax jaxlib")

# =============================================================================
# Section 1: データ構造定義
# =============================================================================

@dataclass
class MaterialParams:
    """材料パラメータ（SPCC実測値ベース）"""
    rho: float = 7850.0      # 密度 [kg/m3]
    cp: float = 460.0        # 比熱 [J/kg/K]
    k: float = 45.0          # 熱伝導率 [W/m/K] 
    thickness: float = 0.001  # 板厚 1mm（標準）
    sigma0: float = 300e6    # 初期降伏応力 [Pa]（SPCC実測）
    n: float = 0.22          # 加工硬化指数（SPCC標準）
    m: float = 0.015         # 速度感受指数
    r_value: float = 1.4     # ランクフォード値（深絞り性）

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
    # 非対称パラメータ
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
# Section 2: データ変換関数
# =============================================================================

def schedule_to_jax_dict(schedule: PressSchedule) -> Dict:
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

def mat_to_jax_dict(mat: MaterialParams) -> Dict:
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

# =============================================================================
# Section 3: 物理計算ヘルパー関数
# =============================================================================

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
    padded = jnp.pad(x, (window_size//2, window_size//2), mode='edge')
    smoothed = jnp.convolve(padded, kernel, mode='valid')
    return smoothed[:len(x)]

@jit
def beta_gain_rbf_jax(beta, centers, widths, A_neg, A_pos, base=1.0, A_max=0.8):
    """
    RBFベースの滑らかなβゲイン関数
    環ちゃんスペシャル✨ 多項式の暴発なしで表現力MAX！
    """
    b = jnp.clip(beta, -0.99, 0.99)
    widths = jnp.maximum(widths, 1e-3)  # 安全幅
    
    # RBF計算
    r2 = (b - centers) ** 2 / (2.0 * widths ** 2)
    logits = -r2
    w = jax.nn.softmax(logits)
    
    # 非対称振幅
    A_side = jnp.where(b < 0.0, A_neg, A_pos)
    A_eff = jnp.sum(w * A_side)
    A_eff = jnp.clip(A_eff, 0.0, A_max)
    
    # tanh飽和で安全運転！
    safe = base + jnp.tanh(A_eff) * A_max
    return safe

@jit
def kscale_rbf_barycentric_jax(beta, nodes_b, nodes_K, nodes_bw):
    """
    K_scaleの連続補間（3点問題を解決！）
    """
    b = jnp.clip(beta, -0.99, 0.99)
    bw = jnp.maximum(nodes_bw, 1e-3)
    logits = -((b - nodes_b)**2 / (2*bw**2))
    w = jax.nn.softmax(logits)
    return jnp.sum(w * nodes_K)

@jit
def update_path_memory(m_prev, beta_now, beta_prev, alpha=0.9):
    """パス履歴メモリの更新（環ちゃんのトランザクション理論！）"""
    s = jnp.sign(beta_now - beta_prev)
    m = alpha * m_prev + (1 - alpha) * s
    return jnp.clip(m, -1.0, 1.0)

@jit
def apply_memory_to_kscale(ks, m, k_mem_gain=0.08):
    """履歴による補正"""
    return ks * (1.0 + k_mem_gain * m)
    
# =============================================================================
# Section 4: メインシミュレーションエンジン
# =============================================================================

@jit
def simulate_lambda_jax(schedule_dict, mat_dict, edr_dict):
    """
    JAX版EDRシミュレーション
    
    Args:
        schedule_dict: 時系列データ（JAX dict）
        mat_dict: 材料パラメータ（JAX dict）
        edr_dict: EDRパラメータ（JAX dict）
    
    Returns:
        dict: {"Lambda": Λ(t), "Damage": D(t)}
    """
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
        # キャリーに beta_prev と m_path を追加！
        T, cv, rho_d, ep_eq, h_eff, eps3, beta_hist, beta_prev, m_path = carry
        idx = inputs
        
        # 時刻tの値を取得
        epsM_dot_t = epsM_dot[idx]
        epsm_dot_t = epsm_dot[idx]
        triax_t = triax[idx]
        mu_t = mu[idx]
        pN_t = pN[idx]
        vslip_t = vslip[idx]
        htc_t = htc[idx]
        Tdie_t = Tdie[idx]
        contact_t = contact[idx]
        
        # ===== 相当ひずみ速度 =====
        epdot_eq = equiv_strain_rate_jax(epsM_dot_t, epsm_dot_t)
        
        # ===== 板厚更新 =====
        d_eps3 = -(epsM_dot_t + epsm_dot_t) * dt
        eps3_new = eps3 + d_eps3
        h_eff_new = jnp.maximum(mat_dict['h0'] * jnp.exp(eps3_new), 0.2 * mat_dict['h0'])
        
        # ===== 熱収支 =====
        q_fric = mu_t * pN_t * vslip_t * contact_t
        dTdt = (2.0 * htc_t * (Tdie_t - T) + 2.0 * edr_dict['chi'] * q_fric) / \
               (mat_dict['rho'] * mat_dict['cp'] * h_eff_new)
        dTdt = jnp.clip(dTdt, -1000.0, 1000.0)
        T_new = jnp.clip(T + dTdt * dt, 200.0, 2000.0)
        
        # ===== 欠陥更新 =====
        rho_d_new = step_rho_jax(rho_d, epdot_eq, T, dt)
        cv_new = step_cv_jax(cv, T, rho_d_new, dt)
        
        # ===== K成分計算 =====
        K_th = mat_dict['rho'] * mat_dict['cp'] * jnp.maximum(dTdt, 0.0)
        
        sigma_eq = flow_stress_jax(ep_eq, epdot_eq, mat_dict['sigma0'], 
                                   mat_dict['n'], mat_dict['m'], mat_dict['r_value'], T)
        K_pl = 0.9 * sigma_eq * epdot_eq
        
        mu_eff = mu_effective_jax(mu_t, T, pN_t, vslip_t)
        q_fric_eff = mu_eff * pN_t * vslip_t * contact_t
        K_fr = (2.0 * edr_dict['chi'] * q_fric_eff) / h_eff_new
        
        # ===== β処理と履歴更新 =====
        beta_inst = epsm_dot_t / (epsM_dot_t + 1e-8)
        beta_hist_new = jnp.roll(beta_hist, -1).at[4].set(beta_inst)
        beta_smooth = jnp.mean(beta_hist_new)
        
        # ===== 新機能！K_scale（RBF連続補間）=====
        ks_base = kscale_rbf_barycentric_jax(
            beta_smooth,
            edr_dict['ks_nodes_b'],
            edr_dict['ks_nodes_K'],
            edr_dict['ks_nodes_bw']
        )
        
        # ===== 新機能！パスメモリ更新 =====
        m_path_new = update_path_memory(m_path, beta_smooth, beta_prev, alpha=0.9)
        k_scale_eff = apply_memory_to_kscale(ks_base, m_path_new, edr_dict['k_mem_gain'])
        
        # ===== 新機能！βゲイン（RBF表現）=====
        beta_gain = beta_gain_rbf_jax(
            beta_smooth,
            edr_dict['beta_centers'],
            edr_dict['beta_widths'],
            edr_dict['beta_A_neg'],
            edr_dict['beta_A_pos'],
            base=1.0,
            A_max=0.8
        )
        
        # ===== K_total計算（改良版！）=====
        K_total = k_scale_eff * (K_th + K_pl + K_fr) * beta_gain
        K_total = jnp.maximum(K_total, 0.0)
        K_total = jnp.where(jnp.isnan(K_total), 0.0, K_total)
        
        # ===== V_eff（温度依存性）=====
        T_ratio = jnp.minimum((T - 273.15) / (1500.0 - 273.15), 1.0)
        temp_factor = 1.0 - 0.5 * T_ratio
        V_eff = edr_dict['V0'] * temp_factor * \
                (1.0 - edr_dict['av'] * cv - edr_dict['ad'] * jnp.sqrt(jnp.maximum(rho_d, 1e10)))
        V_eff = jnp.maximum(V_eff, 0.01 * edr_dict['V0'])
        V_eff = jnp.where(jnp.isnan(V_eff), edr_dict['V0'], V_eff)
        
        # ===== 三軸度補正 =====
        D_triax = jnp.exp(-edr_dict['triax_sens'] * jnp.maximum(triax_t, 0.0))
        D_triax = jnp.where(jnp.isnan(D_triax), 1.0, D_triax)
        
        # ===== Λ計算 =====
        Lambda = K_total / jnp.maximum(V_eff * D_triax, 1e7)
        Lambda = jnp.minimum(Lambda, 10.0)
        Lambda = jnp.where(jnp.isnan(Lambda), 0.0, Lambda)
        
        # ===== 相当塑性ひずみ更新 =====
        ep_eq_new = ep_eq + epdot_eq * dt
        
        # 新しいキャリー（beta_smooth と m_path_new を追加！）
        new_carry = (T_new, cv_new, rho_d_new, ep_eq_new, h_eff_new, 
                     eps3_new, beta_hist_new, beta_smooth, m_path_new)
        return new_carry, Lambda
    
    # ===== 初期状態（拡張版）=====
    init_beta_hist = jnp.zeros(5)
    init_beta_prev = 0.0  # 初期β
    init_m_path = 0.0     # 初期パスメモリ
    init_carry = (T0, 1e-7, 1e11, 0.0, mat_dict['h0'], 0.0, 
                  init_beta_hist, init_beta_prev, init_m_path)
    
    # scan実行
    indices = jnp.arange(len(t) - 1)
    _, Lambdas = jax.lax.scan(time_step, init_carry, indices)
    
    # Damage積分
    Damage = jnp.cumsum(jnp.maximum(Lambdas - edr_dict['Lambda_crit'], 0.0) * dt)
    
    return {"Lambda": Lambdas, "Damage": Damage}

# =============================================================================
# Section 5: パラメータ管理（環ちゃん改良版 - RBFベース）
# =============================================================================

def init_edr_params_jax():
    """JAX用パラメータ初期化（RBF表現力強化版）"""
    # 基本パラメータ
    base_params = {
        'log_V0': jnp.log(1.5e9),
        'log_av': jnp.log(4e4),
        'log_ad': jnp.log(1e-7),
        'logit_chi': jnp.log(0.09 / (1 - 0.09)),
        'logit_triax_sens': jnp.log(0.25 / (1 - 0.25)),
        'Lambda_crit': jnp.array(1.05),
    }
    
    # β基底のRBFパラメータ（K=4基底）
    K = 4
    beta_params = {
        'beta_centers': jnp.linspace(-0.6, 0.6, K),  # 基底中心
        'beta_log_widths': jnp.full((K,), jnp.log(0.18)),  # 基底幅
        'beta_logit_A_neg': jnp.full((K,), jnp.log(0.40/(1-0.40))),  # β<0側振幅
        'beta_logit_A_pos': jnp.full((K,), jnp.log(0.45/(1-0.45))),  # β>=0側振幅
    }
    
    # K_scaleのRBFノード（5点で滑らか補間）
    ks_nodes_b = jnp.array([-0.6, -0.3, 0.0, 0.3, 0.6])  # draw, 中間, plane, 中間, biax
    ks_params = {
        'ks_nodes_b': ks_nodes_b,
        'ks_nodes_log_bw': jnp.full((5,), jnp.log(0.22)),  # ノード幅
        'ks_nodes_logit_K': jnp.array([
            jnp.log(0.30/(1-0.30)),  # draw域
            jnp.log(0.26/(1-0.26)),  # 中間1
            jnp.log(0.25/(1-0.25)),  # plane域
            jnp.log(0.22/(1-0.22)),  # 中間2
            jnp.log(0.20/(1-0.20)),  # biax域
        ]),
        'k_mem_gain_logit': jnp.log(0.08/(1-0.08)),  # パス履歴ゲイン
    }
    
    # 全部合体！
    return {**base_params, **beta_params, **ks_params}

def transform_params_jax(raw_params):
    """制約付きパラメータ変換（RBF表現力強化版）"""
    # 基本パラメータ変換
    edr = {
        'V0': jnp.exp(raw_params['log_V0']),
        'av': jnp.exp(raw_params['log_av']),
        'ad': jnp.exp(raw_params['log_ad']),
        'chi': soft_clamp(raw_params['logit_chi'], 0.05, 0.3),
        'triax_sens': soft_clamp(raw_params['logit_triax_sens'], 0.1, 0.5),
        'Lambda_crit': jnp.clip(raw_params['Lambda_crit'], 0.95, 1.10),
    }
    
    # β基底パラメータ変換（RBF）
    edr.update({
        'beta_centers': soft_clamp(raw_params['beta_centers'], -0.9, 0.9),
        'beta_widths': jnp.exp(raw_params['beta_log_widths']),
        'beta_A_neg': soft_clamp(raw_params['beta_logit_A_neg'], 0.0, 0.9),
        'beta_A_pos': soft_clamp(raw_params['beta_logit_A_pos'], 0.0, 0.9),
    })
    
    # K_scaleノードパラメータ変換（RBF）
    edr.update({
        'ks_nodes_b': soft_clamp(raw_params['ks_nodes_b'], -0.9, 0.9),
        'ks_nodes_bw': jnp.exp(raw_params['ks_nodes_log_bw']),
        'ks_nodes_K': soft_clamp(raw_params['ks_nodes_logit_K'], 0.05, 0.6),
        'k_mem_gain': soft_clamp(raw_params['k_mem_gain_logit'], 0.0, 0.2),
    })
    
    # 互換性のため旧パラメータ名もマップ（段階的移行用）
    # ※将来的には削除可能
    edr.update({
        'K_scale': edr['ks_nodes_K'][2],  # plane値を代表値に
        'K_scale_draw': edr['ks_nodes_K'][0],
        'K_scale_plane': edr['ks_nodes_K'][2], 
        'K_scale_biax': edr['ks_nodes_K'][4],
        'beta_A': jnp.mean(edr['beta_A_neg']),  # 平均値で代替
        'beta_bw': jnp.mean(edr['beta_widths']),
        'beta_A_pos': jnp.mean(edr['beta_A_pos']),
    })
    
    return edr

def edr_dict_to_dataclass(edr_dict):
    """dict → EDRParams変換（後方互換性維持）"""
    # 新パラメータがある場合はそれを使用、なければ旧パラメータから推定
    return EDRParams(
        V0=float(jax.device_get(edr_dict['V0'])),
        av=float(jax.device_get(edr_dict['av'])),
        ad=float(jax.device_get(edr_dict['ad'])),
        chi=float(jax.device_get(edr_dict['chi'])),
        K_scale=float(jax.device_get(edr_dict.get('K_scale', 
                                                   edr_dict.get('ks_nodes_K', [0.25])[2] if 'ks_nodes_K' in edr_dict else 0.25))),
        triax_sens=float(jax.device_get(edr_dict['triax_sens'])),
        Lambda_crit=float(jax.device_get(edr_dict['Lambda_crit'])),
        K_scale_draw=float(jax.device_get(edr_dict.get('K_scale_draw',
                                                        edr_dict.get('ks_nodes_K', [0.30])[0] if 'ks_nodes_K' in edr_dict else 0.30))),
        K_scale_plane=float(jax.device_get(edr_dict.get('K_scale_plane',
                                                         edr_dict.get('ks_nodes_K', [0.25])[2] if 'ks_nodes_K' in edr_dict else 0.25))),
        K_scale_biax=float(jax.device_get(edr_dict.get('K_scale_biax',
                                                        edr_dict.get('ks_nodes_K', [0.20])[4] if 'ks_nodes_K' in edr_dict else 0.20))),
        beta_A=float(jax.device_get(edr_dict.get('beta_A',
                                                  jnp.mean(edr_dict.get('beta_A_neg', 0.40)) if 'beta_A_neg' in edr_dict else 0.40))),
        beta_bw=float(jax.device_get(edr_dict.get('beta_bw',
                                                   jnp.mean(edr_dict.get('beta_widths', 0.22)) if 'beta_widths' in edr_dict else 0.22))),
        beta_A_pos=float(jax.device_get(edr_dict.get('beta_A_pos',
                                                      jnp.mean(edr_dict.get('beta_A_pos', 0.45)) if 'beta_A_pos' in edr_dict else 0.45))),
    )
    
# =============================================================================
# Section 6: 基本ヘルパー関数
# =============================================================================

def predict_FLC_point(path_ratio: float, major_rate: float, duration_max: float,
                     mat: MaterialParams, edr: EDRParams,
                     base_contact: float=1.0, base_mu: float=0.08,
                     base_pN: float=200e6, base_vslip: float=0.02,
                     base_htc: float=8000.0, Tdie: float=293.15,
                     T0: float=293.15) -> Tuple[float, float]:
    """FLC限界点予測"""
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
    
    # スムージング
    Lambda_smooth = np.array(smooth_signal_jax(jnp.array(Lambda), window_size=11))
    
    # 限界点を探す
    idx = np.where(Lambda_smooth > edr.Lambda_crit)[0]
    if len(idx) > 0:
        k = idx[0]
        return float(epsM[k]), float(epsm[k])
    else:
        return float(epsM[-1]), float(epsm[-1])

# =============================================================================
# Section 7: デモデータ生成
# =============================================================================

def generate_demo_experiments() -> List[ExpBinary]:
    """実機条件に基づくデモ実験データ"""
    def mk_schedule(beta, mu_base, mu_variation=None, stress_profile="normal"):
        dt = 1e-3
        T = 0.5  # 実際の成形時間
        t = np.arange(0, T+dt, dt)
        
        # ひずみプロファイル（実測ベース）
        if stress_profile == "high":
            epsM = 0.45 * (t/T)**0.9  # 高ひずみ
        elif stress_profile == "medium":
            epsM = 0.35 * (t/T)**1.0  # 中ひずみ
        else:
            epsM = 0.25 * (t/T)**1.1  # 通常
        
        epsm = beta * epsM
        
        # 摩擦係数（実機変動を模擬）
        mu = np.full_like(t, mu_base)
        if mu_variation == "increase":
            # 潤滑切れを模擬
            mu += 0.04 * (t/T)  # 徐々に増加
        elif mu_variation == "jump":
            # 突発的な潤滑不良
            j = int(0.3/dt)
            mu[j:] *= 1.8
        
        triax_val = float(triax_from_path_jax(beta))
        
        # 接触圧（実機データ）
        if abs(beta + 0.5) < 0.1:  # 深絞り
            pN_val = 180e6
        elif abs(beta) < 0.1:  # 平面ひずみ
            pN_val = 220e6
        else:  # 張出し
            pN_val = 250e6
        
        return PressSchedule(
            t=t, eps_maj=epsM, eps_min=epsm,
            triax=np.full_like(t, triax_val), 
            mu=mu,
            pN=np.full_like(t, pN_val),
            vslip=np.full_like(t, 0.05),  # 実機速度
            htc=np.full_like(t, 10000.0),  # 実測値
            Tdie=np.full_like(t, 353.15),  # 80°C温間
            contact=np.full_like(t, 1.0), 
            T0=293.15
        )
    
    # より多様な実験条件
    exps = [
        # 深絞り条件
        ExpBinary(mk_schedule(-0.5, 0.10, None, "normal"), failed=0, label="draw_safe"),
        ExpBinary(mk_schedule(-0.5, 0.12, "jump", "high"), failed=1, label="draw_fail"),
        
        # 平面ひずみ条件
        ExpBinary(mk_schedule(0.0, 0.11, None, "medium"), failed=0, label="plane_safe"),
        ExpBinary(mk_schedule(0.0, 0.13, "increase", "high"), failed=1, label="plane_fail"),
        
        # 張出し条件
        ExpBinary(mk_schedule(0.5, 0.12, None, "normal"), failed=0, label="stretch_safe"),
        ExpBinary(mk_schedule(0.5, 0.15, "jump", "medium"), failed=1, label="stretch_fail"),
        
        # 追加：中間条件
        ExpBinary(mk_schedule(-0.25, 0.11, None, "medium"), failed=0, label="mixed1_safe"),
        ExpBinary(mk_schedule(0.25, 0.13, "increase", "high"), failed=1, label="mixed2_fail"),
    ]
    return exps[:6]  # 6個に制限

def generate_demo_flc() -> List[FLCPoint]:
    """SPCC実測FLCデータ（JIS規格ベース）"""
    return [
        FLCPoint(-0.5, 0.38, -0.19, 0.5, 1.0, "draw"),    # 深絞り
        FLCPoint(-0.25, 0.32, -0.08, 0.5, 1.0, "mixed1"),  # 中間1
        FLCPoint(0.0, 0.25, 0.0, 0.5, 1.0, "plane"),       # 平面ひずみ
        FLCPoint(0.25, 0.23, 0.058, 0.5, 1.0, "mixed2"),   # 中間2
        FLCPoint(0.5, 0.20, 0.10, 0.5, 1.0, "stretch"),    # 等二軸
        FLCPoint(1.0, 0.18, 0.18, 0.5, 1.0, "equibiax"),   # 完全等二軸
    ]

# =============================================================================
# エクスポート
# =============================================================================

__all__ = [
    # データ構造
    'MaterialParams',
    'EDRParams',
    'PressSchedule',
    'ExpBinary',
    'FLCPoint',
    
    # 変換関数
    'schedule_to_jax_dict',
    'mat_to_jax_dict',
    
    # 物理計算ヘルパー
    'soft_clamp',
    'triax_from_path_jax',
    'equiv_strain_rate_jax',
    'flow_stress_jax',
    'smooth_signal_jax',
    
    # メインシミュレーション
    'simulate_lambda_jax',

    # デモデータ作成
    'generate_demo_experiments',
    'generate_demo_flc',
    
    # パラメータ管理
    'init_edr_params_jax',
    'transform_params_jax',
    'edr_dict_to_dataclass',
    
    # 基本ヘルパー
    'predict_FLC_point',
]
