#!/usr/bin/env python3
"""
===============================================================================
エネルギー密度比（EDR）理論 - 制約充足問題（CSP）アプローチ
Version 3.5.4: 白層キャリブレーション版
===============================================================================
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# ===============================
# 1. 実験データ定義
# ===============================

@dataclass
class ExperimentalData:
    """実験データ"""
    cutting_speed: float      # [m/min]
    feed_rate: float          # [mm/rev]
    measured_temperature: float  # [K]
    measured_white_layer: Optional[float]  # [μm]
    measured_shear_angle: float  # [deg]
    
EXPERIMENTAL_DATA = [
    # Komanduri & Hou (2001) - Ti6Al4V正面旋削
    ExperimentalData(200, 0.10, 1223, None, 50.0),
    ExperimentalData(400, 0.10, 1373, None, 50.0),
    ExperimentalData(600, 0.10, 1523, None, 50.0),
    
    # Putz et al. (2015) - 白層データ付き
    ExperimentalData(300, 0.15, 1323, 5.2, 50.0),
    ExperimentalData(500, 0.15, 1453, 8.5, 50.0),
]

# ===============================
# 2. モデル本体（v3.5.4改良版）
# ===============================

class TrueEDRModelCalibratedV354:
    """
    v3.5.4: 白層キャリブレーション版
    - 白層計算の感度向上
    - 熱拡散限界の強化
    """
    
    def __init__(self, material='Ti6Al4V'):
        # 基本的な材料物性
        self.material = material
        self.tau_0 = 880e6  # [Pa]
        self.C = 0.028
        self.Tm = 1933  # [K]
        self.rho = 4430  # [kg/m³]
        self.cp = 526  # [J/(kg·K)]
        self.E_coh_0 = 3.5e10  # [Pa]
        self.S_vib = 0.05
        self.m = 2.0
        
        # β変態
        self.T_beta_transus = 1268.0  # [K]
        self.L_trans = 3.98e8  # [J/m³]
        
        # 熱物性
        self.k = 7.0  # [W/(m·K)]
        self.alpha = self.k / (self.rho * self.cp)
        
        # 基準値
        self.base_delta_s = 5e-5  # [m]
        self.typical_WEL = 3e-6  # [m] ← v3.5.4で調整（5e-6→3e-6）
        self.f_ref = 0.15  # [mm/rev]
        
        # 臨界値
        self.Xi_crit = 0.458
        
        # 物理定数（v3.5.4調整）
        self.v_ref_friction = 1.0  # [m/s]
        self.v_ref_thermal = 300.0  # [m/min]
    
    def calculate_friction_heat_calibrated(self, v_mpm: float, tau: float, 
                                          v_slip: float, fric_gain: float) -> float:
        """
        v3.5.4: 熱拡散限界を強化した摩擦熱計算
        """
        # 低速域：凝着摩擦の増大
        if v_mpm < 300:
            adhesion_factor = 1.0 + (300 - v_mpm) / 300 * 1.5
        elif v_mpm <= 320:
            # 遷移領域
            center_dist = abs(v_mpm - 300) / 20.0
            blend = 0.5 * (1 + np.cos(np.pi * center_dist))
            adhesion_factor = 1.0 + 0.15 * blend
        else:
            adhesion_factor = 1.0
        
        # 高速域：熱伝達限界（v3.5.4で指数調整）
        if v_mpm > 350:
            thermal_transfer_factor = (self.v_ref_thermal / v_mpm) ** 1.2  # 0.9→1.2
            thermal_transfer_factor = max(thermal_transfer_factor, 0.25)  # 0.3→0.25
        else:
            thermal_transfer_factor = 1.0
        
        # 総合摩擦係数
        fric_velocity_factor = adhesion_factor * thermal_transfer_factor
        
        # v_slip依存
        v_slip_scale = (v_slip / self.v_ref_friction) ** 0.5
        
        fric_gain_effective = fric_gain * fric_velocity_factor * v_slip_scale
        
        # 摩擦熱計算
        m_f = 1.0
        dT_fric = fric_gain_effective * (m_f * tau * v_slip) / (self.rho * self.cp)
        
        return dT_fric
    
    def calculate_temperature_with_feed(self, v_mpm: float, f_mm: float, 
                                       phi_deg: float, fric_gain: float,
                                       size_effect_index: float,
                                       Xi_threshold: float,
                                       shear_zone_dynamic_factor: float,
                                       WEL_pred: float = None) -> Tuple[float, float, float, float]:
        """温度計算（収束計算付き）"""
        alpha_deg = 10.0
        phi = np.radians(phi_deg)
        alpha = np.radians(alpha_deg)
        v = v_mpm / 60.0
        
        gamma = np.tan(phi - alpha) + 1.0 / np.tan(phi)
        v_shear = v * np.cos(alpha) / np.cos(phi - alpha)
        
        # delta_s動的スケール
        if WEL_pred is not None and WEL_pred > 0:
            delta_s = self.base_delta_s * (WEL_pred / self.typical_WEL)
        else:
            delta_s = self.base_delta_s
        delta_s = np.clip(delta_s, 1e-6, 100e-6)
        
        gamma_dot = gamma * v_shear / delta_s
        
        # ひずみ速度依存
        if gamma_dot > 1000:
            rate_factor = 1 + self.C * np.log(gamma_dot / 1000)
        else:
            rate_factor = 1.0
        
        # Size Effect
        feed_ratio = f_mm / self.f_ref
        size_effect_initial = feed_ratio ** size_effect_index
        size_effect = size_effect_initial
        
        tau_initial = self.tau_0 * rate_factor * size_effect
        
        # Peclet数
        Pe = v * delta_s / self.alpha
        if Pe < 1:
            beta_eff = 0.90
        elif Pe > 100:
            beta_eff = 0.99
        else:
            beta_eff = 0.90 + 0.09 * np.log10(Pe) / np.log10(100)
        
        W_shear_initial = tau_initial * gamma
        dT_shear_initial = beta_eff * W_shear_initial / (self.rho * self.cp)
        
        v_slip = v * np.cos(alpha)
        dT_fric_initial = self.calculate_friction_heat_calibrated(
            v_mpm, tau_initial, v_slip, fric_gain
        )
        
        T = 293 + dT_shear_initial + dT_fric_initial
        
        # β変態判定
        phase_transition_done = False
        
        # 収束計算
        for iteration in range(10):
            T_old = T
            T_homo = (T - 293) / (self.Tm - 293)
            
            if T > self.T_beta_transus:
                beta_phase_scale = 0.9 * (1 - ((T / self.T_beta_transus) - 1.0)**0.5)
                beta_phase_scale = max(beta_phase_scale, 0.7)
            else:
                beta_phase_scale = 1.0
            
            beta_eff_corrected = beta_eff * beta_phase_scale
            thermal_softening = max(1.0 - T_homo**self.m, 0.1)
            
            # K_simple動的集中
            K_simple = 0.5 * self.rho * v_shear**2 * shear_zone_dynamic_factor
            
            entropy_factor = (T / self.Tm)**1.5
            S = self.S_vib * entropy_factor
            F = self.E_coh_0 - T * S
            V_eff_simple = F * (1.0 - T_homo)**3.0
            V_eff_simple = max(V_eff_simple, 0.01 * self.E_coh_0)
            Xi_ratio_simple = K_simple / V_eff_simple / self.Xi_crit
            
            if Xi_ratio_simple > Xi_threshold:
                Xi_excess = Xi_ratio_simple - Xi_threshold
                min_soft = 0.1 * (1 - np.tanh(Xi_excess / 0.05))  # 0.1→0.05
                min_soft = max(min_soft, 0.01)
            else:
                min_soft = 0.1
            
            thermal_softening = max(thermal_softening, min_soft)
            
            tau = tau_initial * thermal_softening
            W_shear = tau * gamma
            
            dT_shear = beta_eff_corrected * W_shear / (self.rho * self.cp)
            dT_fric = self.calculate_friction_heat_calibrated(
                v_mpm, tau, v_slip, fric_gain
            )
            
            T_new = 293 + dT_shear + dT_fric
            
            if not phase_transition_done and T_old < self.T_beta_transus <= T_new:
                dT_latent = self.L_trans / (self.rho * self.cp)
                T_new -= dT_latent * 0.25
                phase_transition_done = True
            
            if abs(T_new - T) < 0.5:
                break
            T = T_new
        
        return T, beta_eff, size_effect, delta_s
    
    def calculate_white_layer_calibrated(self, T: float, Xi_ratio: float, 
                                        T_threshold: float, q: float,
                                        q_critical: float, v_mpm: float,
                                        amplitude_factor: float) -> float:
        """
        v3.5.4: 感度向上した白層計算
        """
        if Xi_ratio <= self.params.get('Xi_threshold', 0.65):  # デフォルト値も調整
            return 0.0
        
        # Xi寄与（v3.5.4で感度向上）
        Xi_excess = Xi_ratio - self.params['Xi_threshold']
        Xi_contribution = 0.55 * amplitude_factor * np.log1p(8 * Xi_excess)  # 5→8
        
        # 温度寄与
        T_ratio = T / self.Tm
        T_contribution = 0.0
        if T_ratio > T_threshold:
            T_norm = (T_ratio - T_threshold) / (1.0 - T_threshold)
            T_contribution = 0.9 * amplitude_factor * (T_norm ** 0.5)
        
        # 冷却効果
        cooling_factor = 1.0 + 0.5 * np.tanh(2.0 * np.log10(q_critical / q))
        
        # 材料スケール（v3.5.3で削除済み）
        material_scale = 1.0
        
        # 速度スケール
        velocity_scale = self.velocity_dependent_scaling(v_mpm)
        
        WEL_base = Xi_contribution + T_contribution
        WEL_final = WEL_base * cooling_factor * material_scale * velocity_scale
        
        return max(0.0, WEL_final)
    
    def velocity_dependent_scaling(self, v_mpm: float) -> float:
        """速度依存スケーリング"""
        if v_mpm < 300:
            return 1.0
        elif v_mpm < 500:
            return 1.0 + 0.5 * (v_mpm - 300) / 200
        else:
            return 1.6 + 1.0 * (v_mpm - 500) / 100
    
    def predict_with_params(self, v_mpm: float, f_mm: float, 
                           params: Dict, phi_base: float = 50.0) -> Dict:
        """統合予測"""
        self.params = params  # パラメータを保存
        
        fric_gain = params.get('fric_gain', 0.11)
        size_effect_index = params.get('size_effect_index', -0.15)
        Xi_threshold = params.get('Xi_threshold', 0.65)
        
        phi_effective = phi_base
        
        # shear_zone_dynamic_factor計算
        sz_base = params['shear_zone_base']
        sz_vexp = params['shear_zone_v_exp']
        v_ratio = v_mpm / 350.0
        shear_zone_dynamic_factor = sz_base * (v_ratio ** sz_vexp)
        
        # パス1: WEL予測
        T_pass1, beta_eff_pass1, size_effect_pass1, delta_s_pass1 = self.calculate_temperature_with_feed(
            v_mpm, f_mm, phi_effective, fric_gain, size_effect_index, Xi_threshold,
            shear_zone_dynamic_factor, None
        )
        
        # 簡易WEL予測（パス1用）
        phi = np.radians(phi_effective)
        alpha = np.radians(10.0)
        v = v_mpm / 60.0
        v_shear = v * np.cos(alpha) / np.cos(phi - alpha)
        K = 0.5 * self.rho * v_shear**2 * shear_zone_dynamic_factor
        
        T_ratio_pass1 = T_pass1 / self.Tm
        
        entropy_factor = (T_pass1 / self.Tm)**1.5
        S = self.S_vib * entropy_factor
        entropy_term = T_pass1 * S
        F = self.E_coh_0 - entropy_term
        structural_weakening = (1.0 - T_ratio_pass1)**3.0
        V_eff = F * structural_weakening
        
        if T_pass1 > self.T_beta_transus:
            delta_T_phase = T_pass1 - self.T_beta_transus
            phase_weakening_factor = 1.0 - 0.4 * np.tanh(delta_T_phase / 150.0)
            V_eff *= phase_weakening_factor
            phase_active = True
        else:
            phase_weakening_factor = 1.0
            phase_active = False
        
        V_eff = max(V_eff, 0.01 * self.E_coh_0)
        Xi = K / V_eff
        Xi_ratio = Xi / self.Xi_crit
        
        contact_length = f_mm * 1e-3 * 0.5
        t_contact = contact_length / v
        dT_typical = 500.0
        q = dT_typical * np.sqrt(self.alpha / t_contact)
        thermal_conductivity_factor = 0.6
        q *= thermal_conductivity_factor
        velocity_factor = 1.0 + 0.5 * np.log10(max(v / 1.0, 0.1))
        q *= velocity_factor
        q = np.clip(q, 1e4, 1e7)
        if v_mpm > 300:
            q *= (v_mpm / 300.0) ** (-0.65)
        
        WEL_pass1 = self.calculate_white_layer_calibrated(
            T_pass1, Xi_ratio,
            params['T_threshold'],
            q,
            params['q_critical'],
            v_mpm,
            params['amplitude_factor']
        )
        
        # パス2: delta_s更新して再計算
        T_final, beta_eff_final, size_effect_final, delta_s_final = self.calculate_temperature_with_feed(
            v_mpm, f_mm, phi_effective, fric_gain, size_effect_index, Xi_threshold,
            shear_zone_dynamic_factor, WEL_pass1
        )
        
        return {
            'temperature_surface': T_final,
            'white_layer_thickness': WEL_pass1,
            'Xi_ratio': Xi_ratio,
            'phase_active': phase_active,
            'size_effect': size_effect_final
        }

# ===============================
# 3. 制約充足問題（CSP）
# ===============================

class ParamMap:
    """パラメータマッピング"""
    def __init__(self, physics_bounds: Dict):
        self.physics_bounds = physics_bounds
        self.param_names = list(physics_bounds.keys())
        self.n_params = len(self.param_names)
    
    def to_physical(self, z: np.ndarray) -> Dict:
        """正規化空間→物理空間"""
        params = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.physics_bounds[name]
            params[name] = lo + (hi - lo) / (1 + np.exp(-z[i]))
        return params
    
    def to_normalized(self, params: Dict) -> np.ndarray:
        """物理空間→正規化空間"""
        z = np.zeros(self.n_params)
        for i, name in enumerate(self.param_names):
            lo, hi = self.physics_bounds[name]
            val = params[name]
            val = np.clip(val, lo + 1e-10, hi - 1e-10)
            z[i] = np.log((val - lo) / (hi - val))
        return z

def evaluate_constraints_calibrated(z: np.ndarray, pmap: ParamMap, 
                                   experiments: List, model) -> Tuple[float, List]:
    """
    v3.5.4: 白層制約を強化した制約評価
    """
    params = pmap.to_physical(z)
    errors = []
    
    for exp in experiments:
        result = model.predict_with_params(
            exp.cutting_speed, exp.feed_rate, params, exp.measured_shear_angle
        )
        
        # 温度制約
        T_pred = result['temperature_surface']
        T_meas = exp.measured_temperature
        T_error = abs(T_pred - T_meas) / T_meas
        
        eps_T = 0.05  # 維持
        violation_T = max(0, T_error - eps_T)
        errors.append(('T', exp.cutting_speed, exp.feed_rate, T_error, violation_T))
        
        # 白層制約（v3.5.4で強化）
        if exp.measured_white_layer is not None:
            WEL_pred = result['white_layer_thickness']
            WEL_meas = exp.measured_white_layer
            if WEL_meas > 0:
                WEL_error = abs(WEL_pred - WEL_meas) / WEL_meas
                eps_WEL = 0.07  # 0.25→0.15
                violation_WEL = max(0, WEL_error - eps_WEL)
                errors.append(('WEL', exp.cutting_speed, exp.feed_rate, WEL_error, violation_WEL))
    
    total_violation = sum(v for _, _, _, _, v in errors)
    return total_violation, errors

def objective_calibrated(z: np.ndarray, pmap: ParamMap, 
                        experiments: List, model) -> float:
    """
    v3.5.4: 白層重視の目的関数
    """
    # 制約違反
    total_violation, errors = evaluate_constraints_calibrated(z, pmap, experiments, model)
    
    # 正則化項
    reg = 0.01 * np.sum(z**2)
    
    # 白層のペナルティ重み増加
    penalty_weight_T = 30.0
    penalty_weight_WEL = 50.0  # 白層重視
    
    weighted_violation = 0.0
    for error_type, v, f, error, violation in errors:
        if error_type == 'T':
            weighted_violation += penalty_weight_T * violation
        else:  # WEL
            weighted_violation += penalty_weight_WEL * violation
    
    return weighted_violation + reg

# ===============================
# 4. 最適化実行
# ===============================

def optimize_csp_v354():
    """v3.5.4最適化実行"""
    
    # パラメータ範囲（v3.5.4調整版）
    physics_bounds = {
        'shear_zone_base': (1e5, 4.0e5),
        'shear_zone_v_exp': (0.1, 0.75),
        'fric_gain': (0.085, 0.20),
        'size_effect_index': (-0.40, -0.05),
        'Xi_threshold': (0.60, 0.85),
        'T_threshold': (0.5, 0.75),
        'q_critical': (2e6, 4e6),  # 急冷閾値強化
        'amplitude_factor': (0.8, 3.0),
    }
    
    # モデル初期化
    model = TrueEDRModelCalibratedV354('Ti6Al4V')
    pmap = ParamMap(physics_bounds)
    
    # v3.5.3の最適値を初期値に（微調整）
    initial_params = {
        'shear_zone_base': 3.596095e+05,
        'shear_zone_v_exp': 2.160124e-01,
        'fric_gain': 0.11,  # 中間値に
        'size_effect_index': -2.250564e-01,
        'Xi_threshold': 0.65,  # 少し下げる
        'T_threshold': 0.50,  # 少し下げる
        'q_critical': 2.5e6,  # 急冷閾値強化
        'amplitude_factor': 1.5,  # 増やす
    }
    
    x0 = pmap.to_normalized(initial_params)
    
    print("="*70)
    print("CSP v3.5.4 白層キャリブレーション版 最適化開始")
    print("="*70)
    print("初期値（v3.5.3最適値ベース）:")
    for key, val in initial_params.items():
        print(f"  {key:<25}: {val:.6e}")
    print("-"*70)
    
    # 最適化実行
    result = minimize(
        objective_calibrated,
        x0,
        args=(pmap, EXPERIMENTAL_DATA, model),
        method='L-BFGS-B',
        options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 500}
    )
    
    # 結果表示
    print("\n" + "="*70)
    print("最適化結果")
    print("="*70)
    print(f"Success: {result.success}")
    print(f"目的関数値: {result.fun:.6f}")
    
    params_opt = pmap.to_physical(result.x)
    print("\n最適パラメータ:")
    for key, val in params_opt.items():
        print(f"  {key:<25}: {val:.6e}")
    
    # 制約評価
    total_violation, errors = evaluate_constraints_calibrated(
        result.x, pmap, EXPERIMENTAL_DATA, model
    )
    
    print(f"\n制約違反（合計）: {total_violation:.6e}")
    print("\n制約詳細:")
    for error_type, v, f, error, violation in errors:
        status = "✓" if violation < 1e-6 else "✗"
        print(f"  {status} {error_type} @ v={v}, f={f:.2f}: error={error*100:.1f}%, violation={violation:.4f}")
    
    # 検証
    print("\n" + "="*70)
    print("🔮 v3.5.4 白層キャリブレーション版検証")
    print("="*70)
    print(f"{'Reference':<25} {'v':<6} {'f':<6} {'T_meas':<8} {'T_pred':<8} {'T_err%':<8} "
          f"{'WEL_m':<8} {'WEL_p':<8} {'WEL_err%':<10}")
    print("-"*70)
    
    T_errors = []
    WEL_errors = []
    
    for exp in EXPERIMENTAL_DATA:
        result = model.predict_with_params(
            exp.cutting_speed, exp.feed_rate, params_opt, exp.measured_shear_angle
        )
        
        T_pred = result['temperature_surface']
        T_err = abs(T_pred - exp.measured_temperature) / exp.measured_temperature * 100
        T_errors.append(T_err)
        
        ref = "Komanduri & Hou" if exp.measured_white_layer is None else "Putz et al."
        
        if exp.measured_white_layer is not None:
            WEL_pred = result['white_layer_thickness']
            WEL_err = abs(WEL_pred - exp.measured_white_layer) / exp.measured_white_layer * 100
            WEL_errors.append(WEL_err)
            print(f"{ref:<25} {exp.cutting_speed:<6} {exp.feed_rate:<6.2f} "
                  f"{exp.measured_temperature:<8.0f} {T_pred:<8.0f} {T_err:<8.2f} "
                  f"{exp.measured_white_layer:<8.2f} {WEL_pred:<8.2f} {WEL_err:<10.2f}")
        else:
            print(f"{ref:<25} {exp.cutting_speed:<6} {exp.feed_rate:<6.2f} "
                  f"{exp.measured_temperature:<8.0f} {T_pred:<8.0f} {T_err:<8.2f} "
                  f"{'N/A':<8} {'N/A':<8} {'N/A':<10}")
    
    print("\n✨ 温度平均誤差: {:.2f}%".format(np.mean(T_errors)))
    print("✨ 温度最大誤差: {:.2f}%".format(np.max(T_errors)))
    if WEL_errors:
        print("✨ 白層平均誤差: {:.2f}%".format(np.mean(WEL_errors)))
        print("✨ 白層最大誤差: {:.2f}%".format(np.max(WEL_errors)))
    
    print("\n" + "="*70)
    print("v3.5.4 白層キャリブレーション完了！")
    print("="*70)
    
    return params_opt

if __name__ == "__main__":
    params_optimal = optimize_csp_v354()
