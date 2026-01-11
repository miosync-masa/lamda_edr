#!/usr/bin/env python3
"""
LifetimePredictor: 寿命予測API
==============================

責務：
  - クリープ寿命予測（Zhurkov則のδ理論版）
  - 疲労サイクル予測（Coffin-Mansonのδ理論版）
  - 応力腐食割れ速度
  - 期待寿命

すべて純粋関数的（状態を変えない）

Author: Tamaki & Masamichi
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .delta_engine import DeltaEngine

# 物理定数
k_B = 1.380649e-23
eV_to_J = 1.602176634e-19


class LifetimePredictor:
    """
    δ理論に基づく寿命予測
    
    ═══════════════════════════════════════════════════════════
    従来の経験則を第一原理から統一
    ═══════════════════════════════════════════════════════════
    
    統一される現象:
      - Zhurkov クリープ寿命 (1965)
      - Coffin-Manson 疲労 (1954)
      - Larson-Miller パラメータ (1952)
      - Arrhenius 熱活性化 (1889)
    
    使い方:
        engine = DeltaEngine(material)
        predictor = LifetimePredictor(engine)
        
        tau = predictor.creep_lifetime(sigma, T)
        N_f = predictor.fatigue_cycles(delta_amp, T, frequency=10)
    """
    
    def __init__(self, engine: 'DeltaEngine'):
        """
        Args:
            engine: DeltaEngine インスタンス
        """
        self.engine = engine
        self.mat = engine.mat
    
    # ========================================
    # 期待寿命（基本）
    # ========================================
    
    def expected_lifetime(self,
                          delta: np.ndarray,
                          T: np.ndarray,
                          Z_eff: np.ndarray = None) -> np.ndarray:
        """
        期待寿命 τ [s]
        
        τ = 1 / rate = (1/ν₀) × exp(E_a / kT)
        
        Args:
            delta: Lindemann比 δ
            T: 温度 [K]
            Z_eff: 有効配位数（Noneならバルク）
        
        Returns:
            lifetime [s]
        """
        rate = self.engine.jump_rate_vec(delta, T, Z_eff)
        rate = np.maximum(rate, 1e-30)  # ゼロ除算防止
        return 1.0 / rate
    
    # ========================================
    # クリープ寿命（Zhurkov則の統一）
    # ========================================
    
    def creep_lifetime(self,
                       sigma: np.ndarray,
                       T: np.ndarray,
                       Z_eff: np.ndarray = None) -> np.ndarray:
        """
        クリープ寿命予測（Zhurkov則のδ理論版）
        
        ═══════════════════════════════════════════════════════════
        従来のZhurkov則との比較
        ═══════════════════════════════════════════════════════════
        
        Zhurkov (経験則):
          τ = τ₀ × exp((U₀ - γσ) / kT)
          - U₀: 活性化エネルギー（定数）
          - γσ: 応力による障壁低下（線形）
        
        δ理論 (第一原理):
          τ = τ₀ × exp(E_a(δ) / kT)
          - E_a = E_bond × (Z/Z_bulk) × (1 - (δ/δ_L)²)
          - δ = δ_thermal + δ_mech
        
        小δでZhurkov線形近似と一致！
        
        Args:
            sigma: 応力 [Pa]
            T: 温度 [K]
            Z_eff: 有効配位数
        
        Returns:
            lifetime [s]
        """
        sigma = np.atleast_1d(sigma).astype(np.float64)
        T = np.atleast_1d(T).astype(np.float64)
        
        delta_total = self.engine.delta_total_vec(sigma, T)
        return self.expected_lifetime(delta_total, T, Z_eff)
    
    def creep_lifetime_with_Kt(self,
                               sigma_macro: np.ndarray,
                               T: np.ndarray,
                               K_t: np.ndarray,
                               Z_eff: np.ndarray = None) -> np.ndarray:
        """
        クリープ寿命（K_t対応版）
        
        局所応力集中を考慮
        """
        sigma_macro = np.atleast_1d(sigma_macro).astype(np.float64)
        T = np.atleast_1d(T).astype(np.float64)
        K_t = np.atleast_1d(K_t).astype(np.float64)
        
        delta_total = self.engine.delta_with_Kt_vec(sigma_macro, T, K_t)
        return self.expected_lifetime(delta_total, T, Z_eff)
    
    # ========================================
    # 疲労寿命（Coffin-Mansonの統一）
    # ========================================
    
    def fatigue_cycles(self,
                       delta_amplitude: np.ndarray,
                       T: np.ndarray,
                       frequency: float = 1.0,
                       Z_eff: np.ndarray = None) -> np.ndarray:
        """
        疲労サイクル数予測（Coffin-Mansonのδ理論版）
        
        ═══════════════════════════════════════════════════════════
        従来のCoffin-Manson則との比較
        ═══════════════════════════════════════════════════════════
        
        Coffin-Manson (経験則):
          N_f = C × (Δε)^(-β)
          - C, β: 材料定数（フィッティング）
        
        δ理論 (第一原理):
          N_f = f × τ(δ_amp)
          - τ: 期待寿命（Arrhenius）
          - f: 周波数
        
        Args:
            delta_amplitude: δの振幅（片振幅）
            T: 温度 [K]
            frequency: 周波数 [Hz]
            Z_eff: 有効配位数
        
        Returns:
            N_f: 破壊までのサイクル数
        """
        delta_amplitude = np.atleast_1d(delta_amplitude).astype(np.float64)
        T = np.atleast_1d(T).astype(np.float64)
        
        tau = self.expected_lifetime(delta_amplitude, T, Z_eff)
        return tau * frequency
    
    def fatigue_cycles_from_stress(self,
                                   sigma_amplitude: np.ndarray,
                                   T: np.ndarray,
                                   frequency: float = 1.0,
                                   Z_eff: np.ndarray = None) -> np.ndarray:
        """
        疲労サイクル数（応力振幅から）
        
        Args:
            sigma_amplitude: 応力振幅 [Pa]
            T: 温度 [K]
            frequency: 周波数 [Hz]
        """
        sigma_amplitude = np.atleast_1d(sigma_amplitude).astype(np.float64)
        T = np.atleast_1d(T).astype(np.float64)
        
        # σ_amp → δ_amp
        delta_th = self.engine.delta_thermal_vec(T)
        delta_mech = self.engine.delta_mechanical_vec(sigma_amplitude, T)
        delta_amplitude = delta_th + delta_mech
        
        return self.fatigue_cycles(delta_amplitude, T, frequency, Z_eff)
    
    # ========================================
    # 応力腐食割れ（SCC）
    # ========================================
    
    def stress_corrosion_rate(self,
                              sigma: np.ndarray,
                              T: np.ndarray,
                              V_reduction: float = 0.0,
                              Z_eff: np.ndarray = None) -> np.ndarray:
        """
        応力腐食割れ速度 [1/s]
        
        腐食環境での E_bond 低下 → δ_L が見かけ上低下 → E_a激減
        
        Args:
            sigma: 応力 [Pa]
            T: 温度 [K]
            V_reduction: 結合エネルギー低下率 (0〜1)
            Z_eff: 有効配位数
        
        Returns:
            rate [1/s]: 腐食進行速度
        """
        sigma = np.atleast_1d(sigma).astype(np.float64)
        T = np.atleast_1d(T).astype(np.float64)
        
        # 実効δ_L低下
        effective_delta_L = self.mat.delta_L * (1.0 - V_reduction)
        
        # δ計算
        delta_total = self.engine.delta_total_vec(sigma, T)
        
        # 実効δ/δ_L
        delta_ratio = np.clip(delta_total / effective_delta_L, 0, 1)
        
        # E_a（低下したE_bondで）
        E_bond_eff = self.engine.E_bond * (1.0 - V_reduction)
        barrier_factor = (1.0 - delta_ratio ** 2)
        
        if Z_eff is None:
            Z_eff = np.full_like(sigma, self.mat.Z_bulk, dtype=np.float64)
        
        E_a = E_bond_eff * (Z_eff / self.mat.Z_bulk) * barrier_factor
        
        # Arrhenius
        kT = k_B * np.maximum(T, 1.0)
        exponent = np.clip(-E_a / kT, -100, 0)
        
        return self.engine.NU_0 * np.exp(exponent)
    
    # ========================================
    # Larson-Miller パラメータ
    # ========================================
    
    def larson_miller_parameter(self,
                                sigma: np.ndarray,
                                T: np.ndarray,
                                C: float = 20.0) -> np.ndarray:
        """
        Larson-Miller パラメータ
        
        P = T × (C + log₁₀(t_r))
        
        Args:
            sigma: 応力 [Pa]
            T: 温度 [K]
            C: 材料定数（典型値: 20）
        
        Returns:
            P: Larson-Miller パラメータ
        """
        t_r = self.creep_lifetime(sigma, T)
        t_r = np.maximum(t_r, 1e-30)  # log(0)防止
        
        P = T * (C + np.log10(t_r / 3600))  # 時間単位
        return P
    
    # ========================================
    # 診断・分析
    # ========================================
    
    def diagnose(self,
                 sigma: float,
                 T: float,
                 Z_eff: float = None) -> dict:
        """
        単一条件の詳細診断
        
        Args:
            sigma: 応力 [Pa]
            T: 温度 [K]
            Z_eff: 有効配位数
        
        Returns:
            dict: 各種パラメータ
        """
        sigma_arr = np.array([sigma])
        T_arr = np.array([T])
        Z_arr = np.array([Z_eff]) if Z_eff else None
        
        delta_th = self.engine.delta_thermal_vec(T_arr)[0]
        delta_mech = self.engine.delta_mechanical_vec(sigma_arr, T_arr)[0]
        delta_total = delta_th + delta_mech
        
        E_a = self.engine.activation_energy_vec(np.array([delta_total]), Z_arr)[0]
        rate = self.engine.jump_rate_vec(np.array([delta_total]), T_arr, Z_arr)[0]
        tau = self.expected_lifetime(np.array([delta_total]), T_arr, Z_arr)[0]
        
        phase = self.engine.determine_phase_vec(np.array([delta_total]))[0]
        phase_name = ['HOOKE', 'NONLINEAR', 'YIELD', 'PLASTIC', 'FAILURE'][phase]
        
        return {
            'sigma': sigma,
            'T': T,
            'delta_thermal': delta_th,
            'delta_mechanical': delta_mech,
            'delta_total': delta_total,
            'delta_ratio': delta_total / self.mat.delta_L,
            'E_a_eV': E_a / eV_to_J,
            'jump_rate': rate,
            'lifetime_s': tau,
            'lifetime_h': tau / 3600,
            'lifetime_days': tau / 86400,
            'phase': phase_name,
        }
    
    def print_diagnosis(self, sigma: float, T: float, Z_eff: float = None):
        """診断結果を出力"""
        d = self.diagnose(sigma, T, Z_eff)
        
        print(f"\n{'='*60}")
        print(f"Lifetime Diagnosis: {self.mat.name}")
        print(f"{'='*60}")
        print(f"  Conditions:")
        print(f"    σ = {d['sigma']/1e6:.1f} MPa")
        print(f"    T = {d['T']:.0f} K")
        print(f"  δ Analysis:")
        print(f"    δ_thermal    = {d['delta_thermal']:.4f}")
        print(f"    δ_mechanical = {d['delta_mechanical']:.4f}")
        print(f"    δ_total      = {d['delta_total']:.4f}")
        print(f"    δ/δ_L        = {d['delta_ratio']:.3f}")
        print(f"  Energetics:")
        print(f"    E_a = {d['E_a_eV']:.2f} eV")
        print(f"    rate = {d['jump_rate']:.2e} /s")
        print(f"  Lifetime:")
        print(f"    τ = {d['lifetime_s']:.2e} s")
        print(f"      = {d['lifetime_h']:.2e} h")
        print(f"      = {d['lifetime_days']:.2e} days")
        print(f"  Phase: {d['phase']}")
        print(f"{'='*60}")


# ========================================
# テスト
# ========================================

if __name__ == "__main__":
    print("LifetimePredictor module loaded successfully!")
    print("Requires DeltaEngine for full functionality.")
