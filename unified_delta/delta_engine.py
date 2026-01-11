#!/usr/bin/env python3
"""
DeltaEngine: δ理論の物理計算コア
=================================

責務：
  - δ計算（thermal, mechanical, total）
  - 弾性定数の温度依存性（Born Collapse含む）
  - Debye-Waller（熱変位）
  - 活性化エネルギー E_a
  - 相判定・融解判定
  - カスケード破壊シミュレーション

純粋な「予測式」（寿命計算など）は LifetimePredictor へ

Author: Tamaki & Masamichi
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .lattice_field import LatticeField
    from .materials import MaterialGPU

# CuPy（なければNumPyにフォールバック）
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

# ========================================
# 物理定数
# ========================================
k_B = 1.380649e-23      # Boltzmann定数 [J/K]
u_kg = 1.66053906660e-27  # 原子質量単位 [kg]
eV_to_J = 1.602176634e-19  # eV → J


class DeformationPhase(Enum):
    """変形相"""
    HOOKE = 0
    NONLINEAR = 1
    YIELD = 2
    PLASTIC = 3
    FAILURE = 4


class DeltaEngine:
    """
    δ理論の物理計算エンジン
    
    ═══════════════════════════════════════════════════════════
    Lindemann則 + Born Collapse + Debye-Waller の統合
    ═══════════════════════════════════════════════════════════
    
    δ = √⟨u²⟩ / r_nn
    
    3つの物理が1つの式に：
      1. Debye-Waller: ⟨u²⟩ ∝ T / G(T)
      2. Born Collapse: G(T) の急降下
      3. Lindemann: δ = δ_L で融解
    
    使い方:
        engine = DeltaEngine(material)
        delta = engine.delta_total_vec(sigma, T)
        E_a = engine.activation_energy_vec(delta)
    """
    
    # 相判定の閾値
    DELTA_HOOKE = 0.01
    DELTA_NONLINEAR = 0.03
    DELTA_YIELD = 0.05
    
    # Z依存融点のスケーリング指数
    ALPHA_MELT = 1.2
    
    # Debye周波数（試行頻度）
    NU_0 = 1e13  # [Hz]
    
    def __init__(self, material: 'MaterialGPU'):
        """
        Args:
            material: MaterialGPU インスタンス
        """
        self.mat = material
        self.M = material.M_amu * u_kg  # 原子質量 [kg]
        
        # 弾性定数
        self.G0 = material.E0 / (2.0 * (1.0 + material.nu))  # 剛性率
        self.K0 = material.E0 / (3.0 * (1.0 - 2.0 * material.nu))  # 体積弾性率
        
        # 結合エネルギー [J]
        self.E_bond = material.E_bond_eV * eV_to_J
    
    # ========================================
    # 熱軟化（Λ³理論）
    # ========================================
    
    def thermal_softening_vec(self, T: np.ndarray) -> np.ndarray:
        """
        熱軟化係数 f_soft(T)
        
        f_soft = exp[-λ_eff × α × ΔT]
        
        Λ³理論の核心：格子膨張による結合弱化
        """
        xp = cp if GPU_AVAILABLE else np
        T = xp.asarray(T)
        
        T_ref = 293.0
        delta_T = xp.maximum(T - T_ref, 0)
        lambda_eff = self.mat.lambda_base * (1.0 + self.mat.kappa * delta_T / 1000.0)
        
        result = xp.exp(-lambda_eff * self.mat.alpha * delta_T)
        
        return cp.asnumpy(result) if GPU_AVAILABLE else result
    
    # ========================================
    # 弾性定数の温度依存性
    # ========================================
    
    def shear_modulus_vec(self, T: np.ndarray) -> np.ndarray:
        """
        温度依存剛性率 G(T)（2レジーム）
        
        ═══════════════════════════════════════════════════════════
        Region 1 (T < 0.9 T_m): Λ³ Thermal Softening
          G(T) = G₀ × exp[-λ_eff × α × ΔT]
        
        Region 2 (T ≥ 0.9 T_m): Born Collapse
          G(T) → G₀ × fG_melt へ急降下
        ═══════════════════════════════════════════════════════════
        """
        xp = cp if GPU_AVAILABLE else np
        T_np = cp.asnumpy(T) if GPU_AVAILABLE and hasattr(T, 'get') else np.asarray(T)
        T_arr = xp.asarray(T_np)
        
        T_ref = 293.0
        T_melt = self.mat.T_melt
        T_born = 0.9 * T_melt
        fG_melt = self.mat.fG  # Born崩壊係数
        
        # Region 1: Thermal Softening
        f_soft = xp.asarray(self.thermal_softening_vec(T_np))
        
        # Region 2: Born Collapse
        G_at_born = float(self.thermal_softening_vec(np.array([T_born]))[0])
        ratio = xp.clip((T_arr - T_born) / (T_melt - T_born), 0, 1)
        f_born = G_at_born - (G_at_born - fG_melt) * ratio
        
        # 結合
        f_eff = xp.where(T_arr < T_born, f_soft, f_born)
        f_eff = xp.where(T_arr <= T_ref, 1.0, f_eff)
        
        G = self.G0 * f_eff
        
        return cp.asnumpy(G) if GPU_AVAILABLE else G
    
    def youngs_modulus_vec(self, T: np.ndarray) -> np.ndarray:
        """ヤング率 E(T)"""
        soft = self.thermal_softening_vec(T)
        return self.mat.E0 * soft
    
    # ========================================
    # Debye-Waller（熱変位）
    # ========================================
    
    def sound_velocities_vec(self, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """音速 v_t（横波）, v_l（縦波）"""
        xp = cp if GPU_AVAILABLE else np
        T_np = cp.asnumpy(T) if GPU_AVAILABLE and hasattr(T, 'get') else np.asarray(T)
        T = xp.asarray(T_np)
        
        G = xp.asarray(self.shear_modulus_vec(T_np))
        K = self.K0 * (1.0 - 0.3 * (T / self.mat.T_melt) ** 2)
        
        # 密度（熱膨張考慮）
        rho = self.mat.rho / (1.0 + self.mat.alpha * (T - 300.0)) ** 3
        
        v_t = xp.sqrt(G / rho)
        v_l = xp.sqrt((K + 4.0 * G / 3.0) / rho)
        
        if GPU_AVAILABLE:
            return cp.asnumpy(v_t), cp.asnumpy(v_l)
        return v_t, v_l
    
    def number_density_vec(self, T: np.ndarray) -> np.ndarray:
        """原子数密度 n(T) [atoms/m³]"""
        xp = cp if GPU_AVAILABLE else np
        T = xp.asarray(T)
        
        a = self.mat.a_300K * (1.0 + self.mat.alpha * (T - 300.0))
        
        if self.mat.structure == 'BCC':
            atoms_per_cell = 2.0
        elif self.mat.structure == 'FCC':
            atoms_per_cell = 4.0
        else:
            atoms_per_cell = 4.0
        
        n = atoms_per_cell / (a ** 3)
        
        return cp.asnumpy(n) if GPU_AVAILABLE else n
    
    def debye_wavevector_vec(self, T: np.ndarray) -> np.ndarray:
        """Debye波数 k_D = (6π²n)^(1/3)"""
        xp = cp if GPU_AVAILABLE else np
        n = xp.asarray(self.number_density_vec(T))
        k_D = (6.0 * np.pi ** 2 * n) ** (1.0 / 3.0)
        return cp.asnumpy(k_D) if GPU_AVAILABLE else k_D
    
    def inverse_omega_squared_vec(self, T: np.ndarray) -> np.ndarray:
        """⟨1/ω²⟩（Debye模型）"""
        xp = cp if GPU_AVAILABLE else np
        
        v_t, v_l = self.sound_velocities_vec(T)
        k_D = self.debye_wavevector_vec(T)
        
        v_t = xp.asarray(v_t)
        v_l = xp.asarray(v_l)
        k_D = xp.asarray(k_D)
        
        inv_omega2 = (1.0 / (3.0 * k_D ** 2)) * (2.0 / v_t ** 2 + 1.0 / v_l ** 2)
        
        return cp.asnumpy(inv_omega2) if GPU_AVAILABLE else inv_omega2
    
    def thermal_displacement_squared_vec(self, T: np.ndarray) -> np.ndarray:
        """
        熱的原子変位の二乗 ⟨u²⟩_thermal
        
        ⟨u²⟩ = (k_B T / M) × ⟨1/ω²⟩
        """
        xp = cp if GPU_AVAILABLE else np
        T = xp.asarray(T)
        T = xp.maximum(T, 1.0)  # ゼロ温度チェック
        
        inv_omega2 = xp.asarray(self.inverse_omega_squared_vec(
            cp.asnumpy(T) if GPU_AVAILABLE else T
        ))
        
        u2_thermal = (k_B * T / self.M) * inv_omega2
        
        return cp.asnumpy(u2_thermal) if GPU_AVAILABLE else u2_thermal
    
    def nearest_neighbor_distance_vec(self, T: np.ndarray) -> np.ndarray:
        """
        最近接原子間距離 r_nn(T)
        
        BCC: r_nn = a√3/2
        FCC: r_nn = a/√2
        """
        xp = cp if GPU_AVAILABLE else np
        T = xp.asarray(T)
        
        a = self.mat.a_300K * (1.0 + self.mat.alpha * (T - 300.0))
        
        if self.mat.structure == 'BCC':
            r_nn = a * np.sqrt(3) / 2
        elif self.mat.structure == 'FCC':
            r_nn = a / np.sqrt(2)
        else:
            r_nn = a / np.sqrt(2)
        
        return cp.asnumpy(r_nn) if GPU_AVAILABLE else r_nn
    
    # ========================================
    # δ計算（コア！）
    # ========================================
    
    def delta_thermal_vec(self, T: np.ndarray) -> np.ndarray:
        """
        熱的Lindemann比 δ_thermal = √⟨u²⟩ / r_nn
        
        Debye-Waller + Born Collapse → Lindemann則
        """
        xp = cp if GPU_AVAILABLE else np
        T_np = cp.asnumpy(T) if GPU_AVAILABLE and hasattr(T, 'get') else np.asarray(T)
        
        u2 = self.thermal_displacement_squared_vec(T_np)
        r_nn = self.nearest_neighbor_distance_vec(T_np)
        
        u2 = xp.asarray(u2)
        r_nn = xp.asarray(r_nn)
        
        delta = xp.sqrt(u2) / r_nn
        
        return cp.asnumpy(delta) if GPU_AVAILABLE else delta
    
    def delta_mechanical_vec(self, sigma: np.ndarray, T: np.ndarray) -> np.ndarray:
        """機械的δ: δ_mech = |σ| / E(T)"""
        E_T = self.youngs_modulus_vec(T)
        return np.abs(sigma) / np.maximum(E_T, 1e6)
    
    def delta_total_vec(self, sigma: np.ndarray, T: np.ndarray) -> np.ndarray:
        """合計δ: δ = δ_thermal + δ_mech"""
        return self.delta_thermal_vec(T) + self.delta_mechanical_vec(sigma, T)
    
    def delta_with_Kt_vec(self,
                          sigma_macro: np.ndarray,
                          T: np.ndarray,
                          K_t: np.ndarray) -> np.ndarray:
        """K_t込みのδ: σ_local = K_t × σ_macro"""
        delta_th = self.delta_thermal_vec(T)
        sigma_local = K_t * sigma_macro
        delta_mech = np.abs(sigma_local) / np.maximum(self.youngs_modulus_vec(T), 1e6)
        return delta_th + delta_mech
    
    # ========================================
    # 活性化エネルギー
    # ========================================
    
    def activation_energy_vec(self,
                              delta: np.ndarray,
                              Z_eff: np.ndarray = None) -> np.ndarray:
        """
        活性化エネルギー E_a [J]
        
        ═══════════════════════════════════════════════════════════
        E_a = E_bond × (Z_eff/Z_bulk) × (1 - (δ/δ_L)²)
        ═══════════════════════════════════════════════════════════
        
        調和近似から導出。Zhurkov則・Arrhenius則と整合。
        """
        if Z_eff is None:
            Z_eff = np.full_like(delta, self.mat.Z_bulk, dtype=np.float64)
        
        delta_ratio = np.clip(delta / self.mat.delta_L, 0, 1)
        barrier_factor = (1.0 - delta_ratio ** 2)
        E_a = self.E_bond * (Z_eff / self.mat.Z_bulk) * barrier_factor
        
        return E_a
    
    def jump_rate_vec(self,
                      delta: np.ndarray,
                      T: np.ndarray,
                      Z_eff: np.ndarray = None) -> np.ndarray:
        """
        熱活性化ジャンプレート [1/s]
        
        rate = ν₀ × exp(-E_a / kT)
        """
        E_a = self.activation_energy_vec(delta, Z_eff)
        kT = k_B * np.maximum(T, 1.0)
        
        exponent = np.clip(-E_a / kT, -100, 0)
        return self.NU_0 * np.exp(exponent)
    
    # ========================================
    # 相判定・融解判定
    # ========================================
    
    def determine_phase_vec(self, delta: np.ndarray) -> np.ndarray:
        """相判定 → 整数配列"""
        phase = np.zeros(len(delta), dtype=np.int32)
        phase[delta >= self.DELTA_HOOKE] = 1      # NONLINEAR
        phase[delta >= self.DELTA_NONLINEAR] = 2  # YIELD
        phase[delta >= self.DELTA_YIELD] = 3      # PLASTIC
        phase[delta >= self.mat.delta_L] = 4      # FAILURE
        return phase
    
    def local_melting_temperature_vec(self, Z_eff: np.ndarray) -> np.ndarray:
        """Z依存融点"""
        Z_ratio = np.clip(Z_eff / self.mat.Z_bulk, 0.1, 1.0)
        return self.mat.T_melt * (Z_ratio ** self.ALPHA_MELT)
    
    def is_molten_vec(self, T: np.ndarray, Z_eff: np.ndarray) -> np.ndarray:
        """融解判定"""
        T_melt_local = self.local_melting_temperature_vec(Z_eff)
        return T > T_melt_local
    
    def stochastic_collapse_mask(self,
                                  delta: np.ndarray,
                                  T: np.ndarray,
                                  Z_eff: np.ndarray = None,
                                  dt: float = 1e-6) -> np.ndarray:
        """
        確率的崩壊マスク（モンテカルロ）
        
        Args:
            delta: 現在のδ
            T: 温度
            Z_eff: 有効配位数
            dt: 時間ステップ [s]
        
        Returns:
            collapse_mask: bool配列
        """
        rate = self.jump_rate_vec(delta, T, Z_eff)
        P_collapse = 1.0 - np.exp(-rate * dt)
        
        random = np.random.random(len(delta))
        deterministic = delta >= self.mat.delta_L
        
        return deterministic | (random < P_collapse)
    
    # ========================================
    # カスケード破壊シミュレーション
    # ========================================
    
    def run_cascade(self,
                    field: 'LatticeField',
                    sigma_macro: float,
                    T_init: float,
                    max_iterations: int = 50,
                    efficiency: float = 0.1,
                    K_t_params: dict = None) -> Dict:
        """
        カスケード破壊シミュレーション
        
        Args:
            field: LatticeField インスタンス
            sigma_macro: マクロ応力 [Pa]
            T_init: 初期温度 [K]
            max_iterations: 最大反復回数
            efficiency: 発熱効率
            K_t_params: K_t計算パラメータ {'A': 30.0, 'r_char': 1.0}
        
        Returns:
            dict: delta, T, history, iterations, collapsed, white_layer_frac, T_max
        """
        K_t_params = K_t_params or {'A': 30.0, 'r_char': 1.0}
        
        # 初期化
        shape = field.config.shape
        T = np.full(shape, T_init)
        
        # 1結合あたりの発熱
        dT_per_bond = self.E_bond / (3 * k_B) * efficiency
        
        # K_t場
        K_t = field.stress_concentration_factor(**K_t_params)
        
        # σ配列
        sigma = np.full(shape, sigma_macro)
        
        # δ計算
        delta = self._compute_delta_field(sigma, T, K_t, field.lattice)
        
        history = [np.sum((delta >= self.mat.delta_L) & field.lattice)]
        
        for it in range(max_iterations):
            # 崩壊マスク
            collapsed = (delta >= self.mat.delta_L) & field.lattice
            n_collapsed = np.sum(collapsed)
            
            if n_collapsed == 0 or n_collapsed == history[-1]:
                break
            
            # 発熱をSpMV伝播
            heat_source = collapsed.astype(np.float32) * dT_per_bond
            heat_received = field.propagate(heat_source)
            
            # 温度更新
            T = T + heat_received
            T = np.clip(T, 0, self.mat.T_melt * 10)
            
            # Z低下（field内部で処理）
            field.mark_failed(collapsed)
            
            # K_t再計算
            K_t = field.stress_concentration_factor(**K_t_params)
            
            # δ再計算
            delta = self._compute_delta_field(sigma, T, K_t, field.lattice)
            
            history.append(np.sum((delta >= self.mat.delta_L) & field.lattice))
        
        # 融解判定
        Z_flat = field.Z[field.lattice].astype(np.float64)
        T_flat = T[field.lattice]
        molten = self.is_molten_vec(T_flat, Z_flat)
        
        return {
            'delta': delta,
            'T': T,
            'history': history,
            'iterations': it + 1,
            'collapsed': history[-1],
            'white_layer_frac': np.mean(molten),
            'T_max': T.max(),
        }
    
    def _compute_delta_field(self,
                             sigma: np.ndarray,
                             T: np.ndarray,
                             K_t: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
        """δ場を計算（内部用）"""
        delta = np.zeros_like(T)
        
        if not np.any(mask):
            return delta
        
        T_flat = T[mask]
        sigma_flat = sigma[mask]
        K_t_flat = K_t[mask]
        
        delta_flat = self.delta_with_Kt_vec(sigma_flat, T_flat, K_t_flat)
        delta[mask] = delta_flat
        
        return delta


# ========================================
# テスト
# ========================================

if __name__ == "__main__":
    # MaterialGPUが必要なので、ここでは簡易テスト
    print("DeltaEngine module loaded successfully!")
    print(f"GPU available: {GPU_AVAILABLE}")
    print(f"Physical constants:")
    print(f"  k_B = {k_B:.6e} J/K")
    print(f"  u_kg = {u_kg:.6e} kg")
    print(f"  eV_to_J = {eV_to_J:.6e} J/eV")
