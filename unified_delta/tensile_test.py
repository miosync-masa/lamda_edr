#!/usr/bin/env python3
"""
Tensile Test Simulation
=======================

統一δ理論に基づく引張試験シミュレーション

LatticeField + DeltaEngine を組み合わせて：
  - K_t効果（応力集中）
  - カスケード破壊
  - 相転移追跡

Author: Tamaki & Masamichi
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .materials import MaterialGPU
from .lattice_field import LatticeField
from .delta_engine import DeltaEngine, DeformationPhase


@dataclass
class TensileResult:
    """引張試験の1ステップ結果"""
    sigma_app: float          # 印加応力 [Pa]
    delta_thermal: float      # 熱的δ
    delta_mech_max: float     # 機械的δ最大値
    delta_max: float          # 合計δ最大値
    delta_mean: float         # 合計δ平均
    yield_frac: float         # 降伏した割合
    fail_frac: float          # 破壊した割合
    n_failed: int             # 累積破壊サイト数
    phase_counts: Dict[str, int]  # 各相のカウント
    dominant_phase: str       # 支配的な相


class TensileTest:
    """
    統一δ理論に基づく引張試験シミュレータ
    
    K_t 効果 + カスケード破壊で：
      最弱点 (空孔近傍) → FAILURE 開始
      → 隣接へ伝播 → マクロ破壊
    
    使い方:
        test = TensileTest(N=30, material=MaterialGPU.Fe())
        results = test.run(sigma_max=500e6, T=300.0)
        test.plot_results()
    """
    
    PHASE_THRESHOLDS = {
        'HOOKE': 0.01,
        'NONLINEAR': 0.03,
        'YIELD': 0.05,
    }
    
    def __init__(self,
                 N: int = 30,
                 material: MaterialGPU = None,
                 vacancy_fraction: float = 0.02):
        """
        Args:
            N: 格子サイズ（N³）
            material: 材料データ
            vacancy_fraction: 空孔率
        """
        self.material = material or MaterialGPU.Fe()
        self.engine = DeltaEngine(self.material)
        self.field = LatticeField.create(
            N=N,
            Z_bulk=self.material.Z_bulk,
            vacancy_fraction=vacancy_fraction
        )
        self.N = N
        
        # 状態
        self.failed = np.zeros((N, N, N), dtype=bool)
        self.history: List[TensileResult] = []
        
        # K_tパラメータ（後で設定可能）
        self.A = 30.0
        self.r_char = 1.0
    
    def reset(self):
        """状態をリセット"""
        self.failed = np.zeros((self.N, self.N, self.N), dtype=bool)
        self.history = []
    
    def run(self,
            sigma_max: float = 500e6,
            n_steps: int = 50,
            T: float = 300.0,
            A: float = 30.0,
            r_char: float = 1.0,
            verbose: bool = True) -> List[TensileResult]:
        """
        引張試験を実行
        
        Args:
            sigma_max: 最大応力 [Pa]
            n_steps: ステップ数
            T: 温度 [K]
            A: K_t の増幅係数
            r_char: K_t の特性距離
            verbose: 詳細出力
        
        Returns:
            各ステップの TensileResult リスト
        """
        self.reset()
        self.A = A
        self.r_char = r_char
        
        sigma_steps = np.linspace(0, sigma_max, n_steps)
        
        if verbose:
            self._print_header(T, sigma_max)
        
        # 事前計算
        delta_thermal = self.engine.delta_thermal_vec(np.array([T]))[0]
        E_T = self.engine.youngs_modulus_vec(np.array([T]))[0]
        K_t_field = self.field.stress_concentration_factor(A=A, r_char=r_char)
        
        if verbose:
            print(f"δ_thermal(T={T}K) = {delta_thermal:.4f}")
            print(f"E(T) = {E_T/1e9:.1f} GPa")
            print("-" * 80)
            print(f"{'Step':<6} {'σ(MPa)':<10} {'δ_max':<10} {'δ_mean':<10} "
                  f"{'Yield%':<10} {'Fail%':<10} {'Phase':<15}")
            print("-" * 80)
        
        for step, sigma_app in enumerate(sigma_steps):
            result = self._run_step(sigma_app, T, delta_thermal, E_T, K_t_field)
            self.history.append(result)
            
            if verbose:
                print(f"{step:<6} {sigma_app/1e6:<10.1f} {result.delta_max:<10.4f} "
                      f"{result.delta_mean:<10.4f} {result.yield_frac*100:<10.2f} "
                      f"{result.fail_frac*100:<10.2f} {result.dominant_phase:<15}")
            
            # 破断判定
            if result.fail_frac > 0.5:
                if verbose:
                    print(f"\n*** FRACTURE at σ = {sigma_app/1e6:.1f} MPa ***")
                break
        
        return self.history
    
    def _run_step(self,
                  sigma_app: float,
                  T: float,
                  delta_thermal: float,
                  E_T: float,
                  K_t_field: np.ndarray) -> TensileResult:
        """1ステップを実行"""
        N = self.N
        
        # アクティブマスク
        mask = self.field.lattice & ~self.failed
        
        # 1. 局所応力 → δ_mech
        sigma_local = K_t_field * sigma_app
        delta_mech = np.zeros((N, N, N))
        delta_mech[mask] = sigma_local[mask] / E_T
        
        # 2. 合計δ
        delta_total = np.zeros((N, N, N))
        delta_total[mask] = delta_thermal + delta_mech[mask]
        
        # 3. 相判定
        phase_map = self._determine_phases(delta_total, mask)
        
        # 4. 新しく破壊した格子点
        delta_L = self.material.delta_L
        newly_failed = (delta_total >= delta_L) & mask & ~self.failed
        
        if np.any(newly_failed):
            self.failed |= newly_failed
            self._cascade_propagate(newly_failed, K_t_field)
        
        # 5. 統計
        return self._compute_stats(sigma_app, delta_thermal, delta_mech, 
                                   delta_total, phase_map, mask)
    
    def _determine_phases(self, delta_total: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """相判定"""
        N = self.N
        delta_L = self.material.delta_L
        
        phase_map = np.zeros((N, N, N), dtype=int)
        phase_map[mask & (delta_total < 0.01)] = 0  # HOOKE
        phase_map[mask & (delta_total >= 0.01) & (delta_total < 0.03)] = 1  # NONLINEAR
        phase_map[mask & (delta_total >= 0.03) & (delta_total < 0.05)] = 2  # YIELD
        phase_map[mask & (delta_total >= 0.05) & (delta_total < delta_L)] = 3  # PLASTIC
        phase_map[mask & (delta_total >= delta_L)] = 4  # FAILURE
        
        return phase_map
    
    def _cascade_propagate(self, newly_failed: np.ndarray, K_t_field: np.ndarray):
        """
        カスケード: 破壊点の隣接原子のZを減少
        
        LatticeField.propagate() を使用！
        """
        # Z低下を伝播（SpMV！）
        Z_loss = self.field.propagate(newly_failed.astype(np.float32))
        self.field.Z = np.maximum(self.field.Z - Z_loss.astype(np.int32), 1)
    
    def _compute_stats(self,
                       sigma_app: float,
                       delta_thermal: float,
                       delta_mech: np.ndarray,
                       delta_total: np.ndarray,
                       phase_map: np.ndarray,
                       mask: np.ndarray) -> TensileResult:
        """統計を計算"""
        active_mask = self.field.lattice & ~self.failed
        n_active = np.sum(active_mask)
        n_total = np.sum(self.field.lattice)
        
        if n_active > 0:
            delta_active = delta_total[active_mask]
            delta_max = np.max(delta_active)
            delta_mean = np.mean(delta_active)
            
            # 各相のカウント
            phase_names = ['HOOKE', 'NONLINEAR', 'YIELD', 'PLASTIC', 'FAILURE']
            phase_counts = {name: int(np.sum(phase_map[active_mask] == i)) 
                           for i, name in enumerate(phase_names)}
            
            yield_frac = np.sum(delta_active >= 0.05) / n_active
            fail_frac = np.sum(self.failed & self.field.lattice) / n_total
            
            dominant_phase = max(phase_counts, key=phase_counts.get)
        else:
            delta_max = 0
            delta_mean = 0
            phase_counts = {name: 0 for name in ['HOOKE', 'NONLINEAR', 'YIELD', 'PLASTIC', 'FAILURE']}
            yield_frac = 1.0
            fail_frac = 1.0
            dominant_phase = "FAILURE"
        
        return TensileResult(
            sigma_app=sigma_app,
            delta_thermal=delta_thermal,
            delta_mech_max=np.max(delta_mech[mask]) if np.any(mask) else 0,
            delta_max=delta_max,
            delta_mean=delta_mean,
            yield_frac=yield_frac,
            fail_frac=fail_frac,
            n_failed=int(np.sum(self.failed & self.field.lattice)),
            phase_counts=phase_counts,
            dominant_phase=dominant_phase,
        )
    
    def _print_header(self, T: float, sigma_max: float):
        """ヘッダー出力"""
        print("\n" + "=" * 80)
        print("TENSILE TEST (Unified δ-Theory + K_t)")
        print("=" * 80)
        print(f"Material: {self.material.name}")
        print(f"Temperature: {T} K")
        print(f"σ_max: {sigma_max/1e6:.1f} MPa")
        print(f"δ_L: {self.material.delta_L}")
        print(f"K_t params: A={self.A}, r_char={self.r_char}")
        print("-" * 80)
    
    def plot_results(self, save_path: str = None):
        """結果をプロット"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return
        
        if not self.history:
            print("No results to plot")
            return
        
        sigma = [r.sigma_app / 1e6 for r in self.history]
        delta_max = [r.delta_max for r in self.history]
        delta_mean = [r.delta_mean for r in self.history]
        fail_frac = [r.fail_frac * 100 for r in self.history]
        yield_frac = [r.yield_frac * 100 for r in self.history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # δ vs σ
        ax = axes[0, 0]
        ax.plot(sigma, delta_max, 'r-', lw=2, label='δ_max')
        ax.plot(sigma, delta_mean, 'b-', lw=2, label='δ_mean')
        ax.axhline(self.material.delta_L, color='k', ls='--', label=f'δ_L={self.material.delta_L}')
        ax.axhline(0.05, color='orange', ls=':', label='δ_yield=0.05')
        ax.set_xlabel('Stress σ (MPa)')
        ax.set_ylabel('δ')
        ax.legend()
        ax.set_title('Lindemann Parameter vs Stress')
        ax.grid(True, alpha=0.3)
        
        # Fail% vs σ
        ax = axes[0, 1]
        ax.plot(sigma, fail_frac, 'r-', lw=2, label='Failed')
        ax.plot(sigma, yield_frac, 'orange', lw=2, label='Yielded')
        ax.set_xlabel('Stress σ (MPa)')
        ax.set_ylabel('Fraction (%)')
        ax.legend()
        ax.set_title('Failure & Yield Fraction')
        ax.grid(True, alpha=0.3)
        
        # Phase evolution
        ax = axes[1, 0]
        phases = ['HOOKE', 'NONLINEAR', 'YIELD', 'PLASTIC']
        for phase in phases:
            counts = [r.phase_counts.get(phase, 0) for r in self.history]
            ax.plot(sigma, counts, lw=2, label=phase)
        ax.set_xlabel('Stress σ (MPa)')
        ax.set_ylabel('Site count')
        ax.legend()
        ax.set_title('Phase Evolution')
        ax.grid(True, alpha=0.3)
        
        # δ distribution (histogram of final state)
        ax = axes[1, 1]
        final_sigma = sigma[-1]
        ax.axvline(self.material.delta_L, color='r', ls='--', label=f'δ_L')
        ax.axvline(0.05, color='orange', ls=':', label='δ_yield')
        ax.set_xlabel('δ')
        ax.set_ylabel('Count')
        ax.set_title(f'Final state at σ={final_sigma:.0f} MPa')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def summary(self) -> Dict:
        """結果サマリー"""
        if not self.history:
            return {}
        
        # 破断応力を探す
        fracture_sigma = None
        for r in self.history:
            if r.fail_frac > 0.5:
                fracture_sigma = r.sigma_app
                break
        
        # 降伏応力を探す（5%が降伏した点）
        yield_sigma = None
        for r in self.history:
            if r.yield_frac > 0.05:
                yield_sigma = r.sigma_app
                break
        
        return {
            'material': self.material.name,
            'N': self.N,
            'fracture_stress_MPa': fracture_sigma / 1e6 if fracture_sigma else None,
            'yield_stress_MPa': yield_sigma / 1e6 if yield_sigma else None,
            'final_fail_frac': self.history[-1].fail_frac,
            'delta_L': self.material.delta_L,
            'n_steps': len(self.history),
        }


# ========================================
# 便利関数
# ========================================

def quick_tensile_test(material: MaterialGPU = None,
                       sigma_max: float = 500e6,
                       T: float = 300.0,
                       N: int = 30) -> TensileTest:
    """
    クイック引張試験
    
    Usage:
        test = quick_tensile_test(MaterialGPU.Fe(), sigma_max=600e6)
        print(test.summary())
    """
    test = TensileTest(N=N, material=material or MaterialGPU.Fe())
    test.run(sigma_max=sigma_max, T=T)
    return test


# ========================================
# メイン
# ========================================

if __name__ == "__main__":
    print("="*70)
    print("Tensile Test Simulation")
    print("="*70)
    
    # Fe at 300K
    test = TensileTest(N=30, material=MaterialGPU.Fe(), vacancy_fraction=0.02)
    results = test.run(sigma_max=600e6, n_steps=40, T=300.0)
    
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    for k, v in test.summary().items():
        print(f"  {k}: {v}")
    
    # プロット（matplotlib があれば）
    try:
        test.plot_results(save_path='/home/claude/tensile_test_results.png')
    except Exception as e:
        print(f"Plot skipped: {e}")
