#!/usr/bin/env python3
"""
TensileTest with Q_Λ Tracking
==============================

引張試験で位相場のトポロジカルチャージ Q_Λ を追跡
リコネクション（結合切断）をトポロジー変化として検出

理論:
  - E = mc² = Vorticity（渦度）
  - α = 0.6 = SO(5)→SO(4) 対称性の破れ
  - δ = 0.6 δ_L で Q がジャンプ開始（Born崩壊）
  - δ = δ_L で Q がランダム化（Lindemann融解）

Author: Tamaki & Masamichi
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/claude')

from unified_delta import MaterialGPU, DeltaEngine, LatticeField


@dataclass
class TopologyResult:
    """トポロジー追跡結果"""
    step: int
    sigma: float           # 応力 [Pa]
    delta_mean: float      # 平均δ
    delta_max: float       # 最大δ
    Q_total: float         # 総トポロジカルチャージ
    Q_std: float           # Q の標準偏差
    unstable_fraction: float  # 不安定サイトの割合
    reconnection: bool     # リコネクション発生？
    dQ: float             # Q の変化


class TensileTestWithTopology:
    """
    引張試験 + トポロジー追跡
    
    Usage:
        test = TensileTestWithTopology(material=MaterialGPU.Fe(), N=30)
        results = test.run(sigma_max=800e6, n_steps=50)
        test.plot_results()
    """
    
    def __init__(self,
                 material: MaterialGPU = None,
                 N: int = 30,
                 vacancy_fraction: float = 0.02,
                 T: float = 600.0,
                 seed: int = 42):
        """
        Args:
            material: 材料パラメータ
            N: 格子サイズ
            vacancy_fraction: 空孔率
            T: 温度 [K]
            seed: 乱数シード
        """
        self.material = material or MaterialGPU.Fe()
        self.T = T
        self.seed = seed
        
        # エンジン
        self.engine = DeltaEngine(self.material)
        
        # 格子場
        self.field = LatticeField.create(
            N=N, 
            Z_bulk=self.material.Z_bulk,
            vacancy_fraction=vacancy_fraction,
            seed=seed
        )
        
        # 位相場初期化
        self.field.init_phase_field(T=T, seed=seed)
        
        # K_t場
        self.K_t_data = self.field.get_flat_arrays(A=30.0, r_char=1.0)
        
        # 結果格納
        self.results: List[TopologyResult] = []
        self.reconnection_events: List[dict] = []
        
        print(f"TensileTestWithTopology initialized:")
        print(f"  Material: {self.material.name}")
        print(f"  T = {T} K")
        print(f"  δ_L = {self.material.delta_L}")
        print(f"  Lattice: {N}³, vacancies: {vacancy_fraction*100:.1f}%")
    
    def run(self, 
            sigma_max: float = 600e6,
            n_steps: int = 50,
            phase_update_steps: int = 5) -> List[TopologyResult]:
        """
        引張試験を実行
        
        Args:
            sigma_max: 最大応力 [Pa]
            n_steps: ステップ数
            phase_update_steps: 各σでの位相更新回数
            
        Returns:
            結果リスト
        """
        print(f"\n--- Running tensile test ---")
        print(f"  σ: 0 → {sigma_max/1e6:.0f} MPa in {n_steps} steps")
        
        delta_L = self.material.delta_L
        shape = self.field.config.shape
        
        # 応力シーケンス
        sigma_seq = np.linspace(0, sigma_max, n_steps)
        
        # K_t を3D形状に戻す
        K_t_3d = np.ones(shape, dtype=np.float32)
        flat_indices = np.where(self.field.lattice.ravel())[0]
        K_t_flat = self.K_t_data['K_t']
        for i, idx in enumerate(flat_indices):
            ix = idx // (shape[1] * shape[2])
            iy = (idx % (shape[1] * shape[2])) // shape[2]
            iz = idx % shape[2]
            if i < len(K_t_flat):
                K_t_3d[ix, iy, iz] = K_t_flat[i]
        
        for step, sigma in enumerate(sigma_seq):
            # δ計算
            T_arr = np.array([self.T])
            delta_thermal = self.engine.delta_thermal_vec(T_arr)[0]
            E_T = self.engine.youngs_modulus_vec(T_arr)[0]
            
            # δ_mech = K_t × σ / E(T)
            delta_mech_3d = K_t_3d * sigma / E_T
            delta_3d = delta_thermal + delta_mech_3d
            
            # 位相場を更新
            for _ in range(phase_update_steps):
                self.field.update_phase_field(delta_3d, delta_L, self.T, dt=0.02)
            
            # リコネクション検出
            recon_result = self.field.detect_reconnection(Q_threshold=0.3)
            Q_stats = self.field.compute_Q_statistics()
            
            # 結果記録
            result = TopologyResult(
                step=step,
                sigma=sigma,
                delta_mean=float(np.mean(delta_3d[self.field.lattice])),
                delta_max=float(np.max(delta_3d)),
                Q_total=Q_stats['Q_total'],
                Q_std=Q_stats['Q_std'],
                unstable_fraction=Q_stats['unstable_fraction'],
                reconnection=recon_result['reconnection_detected'],
                dQ=recon_result['dQ'],
            )
            self.results.append(result)
            
            # リコネクションイベント
            if result.reconnection:
                self.reconnection_events.append({
                    'step': step,
                    'sigma': sigma,
                    'dQ': result.dQ,
                    'delta_mean': result.delta_mean,
                    'sites': recon_result.get('sites', []),
                })
            
            # 進捗表示
            if step % 10 == 0 or result.reconnection:
                status = "⚡RECON!" if result.reconnection else ""
                print(f"  Step {step:3d}: σ={sigma/1e6:6.1f} MPa, "
                      f"δ={result.delta_mean:.4f} ({result.delta_mean/delta_L*100:5.1f}% of δ_L), "
                      f"Q={result.Q_total:+8.3f}, dQ={result.dQ:+.3f} {status}")
        
        print(f"\n--- Test completed ---")
        print(f"  Total reconnection events: {len(self.reconnection_events)}")
        
        return self.results
    
    def analyze_critical_point(self) -> dict:
        """
        臨界点（Born崩壊点）を解析
        
        理論予測: δ/δ_L ≈ 0.6 でリコネクション開始
        """
        delta_L = self.material.delta_L
        
        # δ/δ_L vs Q_total をプロット用にまとめる
        delta_ratios = [r.delta_mean / delta_L for r in self.results]
        Q_values = [r.Q_total for r in self.results]
        
        # Q の変動が大きくなる点を探す
        Q_changes = [abs(r.dQ) for r in self.results]
        
        # 閾値を超える最初の点
        threshold = 0.1
        critical_idx = None
        for i, dQ in enumerate(Q_changes):
            if dQ > threshold:
                critical_idx = i
                break
        
        result = {
            'predicted_critical_ratio': 0.6,  # α = 6/10
            'delta_L': delta_L,
        }
        
        if critical_idx is not None:
            result['observed_critical_ratio'] = delta_ratios[critical_idx]
            result['critical_sigma'] = self.results[critical_idx].sigma
            result['match'] = abs(delta_ratios[critical_idx] - 0.6) < 0.15
        else:
            result['observed_critical_ratio'] = None
            result['critical_sigma'] = None
            result['match'] = False
        
        return result
    
    def plot_results(self, save_path: str = None):
        """結果をプロット"""
        if not self.results:
            print("No results to plot. Run test first.")
            return
        
        delta_L = self.material.delta_L
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # データ抽出
        sigmas = [r.sigma / 1e6 for r in self.results]  # MPa
        delta_means = [r.delta_mean for r in self.results]
        delta_ratios = [r.delta_mean / delta_L for r in self.results]
        Q_totals = [r.Q_total for r in self.results]
        Q_stds = [r.Q_std for r in self.results]
        dQs = [r.dQ for r in self.results]
        unstable_fracs = [r.unstable_fraction * 100 for r in self.results]
        
        # リコネクション点をマーク
        recon_sigmas = [e['sigma'] / 1e6 for e in self.reconnection_events]
        recon_Q = [self.results[e['step']].Q_total for e in self.reconnection_events]
        
        # === Plot 1: δ/δ_L vs σ ===
        ax1 = axes[0, 0]
        ax1.plot(sigmas, delta_ratios, 'b-', linewidth=2, label='δ/δ_L')
        ax1.axhline(y=0.6, color='r', linestyle='--', linewidth=1.5, 
                    label='Born崩壊点 (α=0.6)')
        ax1.axhline(y=1.0, color='orange', linestyle='--', linewidth=1.5,
                    label='Lindemann (δ=δ_L)')
        ax1.set_xlabel('Stress σ [MPa]')
        ax1.set_ylabel('δ / δ_L')
        ax1.set_title(f'{self.material.name}: Lindemann Ratio vs Stress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(1.2, max(delta_ratios) * 1.1))
        
        # === Plot 2: Q_total vs σ ===
        ax2 = axes[0, 1]
        ax2.plot(sigmas, Q_totals, 'g-', linewidth=2, label='Q_Λ total')
        if recon_sigmas:
            ax2.scatter(recon_sigmas, recon_Q, c='red', s=100, marker='*',
                        zorder=5, label='Reconnection')
        ax2.set_xlabel('Stress σ [MPa]')
        ax2.set_ylabel('Q_Λ (total)')
        ax2.set_title('Topological Charge vs Stress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # === Plot 3: dQ vs δ/δ_L ===
        ax3 = axes[1, 0]
        ax3.plot(delta_ratios, dQs, 'mo-', linewidth=1, markersize=3)
        ax3.axvline(x=0.6, color='r', linestyle='--', linewidth=1.5,
                    label='Born崩壊点 (α=0.6)')
        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('δ / δ_L')
        ax3.set_ylabel('ΔQ (per step)')
        ax3.set_title('Topological Change Rate vs Lindemann Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # === Plot 4: Unstable fraction vs δ/δ_L ===
        ax4 = axes[1, 1]
        ax4.plot(delta_ratios, unstable_fracs, 'c-', linewidth=2)
        ax4.axvline(x=0.6, color='r', linestyle='--', linewidth=1.5,
                    label='Born崩壊点 (α=0.6)')
        ax4.set_xlabel('δ / δ_L')
        ax4.set_ylabel('Unstable sites [%]')
        ax4.set_title('Unstable Sites (|Q - round(Q)| > 0.25)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Tensile Test with Topology Tracking\n'
                     f'{self.material.name}, T={self.T}K, δ_L={delta_L}',
                     fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
        # 臨界点解析
        critical = self.analyze_critical_point()
        print("\n" + "="*60)
        print("Critical Point Analysis")
        print("="*60)
        print(f"  Predicted: δ/δ_L = 0.6 (α = 6/10, SO(5)→SO(4))")
        if critical['observed_critical_ratio']:
            print(f"  Observed:  δ/δ_L = {critical['observed_critical_ratio']:.3f}")
            print(f"  Critical σ: {critical['critical_sigma']/1e6:.1f} MPa")
            match_str = "✓ MATCH!" if critical['match'] else "△ differs"
            print(f"  {match_str}")
        else:
            print(f"  No reconnection detected in this test")


def main():
    """メインテスト"""
    print("="*70)
    print("Tensile Test with Q_Λ Topology Tracking")
    print("="*70)
    print()
    print("Theory:")
    print("  α = 6/10 = SO(5)→SO(4) symmetry breaking")
    print("  Born collapse at δ/δ_L ≈ 0.6 (vorticity network failure)")
    print("  Reconnection = topological charge Q jump")
    print()
    
    # Fe でテスト
    test = TensileTestWithTopology(
        material=MaterialGPU.Fe(),
        N=20,  # 計算時間短縮
        vacancy_fraction=0.03,  # 空孔多めでリコネクション促進
        T=800.0,  # 高温
        seed=42
    )
    
    # 引張試験実行
    results = test.run(sigma_max=1000e6, n_steps=40, phase_update_steps=10)
    
    # プロット
    test.plot_results(save_path='/home/claude/tensile_topology_test.png')


if __name__ == "__main__":
    main()
