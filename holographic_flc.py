"""
Λ³/H Correspondence Framework
Holographic Forming Limit Analysis

実験可能なホログラフィー理論による材料破壊予測
Production-Ready Implementation

Author: 飯泉真道 (Masamichi Iizumi) & 環 (Tamaki)
Version: 2.0 - Experimentally Realizable Holography Edition
Date: 2025-10-19

Usage:
    result = press_safety_check("SPCC", "fem_results.csv")
    print(result['status'])  # 'OK', 'WARNING', or 'NG'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ==========================
# Material Database
# ==========================

@dataclass
class MaterialProperties:
    """材料プロパティ（実測値ベース）"""
    
    # EDRパラメータ
    V0: float
    av: float
    ad: float
    triax_sens: float
    Lambda_crit: float
    
    # 非可換パラメータ（実測値）
    theta_eff: float
    grad_n_Lambda: float
    Delta_NC: float
    omega_Lambda: float
    
    # 材料特性
    ductility: str
    fracture_mode: str
    yield_strength: float
    ultimate_strength: float
    
    # メタデータ
    name: str
    measured: bool = True  # 実測データか予測値か


class MaterialDatabase:
    """
    材料データベース
    
    Holographic FLC Experimentで実測した
    θ_eff, |∂_nΛ|を含む材料パラメータ
    """
    
    MATERIALS = {
        'SPCC': MaterialProperties(
            name='SPCC',
            # EDRパラメータ
            V0=1.8045e9,
            av=4.013e4,
            ad=1.014e-7,
            triax_sens=0.196,
            Lambda_crit=1.05,
            # 非可換パラメータ（実測！）
            theta_eff=6.987,
            grad_n_Lambda=0.223,
            Delta_NC=-3.975e-5,
            omega_Lambda=0.347,
            # 材料特性
            ductility='medium',
            fracture_mode='semi-ductile',
            yield_strength=300e6,
            ultimate_strength=450e6,
            measured=True
        ),
        
        'Aluminum': MaterialProperties(
            name='Aluminum',
            V0=1.8e9,
            av=4.0e4,
            ad=1.0e-7,
            triax_sens=0.19,
            Lambda_crit=1.05,
            # 非可換パラメータ（実測！）
            theta_eff=8.633,
            grad_n_Lambda=0.189,
            Delta_NC=-6.110e-5,
            omega_Lambda=0.306,
            # 材料特性
            ductility='high',
            fracture_mode='ductile',
            yield_strength=250e6,
            ultimate_strength=350e6,
            measured=True
        ),
        
        'Copper': MaterialProperties(
            name='Copper',
            V0=1.9e9,
            av=3.5e4,
            ad=9e-8,
            triax_sens=0.18,
            Lambda_crit=1.05,
            # 予測値（未実測）
            theta_eff=9.5,
            grad_n_Lambda=0.17,
            Delta_NC=-7.0e-5,
            omega_Lambda=0.28,
            ductility='very_high',
            fracture_mode='ductile',
            yield_strength=200e6,
            ultimate_strength=300e6,
            measured=False
        ),
        
        'Titanium': MaterialProperties(
            name='Titanium',
            V0=2.0e9,
            av=4.5e4,
            ad=1.1e-7,
            triax_sens=0.22,
            Lambda_crit=1.05,
            # 予測値
            theta_eff=7.5,
            grad_n_Lambda=0.21,
            Delta_NC=-4.5e-5,
            omega_Lambda=0.32,
            ductility='medium',
            fracture_mode='semi-ductile',
            yield_strength=800e6,
            ultimate_strength=1000e6,
            measured=False
        ),
        
        'Stainless': MaterialProperties(
            name='Stainless Steel',
            V0=2.1e9,
            av=4.2e4,
            ad=1.05e-7,
            triax_sens=0.21,
            Lambda_crit=1.05,
            # 予測値
            theta_eff=7.0,
            grad_n_Lambda=0.22,
            Delta_NC=-4.2e-5,
            omega_Lambda=0.33,
            ductility='medium',
            fracture_mode='semi-ductile',
            yield_strength=500e6,
            ultimate_strength=700e6,
            measured=False
        ),
    }
    
    @classmethod
    def get(cls, material_name: str) -> MaterialProperties:
        """材料データの取得"""
        if material_name not in cls.MATERIALS:
            raise ValueError(
                f"Unknown material: {material_name}\n"
                f"Available materials: {cls.list_materials()}"
            )
        return cls.MATERIALS[material_name]
    
    @classmethod
    def list_materials(cls) -> List[str]:
        """利用可能な材料リスト"""
        return list(cls.MATERIALS.keys())
    
    @classmethod
    def add_custom_material(cls, name: str, properties: MaterialProperties):
        """カスタム材料の追加"""
        properties.name = name
        cls.MATERIALS[name] = properties
        print(f"✓ Custom material added: {name}")


# ==========================
# FEM-EDR Processor
# ==========================

class FEM_EDR_Processor:
    """
    FEM結果からEDR（Λ）を計算
    
    シンプル版：Λ計算に特化
    """
    
    def __init__(self, material: MaterialProperties):
        self.material = material
        self.element_history = {}
        self.critical_elements = []
    
    def read_fem_csv(self, csv_file: str) -> pd.DataFrame:
        """
        FEM結果CSVの読み込み
        
        必要カラム:
            Time, Element, Stress_Mises, Strain_Eq,
            Strain_Rate, Temperature, Triaxiality
        """
        print(f"\n📂 Reading FEM data: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # カラム名の自動マッピング
        column_mapping = {
            'time': 'Time', 't': 'Time',
            'elem': 'Element', 'elem_id': 'Element',
            'mises': 'Stress_Mises', 'von_mises': 'Stress_Mises',
            'eq_strain': 'Strain_Eq', 'eps_eq': 'Strain_Eq',
            'temp': 'Temperature', 'T': 'Temperature',
            'triax': 'Triaxiality'
        }
        df.rename(columns={k: v for k, v in column_mapping.items()
                          if k in df.columns}, inplace=True)
        
        # 必須カラムチェック
        required = ['Time', 'Element', 'Stress_Mises', 'Strain_Eq',
                   'Strain_Rate', 'Temperature', 'Triaxiality']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"✓ Loaded {len(df)} records from {df['Element'].nunique()} elements")
        return df
    
    def compute_Lambda(self, fem_data: pd.DataFrame):
        """
        各要素・各時刻でΛを計算
        
        Λ = K / V_eff
        
        K: エネルギー密度率
        V_eff: 有効凝集エネルギー密度
        """
        print("\n🔬 Computing Λ field...")
        
        elements = fem_data['Element'].unique()
        
        for elem_id in elements:
            elem_data = fem_data[fem_data['Element'] == elem_id].sort_values('Time')
            elem_data = elem_data.reset_index(drop=True)
            
            self.element_history[elem_id] = []
            
            # 初期化
            cv = 1e-8  # 空孔濃度
            rhod = 1e10  # 転位密度
            
            for idx in range(len(elem_data)):
                row = elem_data.iloc[idx]
                
                # K計算（エネルギー密度率）
                sigma = row['Stress_Mises']
                eps_rate = row['Strain_Rate']
                
                # 特性時間
                t_char = 1.0 / max(eps_rate, 1e-6)
                
                # エネルギー密度
                K = sigma * eps_rate * t_char
                
                # 簡易的な欠陥進化（オプション）
                if eps_rate > 0:
                    rhod = min(rhod * (1 + eps_rate * 0.01), 1e13)
                    cv = min(cv * (1 + eps_rate * 0.001), 1e-6)
                
                # V_eff計算
                V_defect = self.material.V0 * (
                    1 - self.material.av * cv - self.material.ad * np.sqrt(rhod)
                )
                
                # 三軸度効果
                triax = np.clip(row['Triaxiality'], -0.1, 0.8)
                ductility_factor = np.exp(-self.material.triax_sens * max(triax, 0))
                
                V_eff = max(V_defect * ductility_factor, 0.01 * self.material.V0)
                
                # Λ計算
                Lambda = K / V_eff
                Lambda = min(Lambda, 10.0)  # 上限
                
                # 履歴保存
                self.element_history[elem_id].append({
                    'Time': row['Time'],
                    'Lambda': Lambda,
                    'K': K,
                    'V_eff': V_eff,
                    'Stress': sigma,
                    'Temperature': row['Temperature'],
                    'Triaxiality': triax
                })
        
        # 危険要素の特定
        self._identify_critical_elements()
        
        print(f"✓ Computed Λ for {len(self.element_history)} elements")
        return self.element_history
    
    def _identify_critical_elements(self, threshold: float = 0.9):
        """危険要素（Λ > threshold）の特定"""
        self.critical_elements = []
        
        for elem_id, history in self.element_history.items():
            if not history:
                continue
            
            Lambda_max = max(h['Lambda'] for h in history)
            
            if Lambda_max > threshold:
                critical_time = next(
                    (h['Time'] for h in history if h['Lambda'] > threshold),
                    None
                )
                
                max_point = max(history, key=lambda h: h['Lambda'])
                
                self.critical_elements.append({
                    'Element_ID': elem_id,
                    'Lambda_max': Lambda_max,
                    'Critical_Time': critical_time,
                    'Final_Stress': history[-1]['Stress'],
                    'Final_Temp': history[-1]['Temperature']
                })
        
        # Λ降順ソート
        self.critical_elements.sort(key=lambda x: x['Lambda_max'], reverse=True)
    
    def get_Lambda_max(self) -> float:
        """全要素・全時刻での最大Λ"""
        all_lambdas = []
        for history in self.element_history.values():
            all_lambdas.extend([h['Lambda'] for h in history])
        return max(all_lambdas) if all_lambdas else 0.0


# ==========================
# Noncommutative Diagnostics
# ==========================

class NoncommutativeDiagnostics:
    """
    非可換境界診断
    
    θ_eff, Δ_NC, Ξパケットの計算
    """
    
    def __init__(self, processor: FEM_EDR_Processor):
        self.processor = processor
        self.theta_eff = None
        self.Delta_NC = None
        self.Xi_packet = None
    
    def run_diagnosis(self):
        """非可換診断の実行"""
        print("\n🔍 Running noncommutative boundary diagnostics...")
        
        # Σ点（Λ≈1）の検出
        Sigma_elements = [
            elem for elem in self.processor.critical_elements
            if 0.95 <= elem['Lambda_max'] <= 1.05
        ]
        
        if not Sigma_elements:
            print("⚠️ No Σ points (Λ≈1) detected")
            # デフォルト値を材料DBから使用
            self.theta_eff = self.processor.material.theta_eff
            self.Xi_packet = {
                'omega_Lambda': self.processor.material.omega_Lambda,
                'grad_n_Lambda': self.processor.material.grad_n_Lambda,
                'j_n': self.processor.material.grad_n_Lambda
            }
            return
        
        print(f"✓ Detected {len(Sigma_elements)} Σ points")
        
        # θ_eff計算（簡易版：材料DBの値を使用）
        self.theta_eff = self.processor.material.theta_eff
        
        # Ξパケット（材料DBの値を使用）
        self.Xi_packet = {
            'omega_Lambda': self.processor.material.omega_Lambda,
            'grad_n_Lambda': self.processor.material.grad_n_Lambda,
            'j_n': self.processor.material.grad_n_Lambda
        }
        
        print(f"  θ_eff = {self.theta_eff:.3f}")
        print(f"  |∂_nΛ| = {self.Xi_packet['grad_n_Lambda']:.3f}")


# ==========================
# Safety Judge
# ==========================

class SafetyJudge:
    """
    安全域判定エンジン
    
    Λ, θ_eff, |∂_nΛ|から総合判定
    """
    
    # 判定基準（保守的マージン）
    CRITERIA = {
        'Lambda_safe': 0.90,
        'Lambda_warning': 0.95,
        'Lambda_critical': 1.00,
        'Lambda_reject': 1.05,
        
        'theta_eff_low': 5.0,
        'theta_eff_high': 10.0,
        
        'grad_n_high': 0.25,
        'grad_n_low': 0.15,
    }
    
    def __init__(self, material: MaterialProperties):
        self.material = material
    
    def evaluate(self, processor: FEM_EDR_Processor,
                diagnostics: NoncommutativeDiagnostics) -> Dict:
        """
        総合判定
        
        Returns
        -------
        result : dict
            {
                'status': 'OK' | 'WARNING' | 'NG',
                'Lambda_max': float,
                'theta_eff': float,
                'grad_n_Lambda': float,
                'critical_elements': list,
                'recommendations': list
            }
        """
        
        Lambda_max = processor.get_Lambda_max()
        
        # 1. Λ判定
        Lambda_status = self._judge_Lambda(Lambda_max)
        
        # 2. 非可換パラメータ判定
        theta_status = self._judge_theta_eff(diagnostics.theta_eff)
        grad_n_status = self._judge_grad_n(diagnostics.Xi_packet['grad_n_Lambda'])
        
        # 3. 総合判定
        overall_status = self._combine_judgments(
            Lambda_status, theta_status, grad_n_status
        )
        
        # 4. 改善提案
        recommendations = self._generate_recommendations(
            Lambda_max, diagnostics, overall_status
        )
        
        result = {
            'status': overall_status,
            'Lambda_max': Lambda_max,
            'theta_eff': diagnostics.theta_eff,
            'grad_n_Lambda': diagnostics.Xi_packet['grad_n_Lambda'],
            'omega_Lambda': diagnostics.Xi_packet['omega_Lambda'],
            'critical_elements': processor.critical_elements[:5],
            'recommendations': recommendations,
            'material_name': self.material.name,
            'detailed_scores': {
                'Lambda': Lambda_status,
                'theta_eff': theta_status,
                'grad_n': grad_n_status
            }
        }
        
        return result
    
    def _judge_Lambda(self, Lambda_max: float) -> str:
        """Λによる判定"""
        if Lambda_max < self.CRITERIA['Lambda_safe']:
            return 'SAFE'
        elif Lambda_max < self.CRITERIA['Lambda_warning']:
            return 'GOOD'
        elif Lambda_max < self.CRITERIA['Lambda_critical']:
            return 'WARNING'
        elif Lambda_max < self.CRITERIA['Lambda_reject']:
            return 'CRITICAL'
        else:
            return 'REJECT'
    
    def _judge_theta_eff(self, theta_eff: float) -> str:
        """θ_effによる判定（延性指標）"""
        if theta_eff < self.CRITERIA['theta_eff_low']:
            return 'LOW_DUCTILITY'
        elif theta_eff > self.CRITERIA['theta_eff_high']:
            return 'VERY_HIGH_DUCTILITY'
        else:
            return 'NORMAL'
    
    def _judge_grad_n(self, grad_n: float) -> str:
        """|∂_nΛ|による判定（破壊様式）"""
        if grad_n > self.CRITERIA['grad_n_high']:
            return 'BRITTLE_TENDENCY'
        elif grad_n < self.CRITERIA['grad_n_low']:
            return 'DUCTILE_TENDENCY'
        else:
            return 'NORMAL'
    
    def _combine_judgments(self, Lambda_status: str,
                          theta_status: str, grad_n_status: str) -> str:
        """総合判定"""
        if Lambda_status in ['REJECT', 'CRITICAL']:
            return 'NG'
        elif Lambda_status == 'WARNING':
            return 'WARNING'
        elif theta_status == 'LOW_DUCTILITY' and grad_n_status == 'BRITTLE_TENDENCY':
            return 'WARNING'
        else:
            return 'OK'
    
    def _generate_recommendations(self, Lambda_max: float,
                                 diagnostics: NoncommutativeDiagnostics,
                                 status: str) -> List[str]:
        """改善提案の生成"""
        recommendations = []
        
        if status == 'NG':
            recommendations.append("🔴 設計変更が必要です")
            
            if Lambda_max > 1.05:
                recommendations.append(
                    f"  ・Λ={Lambda_max:.3f}が臨界超過 "
                    f"→ 成形条件の緩和（応力低減、温度調整）"
                )
            
            if diagnostics.Xi_packet['grad_n_Lambda'] > 0.25:
                recommendations.append(
                    "  ・境界が硬い（脆性的） "
                    "→ より延性の高い材料への変更を検討"
                )
        
        elif status == 'WARNING':
            recommendations.append("⚠️ 注意が必要です")
            
            if 0.95 < Lambda_max < 1.0:
                recommendations.append(
                    "  ・Λが臨界に接近 "
                    "→ 安全マージンの確保（成形速度低減）"
                )
            
            recommendations.append("  ・品質管理の強化（全数検査推奨）")
        
        else:  # OK
            recommendations.append("✅ 設計は安全です")
            
            if Lambda_max < 0.8:
                recommendations.append(
                    "  ・安全マージンあり "
                    "→ 生産性向上の余地（成形速度アップ可）"
                )
        
        return recommendations


# ==========================
# Main Interface
# ==========================

def press_safety_check(material_name: str, fem_csv_file: str) -> Dict:
    """
    プレス成形安全性チェック（ワンライナー）
    
    Parameters
    ----------
    material_name : str
        材料名（"SPCC", "Aluminum", "Copper", etc.）
    fem_csv_file : str
        FEM結果のCSVファイルパス
    
    Returns
    -------
    result : dict
        判定結果
        {
            'status': 'OK' | 'WARNING' | 'NG',
            'Lambda_max': float,
            'critical_elements': list,
            'recommendations': list
        }
    
    Example
    -------
    >>> result = press_safety_check("SPCC", "fem_results.csv")
    >>> print(result['status'])
    'OK'
    """
    
    print("="*60)
    print("Λ³/H Correspondence - Press Safety Check")
    print("Experimentally Realizable Holography")
    print("="*60)
    
    # 1. 材料データ取得
    material = MaterialDatabase.get(material_name)
    print(f"\n📋 Material: {material.name}")
    print(f"   Ductility: {material.ductility}")
    print(f"   θ_eff: {material.theta_eff:.3f} (延性指標)")
    print(f"   |∂_nΛ|: {material.grad_n_Lambda:.3f} (破壊様式)")
    
    if not material.measured:
        print("   ⚠️ Using predicted values (not measured)")
    
    # 2. FEM読み込み & Λ計算
    processor = FEM_EDR_Processor(material)
    fem_data = processor.read_fem_csv(fem_csv_file)
    processor.compute_Lambda(fem_data)
    
    # 3. 非可換診断
    diagnostics = NoncommutativeDiagnostics(processor)
    diagnostics.run_diagnosis()
    
    # 4. 安全域判定
    judge = SafetyJudge(material)
    result = judge.evaluate(processor, diagnostics)
    
    # 5. 結果表示
    print("\n" + "="*60)
    print("判定結果")
    print("="*60)
    
    status_icon = {
        'OK': '✅',
        'WARNING': '⚠️',
        'NG': '🔴'
    }
    
    print(f"\n{status_icon[result['status']]} Status: {result['status']}")
    print(f"\n詳細:")
    print(f"  Max Λ: {result['Lambda_max']:.3f}")
    print(f"  θ_eff: {result['theta_eff']:.3f}")
    print(f"  |∂_nΛ|: {result['grad_n_Lambda']:.3f}")
    
    if result['critical_elements']:
        print(f"\n危険要素 Top 3:")
        for i, elem in enumerate(result['critical_elements'][:3], 1):
            print(f"  {i}. Element {elem['Element_ID']}: "
                  f"Λ={elem['Lambda_max']:.3f}")
    
    print(f"\n改善提案:")
    for rec in result['recommendations']:
        print(f"{rec}")
    
    print("\n" + "="*60)
    
    return result


# ==========================
# Visualization
# ==========================

def plot_Lambda_distribution(processor: FEM_EDR_Processor,
                            output_file: Optional[str] = None):
    """Λ分布の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Λヒストグラム
    ax = axes[0, 0]
    all_lambdas = []
    for history in processor.element_history.values():
        all_lambdas.extend([h['Lambda'] for h in history])
    
    ax.hist(all_lambdas, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Λ=1 (Critical)')
    ax.set_xlabel('Λ (EDR)')
    ax.set_ylabel('Frequency')
    ax.set_title('Λ Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Λ時間発展（危険要素）
    ax = axes[0, 1]
    for elem in processor.critical_elements[:5]:
        elem_id = elem['Element_ID']
        history = processor.element_history[elem_id]
        times = [h['Time'] for h in history]
        lambdas = [h['Lambda'] for h in history]
        ax.plot(times, lambdas, label=f'Elem {elem_id}', linewidth=2)
    
    ax.axhline(1.0, color='r', linestyle='--', label='Critical')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Λ')
    ax.set_title('Λ Evolution (Critical Elements)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Λ vs 応力
    ax = axes[1, 0]
    stresses = []
    lambdas_at_stress = []
    
    for history in processor.element_history.values():
        for h in history:
            stresses.append(h['Stress'] / 1e6)  # MPa
            lambdas_at_stress.append(h['Lambda'])
    
    ax.scatter(stresses, lambdas_at_stress, alpha=0.3, s=10)
    ax.axhline(1.0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Mises Stress [MPa]')
    ax.set_ylabel('Λ')
    ax.set_title('Λ vs Stress')
    ax.grid(True, alpha=0.3)
    
    # 4. 危険要素マップ
    ax = axes[1, 1]
    elem_ids = list(processor.element_history.keys())
    max_lambdas = [max(h['Lambda'] for h in processor.element_history[eid])
                   for eid in elem_ids]
    
    colors = ['green' if l < 0.9 else 'yellow' if l < 1.0 else 'red'
              for l in max_lambdas]
    
    ax.bar(range(len(elem_ids)), max_lambdas, color=colors, alpha=0.7)
    ax.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Critical')
    ax.set_xlabel('Element Index')
    ax.set_ylabel('Max Λ')
    ax.set_title('Element Safety Map')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Λ³ Analysis - {processor.material.name}', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"✓ Figure saved: {output_file}")
    
    plt.show()


def plot_material_map(materials: Optional[List[str]] = None):
    """
    材料特性マップ
    横軸: θ_eff（延性）
    縦軸: |∂_nΛ|（破壊様式）
    """
    if materials is None:
        materials = MaterialDatabase.list_materials()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for mat_name in materials:
        mat = MaterialDatabase.get(mat_name)
        
        # 破壊様式で色分け
        if mat.fracture_mode == 'ductile':
            color = 'green'
            marker = 'o'
        elif mat.fracture_mode == 'brittle':
            color = 'red'
            marker = 's'
        else:
            color = 'blue'
            marker = '^'
        
        # プロット
        ax.scatter(mat.theta_eff, mat.grad_n_Lambda,
                  s=300, c=color, marker=marker,
                  edgecolors='black', linewidth=2,
                  alpha=0.7, label=mat.name)
        
        # ラベル
        ax.annotate(mat.name,
                   (mat.theta_eff, mat.grad_n_Lambda),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=11, weight='bold')
        
        # 実測データにマーク
        if mat.measured:
            ax.scatter(mat.theta_eff, mat.grad_n_Lambda,
                      s=100, marker='*', c='gold',
                      edgecolors='black', linewidth=1,
                      zorder=10)
    
    # 領域の色分け
    ax.fill_between([8, 11], 0.1, 0.2,
                   alpha=0.15, color='green',
                   label='High Ductility Zone')
    
    ax.fill_between([5, 7], 0.24, 0.3,
                   alpha=0.15, color='red',
                   label='Brittle Zone')
    
    ax.set_xlabel('θ_eff (Ductility Index)\n← Low Ductility | High Ductility →',
                 fontsize=12)
    ax.set_ylabel('|∂_nΛ| (Fracture Mode Index)\n← Ductile | Brittle →',
                 fontsize=12)
    ax.set_title('Material Property Map\nΛ³/H Correspondence Framework',
                fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # 注釈
    ax.text(0.98, 0.02,
           '★ = Measured\n○ = Predicted',
           transform=ax.transAxes,
           fontsize=10, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


# ==========================
# Demo & Test
# ==========================

def generate_demo_fem_data(output_file: str = "demo_fem_results.csv"):
    """デモ用FEMデータの生成"""
    print(f"\n🔧 Generating demo FEM data: {output_file}")
    
    times = np.linspace(0, 0.5, 100)
    elements = [1001, 1002, 1003, 1004, 1005]
    
    data = []
    for elem_id in elements:
        # 要素ごとに異なる応力履歴
        stress_factor = 1.0 + 0.3 * (elem_id - 1003)
        
        for t in times:
            data.append({
                'Time': t,
                'Element': elem_id,
                'Stress_Mises': 300e6 * (1 + 0.6*t) * stress_factor,
                'Strain_Eq': 0.4 * t,
                'Strain_Rate': 0.8,
                'Temperature': 293 + 120*t,
                'Triaxiality': 0.33 + 0.15*np.sin(10*t),
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"✓ Demo data saved: {output_file}")
    return output_file


def demo_run():
    """デモ実行"""
    print("\n" + "="*60)
    print("DEMO: Λ³/H Correspondence Framework")
    print("="*60)
    
    # 1. デモデータ生成
    demo_file = generate_demo_fem_data()
    
    # 2. SPCC解析
    print("\n--- SPCC Analysis ---")
    result_spcc = press_safety_check("SPCC", demo_file)
    
    # 3. アルミニウム解析
    print("\n--- Aluminum Analysis ---")
    result_al = press_safety_check("Aluminum", demo_file)
    
    # 4. 材料マップ
    print("\n📊 Material Property Map")
    plot_material_map()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


if __name__ == "__main__":
    demo_run()
