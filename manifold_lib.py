"""
=============================================================================
Manifold Library for EDR Theory
多様体ベースの破断判定ライブラリ

安全/危険軌道の管理・評価・学習を統合的に扱う独立モジュール

【設計思想】
- edr_fit.pyのコア機能に依存しつつ、多様体独自の機能を提供
- 将来的な拡張（危険多様体、混合多様体など）を考慮
- クリーンなインターフェースで再利用性を高める

Author: 飯泉真道 + 環
Date: 2025-01-19
=============================================================================
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =============================================================================
# Section 1: Gram行列とイベント表現
# =============================================================================

@jit
def compute_gram(x: jnp.ndarray) -> jnp.ndarray:
    """
    Gram行列計算（スケール不変版）
    
    Args:
        x: [time] または [batch, time]
    Returns:
        Gram行列 [time, time] または [batch, time, time]
    """
    # 0-mean化
    x_centered = x - jnp.mean(x, axis=-1, keepdims=True)
    # L2正規化
    x_norm = x_centered / (jnp.linalg.norm(x_centered, axis=-1, keepdims=True) + 1e-8)
    # Gram行列
    if x_norm.ndim == 1:
        return jnp.outer(x_norm, x_norm)
    else:
        return x_norm @ x_norm.T

@jit
def gram_distance(G1: jnp.ndarray, G2: jnp.ndarray) -> jnp.ndarray:
    """2つのGram行列間の距離"""
    return jnp.sum((G1 - G2)**2)

@jit
def batch_gram_distance(G_test: jnp.ndarray, G_batch: jnp.ndarray) -> jnp.ndarray:
    """
    テストGram行列と複数のGram行列間の距離を一括計算
    
    Args:
        G_test: テストGram行列 [time, time]
        G_batch: 比較対象Gram行列群 [n_samples, time, time]
    Returns:
        距離配列 [n_samples]
    """
    return vmap(lambda G: gram_distance(G_test, G))(G_batch)

# =============================================================================
# Section 2: 正則化項（物理的異常検出）
# =============================================================================

class RegularizationTerms:
    """正則化項の集約クラス"""
    
    @staticmethod
    @jit
    def total_variation(Lambda: jnp.ndarray) -> jnp.ndarray:
        """
        Total Variation（全変動）
        時間方向の急激な変化を検出
        """
        if Lambda.ndim == 1:
            return jnp.sum(jnp.abs(jnp.diff(Lambda)))
        else:
            # [paths, time]
            tv_time = jnp.sum(jnp.abs(jnp.diff(Lambda, axis=1)))
            tv_path = jnp.sum(jnp.abs(jnp.diff(Lambda, axis=0)))
            return tv_time + tv_path
    
    @staticmethod
    @jit
    def jump_penalty(Lambda: jnp.ndarray, k: float = 2.5) -> jnp.ndarray:
        """
        Jump正則化：外れ値的な急変を検出
        
        Args:
            Lambda: [time] または [paths, time]
            k: 標準偏差の何倍を閾値とするか
        """
        if Lambda.ndim == 1:
            d = jnp.abs(jnp.diff(Lambda))
            threshold = jnp.mean(d) + k * jnp.std(d)
            return jnp.sum(jnp.maximum(0.0, d - threshold))
        else:
            d = jnp.abs(jnp.diff(Lambda, axis=1))
            threshold = jnp.mean(d, axis=1, keepdims=True) + k * jnp.std(d, axis=1, keepdims=True)
            return jnp.sum(jnp.maximum(0.0, d - threshold))
    
    @staticmethod
    @jit
    def topology_penalty(Lambda: jnp.ndarray) -> jnp.ndarray:
        """
        位相連続性正則化
        Phase遷移の滑らかさを評価
        """
        if Lambda.ndim == 1:
            phase = jnp.arctan2(Lambda[1:], Lambda[:-1])
            dphase = jnp.diff(phase)
            dphase = (dphase + jnp.pi) % (2 * jnp.pi) - jnp.pi
            return jnp.sum(dphase**2)
        else:
            phase = jnp.arctan2(Lambda[:, 1:], Lambda[:, :-1])
            dphase = jnp.diff(phase, axis=1)
            dphase = (dphase + jnp.pi) % (2 * jnp.pi) - jnp.pi
            return jnp.sum(dphase**2)
    
    @staticmethod
    @jit
    def l1_norm(Lambda: jnp.ndarray) -> jnp.ndarray:
        """L1正則化（スパース性促進）"""
        return jnp.sum(jnp.abs(Lambda))
    
    @staticmethod
    @jit
    def compute_all(Lambda: jnp.ndarray, weights: Dict[str, float]) -> Dict[str, jnp.ndarray]:
        """全正則化項を計算して辞書で返す"""
        return {
            'tv': RegularizationTerms.total_variation(Lambda),
            'jump': RegularizationTerms.jump_penalty(Lambda),
            'topo': RegularizationTerms.topology_penalty(Lambda),
            'l1': RegularizationTerms.l1_norm(Lambda)
        }

# =============================================================================
# Section 3: 多様体データ構造
# =============================================================================

@dataclass
class Manifold:
    """多様体の基底データクラス"""
    lambdas: jnp.ndarray      # 軌道データ [n_trajectories, time_steps]
    grams: jnp.ndarray        # Gram行列群 [n_trajectories, time_steps, time_steps]
    conditions: List[Dict]    # 各軌道の生成条件
    n_trajectories: int       # 軌道数
    manifold_type: str        # 'safe', 'danger', 'mixed' など
    metadata: Dict = None     # その他のメタデータ

@dataclass
class SafeManifold(Manifold):
    """安全多様体"""
    safety_threshold: float = 0.85
    
    def __post_init__(self):
        self.manifold_type = 'safe'

@dataclass
class DangerManifold(Manifold):
    """危険多様体"""
    danger_threshold: float = 1.15
    
    def __post_init__(self):
        self.manifold_type = 'danger'

# =============================================================================
# Section 4: 多様体ビルダー（抽象基底クラス）
# =============================================================================

class ManifoldBuilder(ABC):
    """多様体構築の基底クラス"""
    
    def __init__(self, mat_dict: Dict, edr_dict: Dict, simulate_fn=None):
        """
        Args:
            mat_dict: 材料パラメータ
            edr_dict: EDRパラメータ
            simulate_fn: シミュレーション関数（edr_fit.simulate_lambda_jaxなど）
        """
        self.mat_dict = mat_dict
        self.edr_dict = edr_dict
        self.simulate_fn = simulate_fn
    
    @abstractmethod
    def build(self, **params) -> Manifold:
        """多様体を構築する（サブクラスで実装）"""
        pass
    
    def _generate_trajectory(self, beta: float, mu: float, pN: float, 
                           duration: float, major_rate: float) -> Tuple[jnp.ndarray, Dict]:
        """
        単一軌道を生成する共通メソッド
        
        Returns:
            Lambda_smooth: スムージング済みΛ(t)
            condition: 生成条件
        """
        from edr_fit import triax_from_path_jax, smooth_signal_jax
        
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
            'mu': jnp.full(N, float(mu)),
            'pN': jnp.full(N, float(pN)),
            'vslip': jnp.full(N, 0.015),
            'htc': jnp.full(N, 8000.0),
            'Tdie': jnp.full(N, 293.15),
            'contact': jnp.full(N, 1.0),
            'T0': 293.15
        }
        
        res = self.simulate_fn(schedule_dict, self.mat_dict, self.edr_dict)
        Lambda = res["Lambda"]
        Lambda_smooth = smooth_signal_jax(Lambda, window_size=11)
        
        condition = {
            'beta': float(beta),
            'mu': float(mu),
            'pN': float(pN),
            'major_rate': float(major_rate),
            'duration': float(duration),
            'peak_Lambda': float(jnp.max(Lambda_smooth))
        }
        
        return Lambda_smooth, condition

# =============================================================================
# Section 5: 安全多様体ビルダー
# =============================================================================

class SafeManifoldBuilder(ManifoldBuilder):
    """安全多様体構築クラス"""
    
    def build(self, 
              n_beta: int = 15,
              n_mu: int = 5,
              n_pN: int = 5,
              duration: float = 0.6,
              safety_margin: float = 0.85,
              verbose: bool = True) -> SafeManifold:
        """
        安全多様体を構築
        
        Args:
            n_beta: β方向のサンプル数
            n_mu: 摩擦係数のサンプル数
            n_pN: 接触圧のサンプル数
            duration: シミュレーション時間
            safety_margin: 安全判定のマージン（Lambda_crit * safety_margin以下を安全）
            verbose: 進捗表示
        
        Returns:
            SafeManifold: 構築された安全多様体
        """
        if verbose:
            print("\n" + "="*60)
            print(" 🛡️ Safe Manifold Construction")
            print("="*60)
            print(f"  Parameter space:")
            print(f"    β: {n_beta} points")
            print(f"    μ: {n_mu} points")
            print(f"    pN: {n_pN} points")
            print(f"    Total: {n_beta * n_mu * n_pN} trajectories")
        
        safe_lambdas = []
        safe_grams = []
        safe_conditions = []
        
        Lambda_crit = self.edr_dict['Lambda_crit']
        safety_threshold = Lambda_crit * safety_margin
        
        # パラメータグリッド
        betas = jnp.linspace(-0.7, 0.7, n_beta)
        mus = jnp.linspace(0.05, 0.10, n_mu)
        pNs = jnp.linspace(150e6, 250e6, n_pN)
        
        count = 0
        safe_count = 0
        
        for beta in betas:
            for mu in mus:
                for pN in pNs:
                    count += 1
                    
                    # 低負荷条件で軌道生成
                    Lambda_smooth, condition = self._generate_trajectory(
                        beta, mu, pN, duration, major_rate=0.4
                    )
                    
                    # 安全判定
                    if condition['peak_Lambda'] < safety_threshold:
                        safe_lambdas.append(np.array(Lambda_smooth))
                        safe_grams.append(np.array(compute_gram(Lambda_smooth)))
                        safe_conditions.append(condition)
                        safe_count += 1
                    
                    if verbose and count % 20 == 0:
                        print(f"    Progress: {count}/{n_beta*n_mu*n_pN}, "
                              f"Safe: {safe_count}")
        
        if verbose:
            print(f"\n  ✅ Complete!")
            print(f"    Generated: {count} trajectories")
            print(f"    Safe: {safe_count} ({safe_count/count*100:.1f}%)")
        
        return SafeManifold(
            lambdas=jnp.array(safe_lambdas),
            grams=jnp.array(safe_grams),
            conditions=safe_conditions,
            n_trajectories=safe_count,
            manifold_type='safe',
            safety_threshold=safety_threshold,
            metadata={'total_generated': count}
        )

# =============================================================================
# Section 6: 危険多様体ビルダー（将来拡張用）
# =============================================================================

class DangerManifoldBuilder(ManifoldBuilder):
    """危険多様体構築クラス"""
    
    def build(self,
              n_beta: int = 15,
              n_mu: int = 5,
              n_pN: int = 5,
              duration: float = 0.8,
              danger_margin: float = 1.15,
              verbose: bool = True) -> DangerManifold:
        """
        危険多様体を構築（破断パターンの学習用）
        
        Args:
            n_beta: β方向のサンプル数
            n_mu: 摩擦係数のサンプル数  
            n_pN: 接触圧のサンプル数
            duration: シミュレーション時間（より長い）
            danger_margin: 危険判定のマージン（Lambda_crit * danger_margin以上を危険）
            verbose: 進捗表示
        
        Returns:
            DangerManifold: 構築された危険多様体
        """
        if verbose:
            print("\n" + "="*60)
            print(" ⚠️  Danger Manifold Construction")
            print("="*60)
            print(f"  High-stress parameter space:")
            print(f"    β: {n_beta} points")
            print(f"    μ: {n_mu} points (higher range)")
            print(f"    pN: {n_pN} points (higher range)")
        
        danger_lambdas = []
        danger_grams = []
        danger_conditions = []
        
        Lambda_crit = self.edr_dict['Lambda_crit']
        danger_threshold = Lambda_crit * danger_margin
        
        # 高負荷パラメータグリッド
        betas = jnp.linspace(-0.8, 0.8, n_beta)
        mus = jnp.linspace(0.10, 0.20, n_mu)  # より高い摩擦
        pNs = jnp.linspace(250e6, 350e6, n_pN)  # より高い圧力
        
        count = 0
        danger_count = 0
        
        for beta in betas:
            for mu in mus:
                for pN in pNs:
                    count += 1
                    
                    # 高負荷条件で軌道生成
                    Lambda_smooth, condition = self._generate_trajectory(
                        beta, mu, pN, duration, major_rate=0.8  # 高ひずみ速度
                    )
                    
                    # 危険判定
                    if condition['peak_Lambda'] > danger_threshold:
                        danger_lambdas.append(np.array(Lambda_smooth))
                        danger_grams.append(np.array(compute_gram(Lambda_smooth)))
                        danger_conditions.append(condition)
                        danger_count += 1
                    
                    if verbose and count % 20 == 0:
                        print(f"    Progress: {count}/{n_beta*n_mu*n_pN}, "
                              f"Danger: {danger_count}")
        
        if verbose:
            print(f"\n  ✅ Complete!")
            print(f"    Generated: {count} trajectories")
            print(f"    Dangerous: {danger_count} ({danger_count/count*100:.1f}%)")
        
        return DangerManifold(
            lambdas=jnp.array(danger_lambdas),
            grams=jnp.array(danger_grams),
            conditions=danger_conditions,
            n_trajectories=danger_count,
            manifold_type='danger',
            danger_threshold=danger_threshold,
            metadata={'total_generated': count}
        )

# =============================================================================
# Section 7: 多様体アナライザー
# =============================================================================

class ManifoldAnalyzer:
    """多様体ベースの解析クラス"""
    
    def __init__(self, regularization_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            regularization_weights: 正則化項の重み
        """
        if regularization_weights is None:
            self.weights = {
                'tv': 0.1,
                'jump': 0.5,
                'topo': 0.1,
                'l1': 1e-3
            }
        else:
            self.weights = regularization_weights
    
    def compute_safety_score(self, 
                            Lambda: jnp.ndarray,
                            safe_manifold: SafeManifold) -> jnp.ndarray:
        """
        安全スコア計算（低いほど安全）
        
        Args:
            Lambda: テスト軌道 [time]
            safe_manifold: 安全多様体
        
        Returns:
            score: 安全スコア
        """
        from edr_fit import smooth_signal_jax
        
        # スムージング
        Lambda_smooth = smooth_signal_jax(Lambda, window_size=11)
        
        # Gram行列計算
        G_test = compute_gram(Lambda_smooth)
        
        # 最近傍安全軌道との距離
        distances = batch_gram_distance(G_test, safe_manifold.grams)
        min_dist = jnp.min(distances)
        
        # 正則化項
        reg_terms = RegularizationTerms.compute_all(Lambda_smooth, self.weights)
        
        # 総合スコア
        score = min_dist
        for key, weight in self.weights.items():
            score += weight * reg_terms[key]
        
        return score
    
    def compute_danger_proximity(self,
                                Lambda: jnp.ndarray,
                                danger_manifold: DangerManifold) -> jnp.ndarray:
        """
        危険近接度計算（高いほど危険）
        
        Args:
            Lambda: テスト軌道 [time]
            danger_manifold: 危険多様体
        
        Returns:
            proximity: 危険近接度
        """
        from edr_fit import smooth_signal_jax
        
        Lambda_smooth = smooth_signal_jax(Lambda, window_size=11)
        G_test = compute_gram(Lambda_smooth)
        
        # 最近傍危険軌道との距離（逆数で近接度に変換）
        distances = batch_gram_distance(G_test, danger_manifold.grams)
        min_dist = jnp.min(distances)
        proximity = 1.0 / (min_dist + 1e-6)
        
        return proximity
    
    def analyze_trajectory(self,
                          Lambda: jnp.ndarray,
                          manifolds: Dict[str, Manifold]) -> Dict:
        """
        軌道の包括的分析
        
        Args:
            Lambda: テスト軌道
            manifolds: 利用可能な多様体の辞書
        
        Returns:
            analysis: 分析結果
        """
        from edr_fit import smooth_signal_jax
        
        Lambda_smooth = smooth_signal_jax(Lambda, window_size=11)
        
        analysis = {
            'peak_Lambda': float(jnp.max(Lambda_smooth)),
            'regularization': RegularizationTerms.compute_all(Lambda_smooth, self.weights)
        }
        
        # 各多様体に対するスコア計算
        if 'safe' in manifolds and isinstance(manifolds['safe'], SafeManifold):
            analysis['safety_score'] = float(
                self.compute_safety_score(Lambda, manifolds['safe'])
            )
        
        if 'danger' in manifolds and isinstance(manifolds['danger'], DangerManifold):
            analysis['danger_proximity'] = float(
                self.compute_danger_proximity(Lambda, manifolds['danger'])
            )
        
        return analysis

# =============================================================================
# Section 8: 便利な統合関数
# =============================================================================

def create_manifold_system(mat_dict: Dict, 
                          edr_dict: Dict,
                          simulate_fn,
                          build_safe: bool = True,
                          build_danger: bool = False,
                          verbose: bool = True) -> Tuple[Dict[str, Manifold], ManifoldAnalyzer]:
    """
    多様体システムを一括構築
    
    Args:
        mat_dict: 材料パラメータ
        edr_dict: EDRパラメータ
        simulate_fn: シミュレーション関数
        build_safe: 安全多様体を構築するか
        build_danger: 危険多様体を構築するか
        verbose: 進捗表示
    
    Returns:
        manifolds: 構築された多様体の辞書
        analyzer: 解析器
    """
    manifolds = {}
    
    if build_safe:
        builder = SafeManifoldBuilder(mat_dict, edr_dict, simulate_fn)
        manifolds['safe'] = builder.build(verbose=verbose)
    
    if build_danger:
        builder = DangerManifoldBuilder(mat_dict, edr_dict, simulate_fn)
        manifolds['danger'] = builder.build(verbose=verbose)
    
    analyzer = ManifoldAnalyzer()
    
    return manifolds, analyzer

# エクスポート
__all__ = [
    # Gram行列関連
    'compute_gram',
    'gram_distance',
    'batch_gram_distance',
    
    # 正則化項
    'RegularizationTerms',
    
    # データ構造
    'Manifold',
    'SafeManifold', 
    'DangerManifold',
    
    # ビルダー
    'ManifoldBuilder',
    'SafeManifoldBuilder',
    'DangerManifoldBuilder',
    
    # アナライザー
    'ManifoldAnalyzer',
    
    # 統合関数
    'create_manifold_system'
]
