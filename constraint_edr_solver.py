"""
Constraint-based EDR Parameter Solver
境界条件制約によるEDRパラメータ逆解析
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# =============================================================================
# Section 1: パラメータ変換（物理範囲の自動保証）
# =============================================================================

def squash(x, lo, hi):
    """シグモイド関数で無制約変数を[lo, hi]に写像"""
    return lo + (hi - lo) * (1.0 / (1.0 + np.exp(-x)))

class ParamMap:
    """パラメータの再パラメータ化管理"""
    def __init__(self, bounds_dict: Dict[str, Tuple[float, float]]):
        self.keys = list(bounds_dict.keys())
        self.bounds = np.array([bounds_dict[k] for k in self.keys], float)
    
    def to_physical(self, z: np.ndarray) -> Dict[str, float]:
        """無制約変数zを物理パラメータに変換"""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return {k: squash(zi, lo[i], hi[i]) 
                for i, (k, zi) in enumerate(zip(self.keys, z))}
    
    def size(self) -> int:
        """パラメータ数"""
        return len(self.keys)

# =============================================================================
# Section 1: パラメータ変換（物理範囲の自動保証）
# =============================================================================

def get_better_initial_guess(flc_points, physics_bounds):
    """データから良い初期値を推定"""
    pmap = ParamMap(physics_bounds)
    z0 = np.zeros(pmap.size())
    
    # より良い初期値（物理的に妥当な値）
    initial_params = {
        'V0': 1e8,           # 中間的な値
        'K_scale': 1.0,
        'K_scale_draw': 1.2, # 深絞りは少し大きめ
        'K_scale_plane': 1.0,
        'K_scale_biax': 0.8, # 張出しは小さめ
        'beta_A': 0.15,      # 適度な深さ
        'beta_bw': 0.6,
        'beta_A_pos': 0.12,
        'Lambda_crit': 0.5,
    }
    
    # 逆変換して初期値を作る
    for i, key in enumerate(pmap.keys):
        if key in initial_params:
            val = initial_params[key]
            lo, hi = pmap.bounds[i]
            # squashの逆変換
            normalized = (val - lo) / (hi - lo)
            normalized = np.clip(normalized, 0.01, 0.99)
            z0[i] = -np.log(1.0/normalized - 1.0)
    
    return z0

# =============================================================================
# Section 2: EDR→FLC予測モデル
# =============================================================================

def compute_flc_point(params_dict: Dict[str, float], beta: float) -> float:
    """
    EDRパラメータからFLC限界ひずみを計算（連続・滑らか版）
    """
    # 基本パラメータ取得
    K_base = params_dict.get('K_scale', 1.0)
    K_draw = params_dict.get('K_scale_draw', 1.0)
    K_plane = params_dict.get('K_scale_plane', 1.0) 
    K_biax = params_dict.get('K_scale_biax', 1.0)
    
    # ===== 1. 経路別スケールを滑らかに補間 =====
    # ガウシアンベースの重み付け（各経路の中心で最大）
    w_draw = np.exp(-((beta + 0.5) / 0.25)**2)   # β=-0.5中心
    w_plane = np.exp(-((beta - 0.0) / 0.25)**2)   # β=0.0中心
    w_biax = np.exp(-((beta - 0.75) / 0.3)**2)    # β=0.75中心
    
    # 正規化
    w_sum = w_draw + w_plane + w_biax + 1e-10
    w_draw /= w_sum
    w_plane /= w_sum
    w_biax /= w_sum
    
    # 加重平均でKを計算
    K = K_base * (w_draw * K_draw + w_plane * K_plane + w_biax * K_biax)
    
    # ===== 2. V字形状（非対称対応） =====
    beta_A = params_dict.get('beta_A', 0.1)
    beta_bw = params_dict.get('beta_bw', 0.5)
    beta_A_pos = params_dict.get('beta_A_pos', beta_A)
    
    # ベース値（連続的に変化）
    # シグモイド関数で滑らかに遷移
    base_draw = 0.40
    base_plane = 0.30
    base_biax = 0.22
    
    # 同じ重みを使ってベース値も補間
    base = w_draw * base_draw + w_plane * base_plane + w_biax * base_biax
    
    # ===== 3. V字の深さ計算 =====
    if beta < 0:
        # 負側：通常のV字
        depth = beta_A * (1 - np.exp(-((beta / beta_bw)**2)))
    else:
        # 正側：やや狭めのV字（非対称性）
        beta_bw_pos = beta_bw * 0.8  # 正側は幅を狭く
        depth = beta_A_pos * (1 - np.exp(-((beta / beta_bw_pos)**2)))
    
    # 深さの適用（βの絶対値に比例）
    depth_effect = depth * (1 - np.exp(-2 * abs(beta)))
    
    # ===== 4. 最終的な限界ひずみ計算 =====
    Em = (base - depth_effect) * K
    
    # 物理的に妥当な範囲にクリップ
    return np.clip(Em, 0.05, 0.8)

def debug_compute_flc():
    # 中央値パラメータで確認
    test_params = {
        'V0': 5e9,
        'K_scale': 1.0,
        'K_scale_draw': 1.0,
        'K_scale_plane': 1.0,
        'K_scale_biax': 1.0,
        'beta_A': 0.1,
        'beta_bw': 0.5,
        'beta_A_pos': 0.1,
        'Lambda_crit': 1.0,
    }
    
    print("現在のcompute_flc_point出力:")
    for beta, Em_target in flc_points:
        Em_pred = compute_flc_point(test_params, beta)
        print(f"β={beta:5.2f}: Pred={Em_pred:.3f}, Target={Em_target:.3f}, "
              f"Error={abs(Em_pred-Em_target):.3f}")

debug_compute_flc()

# =============================================================================
# Section 3: 制約条件の定義
# =============================================================================

def make_boundary_constraint(flc_points, pmap):
    def cons_vec(z):
        p = pmap.to_physical(z)
        return np.array([
            (compute_flc_point(p, b) - Em) / max(Em, 1e-3)
            for (b, Em) in flc_points
        ], float)

    n = len(flc_points)
    eps_rel = 3e-3  # ← 0.3% を物理許容帯に（データ精度に合わせて 1e-3～5e-3）
    lower = -np.full(n, eps_rel)   # ★ ここが等式→帯域に変わる
    upper =  np.full(n, eps_rel)   # ★
    return NonlinearConstraint(cons_vec, lower, upper)

# =============================================================================
# Section 4: 目的関数（空間の滑らかさ）
# =============================================================================

def regularizer(z: np.ndarray, pmap: ParamMap, 
                betas_for_shape: np.ndarray) -> float:
    """空間形状の正則化項"""
    p = pmap.to_physical(z)
    Em = np.array([compute_flc_point(p, b) for b in betas_for_shape])
    
    # (a) 滑らかさ：2次差分
    if len(Em) > 2:
        d2 = np.diff(Em, n=2)
        smooth = np.mean(d2**2)
    else:
        smooth = 0.0
    
    # (b) V字単調性
    mid = len(betas_for_shape) // 2
    left = Em[:mid+1]
    right = Em[mid:]
    
    # 左枝は下降すべき
    mono_left = np.sum(np.maximum(0, np.diff(left)))
    # 右枝は上昇すべき
    mono_right = np.sum(np.maximum(0, -np.diff(right)))
    
    # (c) 対称性（弱い制約）
    if len(left) == len(right):
        sym = np.mean((left[::-1] - right[:len(left)])**2)
    else:
        sym = 0.0
    
    d1 = np.diff(Em)
    continuity = np.mean(np.diff(d1)**2)  # 微分の変化を抑制
    
    return 1.0 * smooth + 0.5 * continuity + 0.3 * (mono_left + mono_right)

# =============================================================================
# Section 5: メインソルバー
# =============================================================================

def solve_constrained_edr(flc_points: List[Tuple[float, float]], 
                          physics_bounds: Dict[str, Tuple[float, float]],
                          z0: np.ndarray = None,
                          verbose: bool = True) -> Tuple[Dict, object]:
    """
    制約充足によるEDRパラメータ推定
    
    Args:
        flc_points: [(beta, Em_target), ...] FLC実測点
        physics_bounds: パラメータの物理範囲
        z0: 初期値（無制約空間）
        verbose: 詳細出力
    
    Returns:
        最適パラメータ辞書, scipy最適化結果
    """
    pmap = ParamMap(physics_bounds)
    n = pmap.size()
    
    if z0 is None:
        z0 = np.zeros(n)  # 中央値スタート
    
    # 境界条件（等式制約）
    nlcons = make_boundary_constraint(flc_points, pmap)
    
    # 目的関数：空間形状の正則化
    beta_min = min(b for b, _ in flc_points)
    beta_max = max(b for b, _ in flc_points)
    beta_grid = np.linspace(beta_min, beta_max, 41)
    
    fun = lambda z: regularizer(z, pmap, beta_grid)
    
    # 最適化実行
    options = {
        'maxiter': 3000,
        'verbose': 2 if verbose else 0,
        'gtol': 1e-8,
        'xtol': 1e-8,
        'barrier_tol': 1e-6,          # ここは eps_rel/3 程度でもOK
        'initial_constr_penalty': 1.0,
    }

    res = minimize(fun, z0, 
                  method='trust-constr',
                  constraints=[nlcons],
                  options=options)
    
    params_phys = pmap.to_physical(res.x)
    
    if verbose:
        print("\n" + "="*60)
        print("最適化完了")
        print(f"Success: {res.success}")
        print(f"目的関数値: {res.fun:.6f}")
        print(f"制約違反: {np.max(np.abs(res.constr)):.6e}")
        print("\n最適パラメータ:")
        for k, v in params_phys.items():
            print(f"  {k:15s}: {v:.6e}")
    
    return params_phys, res

# =============================================================================
# Section 6: 可視化
# =============================================================================

def plot_results(params_opt: Dict, flc_points: List[Tuple[float, float]]):
    """結果の可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # FLC曲線
    beta_range = np.linspace(-0.8, 1.0, 100)
    Em_pred = [compute_flc_point(params_opt, b) for b in beta_range]
    
    ax1.plot(beta_range, Em_pred, 'b-', label='Fitted FLC', linewidth=2)
    
    # 実測点
    betas_data = [b for b, _ in flc_points]
    Ems_data = [e for _, e in flc_points]
    ax1.scatter(betas_data, Ems_data, c='red', s=100, 
               label='Target points', zorder=5)
    
    ax1.set_xlabel('Path ratio β')
    ax1.set_ylabel('Major strain limit')
    ax1.set_title('FLC Fitting Result')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 誤差
    errors = []
    for beta, Em_target in flc_points:
        Em_pred = compute_flc_point(params_opt, beta)
        error = abs(Em_pred - Em_target) / Em_target * 100
        errors.append(error)
    
    ax2.bar(range(len(errors)), errors)
    ax2.set_xlabel('Data point index')
    ax2.set_ylabel('Relative error (%)')
    ax2.set_title('Fitting Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n平均相対誤差: {np.mean(errors):.2f}%")
    print(f"最大相対誤差: {np.max(errors):.2f}%")

# =============================================================================
# Section 7: 実行例
# =============================================================================

if __name__ == "__main__":
    # FLC実測データ（SPCC）
    flc_points = [
        (-0.5, 0.38),   # 深絞り
        (-0.25, 0.32),  # 中間1
        (0.0, 0.25),    # 平面ひずみ
        (0.25, 0.23),   # 中間2
        (0.5, 0.20),    # 張出し
        (1.0, 0.18),    # 等二軸
    ]
    
    # physics_boundsも調整
    physics_bounds_v2 = {
        'K_scale': (0.5, 2.0),
        'K_scale_draw': (0.8, 1.5),
        'K_scale_plane': (0.75, 1.25),   # 少し広げる
        'K_scale_biax': (0.6, 1.1),      # 広げる
        'beta_A': (0.05, 0.3),
        'beta_bw': (0.25, 1.2),          # 下限を緩め
        'beta_A_pos': (0.05, 0.25),
    }

    # 初期値を改善して実行
    z0_better = get_better_initial_guess(flc_points, physics_bounds_v2)
    params_opt, res = solve_constrained_edr(
        flc_points,
        physics_bounds_v2,
        z0=z0_better,
        verbose=True
    )
    
    # 可視化
    plot_results(params_opt, flc_points)
