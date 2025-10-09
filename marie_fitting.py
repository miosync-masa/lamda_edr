"""
=============================================================================
Marie Antoinette Enhanced Fitting System
多様体ベースの破断判定・最適化システム

実際に使用している最適化パイプライン：
- Phase 0: 物理制約のみの教師なし学習
- 安全/危険多様体の構築
- 動的閾値による判定
- 相対距離ベースの高精度判定

Author: 飯泉真道 + 環
Date: 2025-01-19
=============================================================================
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import numpy as np
import optax
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# edr_coreから必要な機能をインポート
from edr_core import (
    MaterialParams,
    EDRParams,
    PressSchedule,
    ExpBinary,
    FLCPoint,
    schedule_to_jax_dict,
    mat_to_jax_dict,
    transform_params_jax,
    init_edr_params_jax,
    simulate_lambda_jax,
    triax_from_path_jax,
    smooth_signal_jax,
)

# manifold_libから多様体機能をインポート
from manifold_lib import (
    SafeManifold,
    DangerManifold,
    SafeManifoldBuilder,
    DangerManifoldBuilder,
    ManifoldAnalyzer,
    compute_gram,
    batch_gram_distance,
    RegularizationTerms,
)

# =============================================================================
# Section 1: 損失関数群（edr_fit.pyから移植・改良）
# =============================================================================

@jit
def loss_single_exp_jax(schedule_dict, mat_dict, edr_dict, failed):
    """単一実験の損失（基本版）"""
    res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
    
    Lambda_smooth = smooth_signal_jax(res["Lambda"], window_size=11)
    
    peak = jnp.max(Lambda_smooth)
    D_end = res["Damage"][-1]
    
    margin = 0.08
    Dcrit = 0.01
    delta = 0.03
    
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
        (peak - (edr_dict['Lambda_crit'] - delta))**2 * 3.0,
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
    """バッチ損失関数（基本版）"""
    edr_dict = transform_params_jax(raw_params)
    mat_dict = mat_to_jax_dict(mat)
    
    total_loss = 0.0
    for exp in exps:
        schedule_dict = schedule_to_jax_dict(exp.schedule)
        loss = loss_single_exp_jax(schedule_dict, mat_dict, edr_dict, exp.failed)
        total_loss += loss
    
    return total_loss / len(exps)

@jit
def predict_flc_from_sim_jax(beta, mat_dict, edr_dict, major_rate=0.6, duration=1.0):
    """v3: 符号変化検出＋線形補間＋未到達ケース対応"""
    
    # β依存でduration延長（張出し側で長く）
    duration_eff = duration * (1.0 + 0.4 * jax.nn.relu(beta - 0.5))
    
    # シミュレーション準備
    dt = 1e-3
    N = int(duration_eff/dt) + 1
    t = jnp.linspace(0, duration_eff, N)
    epsM = major_rate * t
    epsm = beta * epsM
    
    schedule_dict = {
        't': t,
        'eps_maj': epsM,
        'eps_min': epsm,
        'triax': jnp.full(N, triax_from_path_jax(beta)),
        'mu': jnp.full(N, 0.08),
        'pN': jnp.full(N, 200e6),
        'vslip': jnp.full(N, 0.02),
        'htc': jnp.full(N, 8000.0),
        'Tdie': jnp.full(N, 293.15),
        'contact': jnp.full(N, 1.0),
        'T0': 293.15
    }
    
    # シミュレーション実行
    res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
    Lambda_smooth = smooth_signal_jax(res["Lambda"], window_size=11)
    
    epsM_trimmed = epsM[:-1]
    Λcrit = edr_dict['Lambda_crit']
    
    # ===== 1. 符号変化検出による交点探索 =====
    diff = Lambda_smooth[:-1] - Λcrit
    # 符号が変わる点（負→正）を検出
    sign_change = jnp.maximum(-diff[:-1] * diff[1:], 0.0)
    
    # 符号変化点に重みを集中
    w = jnp.exp(50.0 * sign_change)
    w = w / (jnp.sum(w) + 1e-12)
    
    # 最大重み位置を取得
    idx = jnp.argmax(w)
    idx = jnp.minimum(idx, len(epsM_trimmed)-2)
    
    # ===== 2. 線形補間による正確な交点 =====
    Λ1 = Lambda_smooth[idx]
    Λ2 = Lambda_smooth[idx+1]
    e1 = epsM_trimmed[idx]
    e2 = epsM_trimmed[idx+1]
    
    # 線形補間係数
    frac = jnp.clip((Λcrit - Λ1) / (Λ2 - Λ1 + 1e-12), 0.0, 1.0)
    Em_cross = e1 + frac * (e2 - e1)
    
    # ===== 3. 未到達ケースのフォールバック =====
    # Λcritに最も近い点を探す
    distance_to_crit = jnp.abs(Lambda_smooth - Λcrit)
    nearest_idx = jnp.argmin(distance_to_crit)
    nearest_idx = jnp.minimum(nearest_idx, len(epsM_trimmed)-1)
    Em_nearest = epsM_trimmed[nearest_idx]
    
    # ===== 4. miss度合いの計算 =====
    max_Lambda = jnp.max(Lambda_smooth)
    # Λcritに届かないほど1.0に近づく
    miss = jax.nn.sigmoid((Λcrit - max_Lambda) * 8.0)
    
    # ===== 5. 交点と最近点のブレンド =====
    # 届いた場合(miss≈0): Em_cross使用
    # 届かない場合(miss≈1): Em_nearest使用
    Em = (1.0 - miss) * Em_cross + miss * Em_nearest
    
    # ===== 6. 安全装置 =====
    # NaN対策と物理的に妥当な範囲に制限
    Em = jnp.where(jnp.isnan(Em), epsM_trimmed[-1], Em)
    Em = jnp.clip(Em, 0.05, 0.8)
    
    em = beta * Em
    
    # missも返すことで、損失関数でペナルティ追加可能
    return Em, em, Lambda_smooth, miss

def loss_flc_true_jax(raw_params, flc_pts_data, mat_dict):
    """FLC損失関数"""
    edr_dict = transform_params_jax(raw_params)
    
    flc_err = 0.0
    for i in range(len(flc_pts_data['path_ratios'])):
        beta = flc_pts_data['path_ratios'][i]
        Em_gt = flc_pts_data['major_limits'][i]
        em_gt = flc_pts_data['minor_limits'][i]
        
        Em_pred, em_pred, _ = predict_flc_from_sim_jax(beta, mat_dict, edr_dict)
        
        # β依存の重み付け
        w = jnp.where(jnp.abs(beta - 0.5) < 0.1, 6.0,
                      jnp.where(jnp.abs(beta) < 0.1, 1.8, 1.0))
        
        flc_err += w * ((Em_pred - Em_gt)**2 + (em_pred - em_gt)**2)
    
    flc_err = flc_err / len(flc_pts_data['path_ratios'])
    
    # V字形状と滑らかさの正則化
    beta_batch = jnp.linspace(-0.6, 0.6, 21)
    Em_curve = []
    for beta in beta_batch:
        Em, _, _ = predict_flc_from_sim_jax(beta, mat_dict, edr_dict)
        Em_curve.append(Em)
    Em_curve = jnp.array(Em_curve)
    
    center_idx = len(beta_batch) // 2
    center = Em_curve[center_idx]
    valley_loss = 0.1 * jnp.mean(jnp.maximum(0.0, Em_curve - center))
    
    grad2 = jnp.diff(jnp.diff(Em_curve))
    smoothness_loss = 0.05 * jnp.mean(grad2**2) + 0.02 * jnp.mean(jnp.abs(grad2))
    
    valley_weight = jnp.clip(jnp.var(Em_curve), 0.05, 0.3)
    valley_loss = valley_weight * valley_loss
    
    return flc_err + valley_loss + smoothness_loss

# =============================================================================
# Section 2: 動的閾値計算と判定システム
# =============================================================================

@dataclass
class ThresholdConfig:
    """閾値設定"""
    safe_upper: float      # 安全側上限
    danger_lower: float    # 危険側下限
    optimal: float         # 最適分離点
    method: str           # 計算方法

def calculate_dynamic_thresholds(
    exps: List[ExpBinary],
    mat_dict: Dict,
    edr_dict: Dict,
    safe_manifold: SafeManifold,
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = True
) -> ThresholdConfig:
    """
    実験データから動的に閾値を計算
    """
    if weights is None:
        weights = {
            'tv': 0.1,
            'jump': 0.5,
            'topo': 0.1,
            'l1': 1e-3
        }
    
    analyzer = ManifoldAnalyzer(weights)
    
    safe_scores = []
    danger_scores = []
    
    for exp in exps:
        schedule_dict = schedule_to_jax_dict(exp.schedule)
        res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
        Lambda = res["Lambda"]
        
        score = float(jax.device_get(analyzer.compute_safety_score(Lambda, safe_manifold)))
        
        if exp.failed == 0:
            safe_scores.append(score)
        else:
            danger_scores.append(score)
    
    # 統計的に決定
    safe_mean = np.mean(safe_scores)
    safe_std = np.std(safe_scores)
    danger_mean = np.mean(danger_scores)
    danger_std = np.std(danger_scores)
    
    # 2σルールで閾値設定
    safe_upper = safe_mean + 2 * safe_std
    danger_lower = danger_mean - 2 * danger_std
    
    # Fisher判別による最適分離点
    if safe_std + danger_std > 0:
        optimal = (safe_mean * danger_std + danger_mean * safe_std) / (safe_std + danger_std)
    else:
        optimal = (safe_mean + danger_mean) / 2
    
    if verbose:
        print("\n" + "="*60)
        print(" 🎯 動的閾値計算")
        print("="*60)
        print(f"安全サンプル: mean={safe_mean:.4f}, std={safe_std:.4f}")
        print(f"破断サンプル: mean={danger_mean:.4f}, std={danger_std:.4f}")
        print(f"\n閾値設定:")
        print(f"  安全側上限 (mean+2σ): {safe_upper:.4f}")
        print(f"  危険側下限 (mean-2σ): {danger_lower:.4f}")
        print(f"  最適分離点 (Fisher): {optimal:.4f}")
    
    return ThresholdConfig(
        safe_upper=safe_upper,
        danger_lower=danger_lower,
        optimal=optimal,
        method="dynamic_2sigma"
    )

# =============================================================================
# Section 3: 相対距離判定（危険多様体も使用）
# =============================================================================

def dual_manifold_classification(
    Lambda: jnp.ndarray,
    safe_manifold: SafeManifold,
    danger_manifold: Optional[DangerManifold] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict:
    Lambda_smooth = smooth_signal_jax(Lambda, window_size=11)
    G_test = compute_gram(Lambda_smooth)
    
    # float変換を削除！JAX配列のまま
    safe_distances = batch_gram_distance(G_test, safe_manifold.grams)
    safe_dist = jnp.min(safe_distances)  # ← floatなし！
    
    result = {
        'safe_distance': safe_dist,  # JAX配列のまま
        'danger_distance': None,
        'relative_score': None,
        'prediction': None,
        'confidence': None
    }
    
    if danger_manifold is not None:
        danger_distances = batch_gram_distance(G_test, danger_manifold.grams)
        danger_dist = jnp.min(danger_distances)  # ← floatなし！
        result['danger_distance'] = danger_dist
        
        # 相対スコア（JAX配列のまま）
        relative_score = safe_dist / (safe_dist + danger_dist + 1e-8)
        result['relative_score'] = relative_score
        
        # 判定（JAX配列で）
        result['prediction'] = jnp.where(relative_score > 0.5, 1, 0)
        
        # 確信度（JAX配列で）
        result['confidence'] = jnp.abs(relative_score - 0.5) * 2
    
    return result

# =============================================================================
# Section 4: 多様体ベースの損失関数
# =============================================================================

def loss_binary_manifold(
    params,
    exps: List[ExpBinary],
    mat_dict: Dict,
    safe_manifold: SafeManifold,
    danger_manifold: Optional[DangerManifold] = None,
    thresholds: Optional[ThresholdConfig] = None,
    weights: Optional[Dict[str, float]] = None
):
    """
    多様体ベースのバイナリ損失関数（改良版）
    """
    if weights is None:
        weights = {
            'tv': 0.1,
            'jump': 0.5,
            'topo': 0.1,
            'l1': 1e-3
        }
    
    edr_dict = transform_params_jax(params)
    analyzer = ManifoldAnalyzer(weights)
    
    total_loss = 0.0
    
    if danger_manifold is not None:
        # 相対距離ベースの損失
        for exp in exps:
            schedule_dict = schedule_to_jax_dict(exp.schedule)
            res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
            Lambda = res["Lambda"]
            
            result = dual_manifold_classification(
                Lambda, safe_manifold, danger_manifold, weights
            )
            relative_score = result['relative_score']
            
            if exp.failed == 1:
                # 破断: relative_scoreは1に近いべき
                loss = (1.0 - relative_score)**2
            else:
                # 安全: relative_scoreは0に近いべき
                loss = relative_score**2
            
            total_loss += loss
    else:
        # 安全多様体のみの場合（従来版）
        if thresholds is None:
            # デフォルト閾値
            safe_threshold = 0.25
            danger_threshold = 0.35
        else:
            safe_threshold = thresholds.safe_upper
            danger_threshold = thresholds.danger_lower
        
        for exp in exps:
            schedule_dict = schedule_to_jax_dict(exp.schedule)
            res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
            Lambda = res["Lambda"]
            
            score = analyzer.compute_safety_score(Lambda, safe_manifold)
            
            if exp.failed == 1:
                loss = jnp.maximum(0.0, danger_threshold - score)**2
            else:
                loss = jnp.maximum(0.0, score - safe_threshold)**2
            
            total_loss += loss
    
    return total_loss / len(exps)

# =============================================================================
# Section 5: Phase 0 - 教師なし物理制約学習
# =============================================================================
def phase0_unsupervised_learning(
    mat_dict: Dict,
    flc_pts_data: Dict = None,  # 追加：FLC実験データ
    n_steps: int = 300,
    verbose: bool = True
) -> Tuple[Dict, List[float]]:
    """
    Phase 0: 物理制約のみでFLC面を事前学習
    """
    if verbose:
        print("\n" + "="*60)
        print(" 🎂 Phase 0: Unsupervised FLC Manifold Learning")
        print("="*60)
    
    # 実験データからV字深さ目標を計算
    target_v_ratio = None
    if flc_pts_data is not None:
        betas = flc_pts_data['path_ratios']
        majors = flc_pts_data['major_limits']
        
        # β=0に最も近いインデックス
        center_idx = jnp.argmin(jnp.abs(betas))
        # エッジの平均
        edge_avg = (majors[0] + majors[-1]) / 2
        # V字の深さ比率
        target_v_ratio = majors[center_idx] / edge_avg
        
        if verbose:
            print(f"  実験V字深さ目標: {float(target_v_ratio):.3f}")
    
    # 安定版FLC予測
    def predict_flc_stable(path_ratio, edr_dict):
        duration = 1.0
        major_rate = 0.6
        dt = 1e-3
        N = int(duration/dt) + 1
        t = jnp.linspace(0, duration, N)
        epsM = major_rate * t
        epsm = path_ratio * epsM
        
        schedule_dict = {
            't': t,
            'eps_maj': epsM,
            'eps_min': epsm,
            'triax': jnp.full(N, triax_from_path_jax(path_ratio)),
            'mu': jnp.full(N, 0.08),
            'pN': jnp.full(N, 200e6),
            'vslip': jnp.full(N, 0.02),
            'htc': jnp.full(N, 8000.0),
            'Tdie': jnp.full(N, 293.15),
            'contact': jnp.full(N, 1.0),
            'T0': 293.15
        }
        
        res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
        Lambda_smooth = smooth_signal_jax(res["Lambda"], window_size=11)
        
        exceed_mask = Lambda_smooth > edr_dict['Lambda_crit']
        first_exceed = jnp.argmax(exceed_mask)
        has_exceeded = jnp.any(exceed_mask)
        
        epsM_trimmed = epsM[:-1]
        Em = jnp.where(has_exceeded, epsM_trimmed[first_exceed], epsM_trimmed[-1])
        
        return Em
    
    # Phase 0損失関数
    def loss_phase0(raw_params):
        edr_dict = transform_params_jax(raw_params)
        
        beta_grid = jnp.linspace(-0.8, 0.8, 13)
        
        Em_grid = []
        for beta in beta_grid:
            Em = predict_flc_stable(beta, edr_dict)
            Em = jnp.where(jnp.isnan(Em), 0.3, Em)
            Em = jnp.clip(Em, 0.1, 0.8)
            Em_grid.append(Em)
        
        Em_array = jnp.array(Em_grid)
        
        # 物理制約
        monotonicity_loss = jnp.mean(jnp.maximum(0, -jnp.diff(jnp.abs(Em_array))))
        
        center = len(beta_grid) // 2
        left_branch = Em_array[:center]
        right_branch = Em_array[center:]
        convexity_loss = jnp.mean(jnp.maximum(0, jnp.diff(left_branch))) + \
                         jnp.mean(jnp.maximum(0, -jnp.diff(right_branch)))
        
        asymmetry_factor = jnp.clip(edr_dict['beta_A_pos'] / (edr_dict['beta_A'] + 1e-8), 0.5, 2.0)
        symmetry_target = Em_array[::-1] * asymmetry_factor
        symmetry_loss = 0.1 * jnp.mean((Em_array - symmetry_target)**2)
        
        grad2 = jnp.diff(jnp.diff(Em_array))
        smoothness_loss = 0.05 * jnp.mean(grad2**2)
        
        range_loss = jnp.mean(jnp.maximum(0, 0.1 - Em_array)**2) + \
                     jnp.mean(jnp.maximum(0, Em_array - 1.0)**2)
        
        # V字深さ損失（実験データがある場合）
        if target_v_ratio is not None:
            edge_avg = (Em_array[0] + Em_array[-1]) / 2
            v_depth = Em_array[center] / (edge_avg + 1e-8)
            depth_loss = 0.2 * (v_depth - target_v_ratio)**2
        else:
            depth_loss = 0.0
        
        total_loss = monotonicity_loss + convexity_loss + symmetry_loss + \
                    smoothness_loss + range_loss + depth_loss
        
        return jnp.where(jnp.isnan(total_loss), 1e10, total_loss)
    
    # 初期化と最適化
    params_phase0 = init_edr_params_jax()
    
    schedule_phase0 = optax.exponential_decay(
        init_value=3e-3,
        transition_steps=50,
        decay_rate=0.92
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_phase0)
    )
    
    opt_state = optimizer.init(params_phase0)
    grad_fn = jax.grad(loss_phase0)
    
    loss_history = []
    
    for step in range(n_steps):
        grads = grad_fn(params_phase0)
        updates, opt_state = optimizer.update(grads, opt_state, params_phase0)
        params_phase0 = optax.apply_updates(params_phase0, updates)
        
        if step % 100 == 0:
            loss = float(jax.device_get(loss_phase0(params_phase0)))
            loss_history.append(loss)
            if verbose:
                print(f"  Step {step:3d}: Physics Loss = {loss:.6f}")
    
    final_loss = float(jax.device_get(loss_phase0(params_phase0)))
    loss_history.append(final_loss)
    
    if verbose:
        print(f"\n  ✅ Phase 0完了: Physics Loss = {final_loss:.6f}")
    
    return params_phase0, loss_history

# =============================================================================
# Section 5.5: Phase 1 - 純粋なFLCフィッティング（追加！）
# =============================================================================

def phase1_flc_optimization(
    params_init,
    flc_pts_data: Dict,
    mat_dict: Dict,
    n_steps: int = 200,
    verbose: bool = True
) -> Tuple[Dict, List[float]]:
    """
    Phase 1: FLC専用最適化（純粋な相対誤差最小化版）
    """
    if verbose:
        print("\n" + "="*60)
        print(" 🎯 Phase 1: Direct Data Fitting (Relative Error)")
        print("="*60)
        print(f"  データ点数: {len(flc_pts_data['path_ratios'])}")
        print(f"  最適化ステップ: {n_steps}")
    
    # 純粋な相対誤差損失（重み付けなし！）
    def loss_relative_error(params):
        edr_dict = transform_params_jax(params)
        
        total_rel_error = 0.0
        for i in range(len(flc_pts_data['path_ratios'])):
            beta = flc_pts_data['path_ratios'][i]
            Em_gt = flc_pts_data['major_limits'][i]
            
            Em_pred, _, _ = predict_flc_from_sim_jax(beta, mat_dict, edr_dict)
            
            # 純粋な相対誤差（重み付け完全排除）
            rel_error = jnp.abs(Em_pred - Em_gt) / (Em_gt + 1e-8)
            total_rel_error += rel_error
        
        # 平均相対誤差のみ返す（正則化なし）
        avg_rel_error = total_rel_error / len(flc_pts_data['path_ratios'])
        
        # 最小限の滑らかさ制約のみ追加（0.01倍）
        beta_test = jnp.linspace(-0.6, 0.6, 11)
        Em_curve = []
        for b in beta_test:
            Em, _, _ = predict_flc_from_sim_jax(b, mat_dict, edr_dict)
            Em_curve.append(Em)
        Em_array = jnp.array(Em_curve)
        grad2 = jnp.diff(jnp.diff(Em_array))
        smoothness = 0.01 * jnp.mean(grad2**2)  # 超弱い正則化
        
        return avg_rel_error + smoothness
    
    # オプティマイザ設定（相対誤差用にチューニング）
    schedule = optax.exponential_decay(
        init_value=5e-3,  # より積極的に
        transition_steps=30,
        decay_rate=0.93
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # クリップ緩和
        optax.adam(learning_rate=schedule)
    )
    
    opt_state = optimizer.init(params_init)
    params = params_init
    grad_fn = jax.grad(loss_relative_error)
    
    loss_history = []
    rel_error_history = []
    best_rel_error = float('inf')
    best_params = params_init
    
    for step in range(n_steps):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if step % 20 == 0 or step == n_steps - 1:
            loss = float(jax.device_get(loss_relative_error(params)))
            loss_history.append(loss)
            
            # 実際の相対誤差を計算
            edr_dict = transform_params_jax(params)
            total_error = 0.0
            for i in range(len(flc_pts_data['path_ratios'])):
                beta = flc_pts_data['path_ratios'][i]
                Em_gt = flc_pts_data['major_limits'][i]
                Em_pred, _, _ = predict_flc_from_sim_jax(beta, mat_dict, edr_dict)
                error = abs(float(jax.device_get(Em_pred)) - Em_gt) / Em_gt
                total_error += error
            
            avg_error = total_error / len(flc_pts_data['path_ratios'])
            rel_error_history.append(avg_error)
            
            # ベストパラメータを相対誤差で判定
            if avg_error < best_rel_error:
                best_rel_error = avg_error
                best_params = params
            
            if verbose:
                print(f"  Step {step:3d}: Loss = {loss:.6f}, "
                      f"相対誤差 = {avg_error*100:.2f}%")
    
    if verbose:
        print(f"\n  ✅ Phase 1完了!")
        print(f"  最良相対誤差: {best_rel_error*100:.2f}%")
        
        # 各点の詳細表示
        edr_dict = transform_params_jax(best_params)
        print("\n  各点の誤差:")
        for i in range(len(flc_pts_data['path_ratios'])):
            beta = flc_pts_data['path_ratios'][i]
            Em_gt = flc_pts_data['major_limits'][i]
            Em_pred, _, _ = predict_flc_from_sim_jax(beta, mat_dict, edr_dict)
            Em_pred_val = float(jax.device_get(Em_pred))
            error = abs(Em_pred_val - Em_gt) / Em_gt * 100
            print(f"    β={beta:5.2f}: GT={Em_gt:.3f}, Pred={Em_pred_val:.3f}, "
                  f"誤差={error:.1f}%")
    
    return best_params, rel_error_history

# =============================================================================
# Section 6: Phase 1.5B - 制約付き多様体最適化
# =============================================================================

def phase_15b_manifold_optimization(
    params_init,
    flc_pts_data: Dict,
    exps: List[ExpBinary],
    mat_dict: Dict,
    safe_manifold: SafeManifold,
    danger_manifold: Optional[DangerManifold] = None,
    flc_target: float = None,
    n_steps: int = 500,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Phase 1.5B: FLC制約付き多様体ベースBinary最適化
    """
    # 動的閾値計算
    edr_dict_init = transform_params_jax(params_init)
    thresholds = calculate_dynamic_thresholds(
        exps, mat_dict, edr_dict_init, safe_manifold, verbose=verbose
    )
    
    # FLC目標値の設定
    if flc_target is None:
        flc_target = float(jax.device_get(
            loss_flc_true_jax(params_init, flc_pts_data, mat_dict)
        ))
    
    if verbose:
        print("\n" + "="*60)
        print(" 🎂 Phase 1.5B: 制約付き多様体最適化")
        print("="*60)
        print(f"  FLC制約: < {flc_target * 1.03:.6f} (3%許容)")
        print(f"  目的: Binary最小化（多様体ベース）")
        
        if danger_manifold is not None:
            print("  ⚠️  危険多様体も使用（相対距離判定）")
    
    # 制約付き損失関数
    def loss_constrained(params):
        flc_loss = loss_flc_true_jax(params, flc_pts_data, mat_dict)
        
        bin_loss = loss_binary_manifold(
            params, exps, mat_dict, safe_manifold, 
            danger_manifold, thresholds
        )
        
        # FLC閾値と動的重み付け
        flc_threshold = flc_target * 1.03
        flc_margin = (flc_loss - flc_threshold) / (flc_threshold * 0.01)
        w_transition = jax.nn.sigmoid(flc_margin * 5.0)
        w_flc = 0.1 + 0.8 * w_transition
        w_bin = 1.0 - w_flc
        
        return w_flc * flc_loss + w_bin * bin_loss, flc_loss, bin_loss
    
    # オプティマイザ
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.3),
        optax.adamw(learning_rate=1e-3, weight_decay=1e-5)
    )
    
    opt_state = optimizer.init(params_init)
    params = params_init
    
    grad_fn = jax.grad(lambda p: loss_constrained(p)[0])
    
    history = {
        'total': [],
        'flc': [],
        'binary': []
    }
    
    for step in range(n_steps):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if step % 100 == 0 and verbose:
            total_loss, flc_loss, bin_loss = loss_constrained(params)
            
            # JAXトレース値を具体値に変換
            total_val = float(jax.device_get(total_loss))
            flc_val = float(jax.device_get(flc_loss))
            bin_val = float(jax.device_get(bin_loss))
            
            history['total'].append(total_val)
            history['flc'].append(flc_val)
            history['binary'].append(bin_val)
            
            print(f"  Step {step:3d}: Total = {total_val:.6f} "
                  f"(FLC: {flc_val:.6f}, Binary: {bin_val:.6f})")
        
    if verbose:
        print(f"\n  ✅ Phase 1.5B完了！")
    
    return params, history

# =============================================================================
# Section 7: 検証と評価
# =============================================================================
def evaluate_binary_classification(
    exps: List[ExpBinary],
    mat_dict: Dict,
    edr_dict: Dict,
    safe_manifold: SafeManifold,
    danger_manifold: Optional[DangerManifold] = None,
    verbose: bool = True
) -> Dict:
    """
    バイナリ分類の性能評価
    """
    if verbose:
        print("\n" + "="*60)
        print(" 📊 Binary Classification Evaluation")
        print("="*60)
    
    # ManifoldAnalyzerを初期化！
    analyzer = ManifoldAnalyzer()
    
    correct = 0
    results = []
    
    for i, exp in enumerate(exps):
        schedule_dict = schedule_to_jax_dict(exp.schedule)
        res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
        Lambda = res["Lambda"]
        
        # 判定処理
        if danger_manifold is not None:
            # 相対距離判定
            classification = dual_manifold_classification(
                Lambda, safe_manifold, danger_manifold
            )
            pred = int(jax.device_get(classification['prediction']))
            conf = float(jax.device_get(classification['confidence']))
            
            if verbose:
                print(f"\nExp{i} ({exp.label}):")
                print(f"  真値: {'破断' if exp.failed == 1 else '安全'}")
                print(f"  予測: {'破断' if pred == 1 else '安全'}")
                print(f"  安全距離: {float(jax.device_get(classification['safe_distance'])):.4f}")
                print(f"  危険距離: {float(jax.device_get(classification['danger_distance'])):.4f}")
                print(f"  相対スコア: {float(jax.device_get(classification['relative_score'])):.4f}")
                print(f"  確信度: {conf:.2%}")
        else:
            # 安全多様体のみの判定（修正！）
            safety_score = analyzer.compute_safety_score(Lambda, safe_manifold)
            safety_score_val = float(jax.device_get(safety_score))
            
            # 動的閾値を使用（safe_manifoldの統計から）
            threshold = safe_manifold.metadata.get('fisher_threshold', 0.5)
            pred = 1 if safety_score_val > threshold else 0
            
            if verbose:
                print(f"\nExp{i} ({exp.label}):")
                print(f"  真値: {'破断' if exp.failed == 1 else '安全'}")
                print(f"  予測: {'破断' if pred == 1 else '安全'}")
                print(f"  安全スコア: {safety_score_val:.4f}")
                print(f"  閾値: {threshold:.4f}")
            
            classification = {
                'prediction': pred,
                'safety_score': safety_score_val,
                'threshold': threshold
            }
        
        # 正解判定
        if pred == exp.failed:
            correct += 1
            if verbose:
                print("  ✓ 正解！")
        else:
            if verbose:
                print("  ✗ 不正解")
        
        results.append({
            'exp': exp,
            'classification': classification,
            'correct': pred == exp.failed
        })
    
    accuracy = correct / len(exps) * 100
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"最終精度: {accuracy:.2f}%")
        print(f"正解数: {correct}/{len(exps)}")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(exps),
        'results': results
    }
    
# =============================================================================
# Section 8: 結果の保存と可視化
# =============================================================================

import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

def save_optimization_results(
    edr_params_raw: Dict,
    edr_params: EDRParams,
    flc_results: Optional[Dict] = None,
    binary_results: Optional[Dict] = None,
    history: Optional[Dict] = None,
    output_dir: str = "./results",
    prefix: str = None
) -> str:
    """
    最適化結果を保存
    
    Args:
        edr_params_raw: 生のパラメータ（JAX dict）
        edr_params: EDRParamsデータクラス
        flc_results: FLCフィッティング結果
        binary_results: バイナリ分類結果
        history: 最適化履歴
        output_dir: 出力ディレクトリ
        prefix: ファイル名プレフィックス
    
    Returns:
        保存先パス
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix is None:
        prefix = f"edr_marie_{timestamp}"
    
    # EDRパラメータを辞書形式に
    edr_dict = {
        'V0': edr_params.V0,
        'av': edr_params.av,
        'ad': edr_params.ad,
        'chi': edr_params.chi,
        'K_scale': edr_params.K_scale,
        'triax_sens': edr_params.triax_sens,
        'Lambda_crit': edr_params.Lambda_crit,
        'K_scale_draw': edr_params.K_scale_draw,
        'K_scale_plane': edr_params.K_scale_plane,
        'K_scale_biax': edr_params.K_scale_biax,
        'beta_A': edr_params.beta_A,
        'beta_bw': edr_params.beta_bw,
        'beta_A_pos': edr_params.beta_A_pos,
    }
    
    # 結果をまとめる
    results = {
        'timestamp': timestamp,
        'edr_params': edr_dict,
        'edr_params_raw': {k: float(v) for k, v in edr_params_raw.items()},
        'flc_results': flc_results,
        'binary_results': binary_results,
        'history': history
    }
    
    # JSON形式で保存（人間が読める）
    json_path = os.path.join(output_dir, f"{prefix}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Pickle形式でも保存（完全な再現性）
    pkl_path = os.path.join(output_dir, f"{prefix}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'edr_params_raw': edr_params_raw,
            'edr_params_obj': edr_params,
            'full_results': results
        }, f)
    
    print(f"\n✅ 結果を保存しました:")
    print(f"  JSON: {json_path}")
    print(f"  Pickle: {pkl_path}")
    
    return json_path

def plot_flc_fitting_results(
    edr_dict: Dict,
    mat_dict: Dict,
    flc_pts_data: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    FLCフィッティング結果を可視化
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # FLC曲線を計算
    beta_range = np.linspace(-0.8, 0.8, 41)
    Em_pred = []
    em_pred = []
    
    for beta in beta_range:
        Em, em, _ = predict_flc_from_sim_jax(beta, mat_dict, edr_dict)
        Em_pred.append(float(jax.device_get(Em)))
        em_pred.append(float(jax.device_get(em)))
    
    # プロット1: ひずみ空間のFLC
    ax1.plot(em_pred, Em_pred, 'b-', linewidth=2, label='Predicted FLC')
    
    if flc_pts_data is not None:
        ax1.scatter(flc_pts_data['minor_limits'], 
                   flc_pts_data['major_limits'],
                   c='red', s=100, marker='o', label='Experimental', zorder=5)
    
    ax1.set_xlabel('Minor Strain ε₂')
    ax1.set_ylabel('Major Strain ε₁')
    ax1.set_title('Forming Limit Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-0.3, 0.3)
    ax1.set_ylim(0, 0.5)
    
    # プロット2: β vs 限界主ひずみ
    ax2.plot(beta_range, Em_pred, 'b-', linewidth=2, label='Predicted')
    
    if flc_pts_data is not None:
        ax2.scatter(flc_pts_data['path_ratios'],
                   flc_pts_data['major_limits'],
                   c='red', s=100, marker='o', label='Experimental', zorder=5)
    
    ax2.set_xlabel('Path Ratio β')
    ax2.set_ylabel('Major Strain Limit')
    ax2.set_title('V-shape Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(0, 0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  FLC図を保存: {output_path}")
    
    return fig

def generate_report(
    edr_params: EDRParams,
    flc_error: Optional[float] = None,
    binary_accuracy: Optional[float] = None,
    phase0_loss: Optional[float] = None,
    final_losses: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> str:
    """
    最適化結果のレポート生成
    """
    report = []
    report.append("="*80)
    report.append(" EDR Parameter Fitting Report")
    report.append(" Operation Marie Antoinette Enhanced")
    report.append("="*80)
    report.append(f"\n生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # EDRパラメータ
    report.append("【EDRパラメータ】")
    report.append(f"  V0 (基準凝集エネルギー): {edr_params.V0:.2e} Pa")
    report.append(f"  av (空孔影響係数): {edr_params.av:.2e}")
    report.append(f"  ad (転位影響係数): {edr_params.ad:.2e}")
    report.append(f"  chi (摩擦発熱分配率): {edr_params.chi:.4f}")
    report.append(f"  K_scale (総量スケール): {edr_params.K_scale:.4f}")
    report.append(f"  triax_sens (三軸度感度): {edr_params.triax_sens:.4f}")
    report.append(f"  Lambda_crit (臨界Λ): {edr_params.Lambda_crit:.4f}")
    report.append("")
    report.append("  経路別スケール:")
    report.append(f"    深絞り (draw): {edr_params.K_scale_draw:.4f}")
    report.append(f"    平面ひずみ (plane): {edr_params.K_scale_plane:.4f}")
    report.append(f"    等二軸 (biax): {edr_params.K_scale_biax:.4f}")
    report.append("")
    report.append("  FLC形状パラメータ:")
    report.append(f"    beta_A (谷の深さ): {edr_params.beta_A:.4f}")
    report.append(f"    beta_bw (谷の幅): {edr_params.beta_bw:.4f}")
    report.append(f"    beta_A_pos (等二軸側): {edr_params.beta_A_pos:.4f}")
    report.append("")
    
    # 性能指標
    if any([flc_error, binary_accuracy, phase0_loss, final_losses]):
        report.append("【性能指標】")
        
        if phase0_loss is not None:
            report.append(f"  Phase 0 物理損失: {phase0_loss:.6f}")
        
        if flc_error is not None:
            report.append(f"  FLC平均誤差: {flc_error:.4f} ({flc_error*100:.2f}%)")
            if flc_error < 0.05:
                report.append("    → ✅ 優秀 (<5%)")
            elif flc_error < 0.10:
                report.append("    → 🟡 良好 (<10%)")
            else:
                report.append("    → 🔴 要改善 (>10%)")
        
        if binary_accuracy is not None:
            report.append(f"  バイナリ分類精度: {binary_accuracy:.2f}%")
            if binary_accuracy >= 95:
                report.append("    → ✅ 優秀 (≥95%)")
            elif binary_accuracy >= 90:
                report.append("    → 🟡 良好 (≥90%)")
            else:
                report.append("    → 🔴 要改善 (<90%)")
        
        if final_losses:
            report.append("")
            report.append("  最終損失:")
            for key, value in final_losses.items():
                report.append(f"    {key}: {value:.6f}")
    
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # ファイルに保存
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"  レポート保存: {output_path}")
    
    return report_text

def load_optimization_results(filepath: str) -> Dict:
    """
    保存された結果を読み込み
    
    Args:
        filepath: JSONまたはPickleファイルパス
    
    Returns:
        結果辞書
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"未対応のファイル形式: {filepath}")

# =============================================================================
# エクスポート
# =============================================================================

__all__ = [
    # 損失関数
    'loss_single_exp_jax',
    'loss_fn_jax',
    'loss_flc_true_jax',
    'loss_binary_manifold',
    
    # 閾値計算
    'ThresholdConfig',
    'calculate_dynamic_thresholds',
    
    # 判定システム
    'dual_manifold_classification',
    
    # 最適化
    'phase0_unsupervised_learning',
    'phase1_flc_optimization',
    'phase_15b_manifold_optimization',

    # 結果保存
    'save_optimization_results',
    'plot_flc_fitting_results',
    'generate_report',
    'load_optimization_results',
    
    # 評価
    'evaluate_binary_classification',
]
