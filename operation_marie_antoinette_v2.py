"""
=============================================================================
Operation Marie Antoinette v2.0: Inverse Problem Data Augmentation
"データがないなら作ればいいじゃない！"

逆問題×多様体学習×イベントグラムによるゼロショット破断判定

【edr_fit.py完全統合版 v2.0】
✅ edr_fit.pyの機能をフル活用
✅ 重複コード完全削減
✅ 依存関係の明確化
✅ メンテナンス性向上
✅ パフォーマンス最適化

Author: 飯泉真道 + 環
Date: 2025-10-07 (v2.0統合版)
=============================================================================
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# edr_fit.pyから必要な機能を一括import（完全統合版）
# =============================================================================
try:
    from edr_fit import (
        # === コアシミュレーション ===
        simulate_lambda_jax,
        
        # === データ変換関数 ===
        schedule_to_jax_dict,
        mat_to_jax_dict,
        transform_params_jax,
        edr_dict_to_dataclass,
        
        # === 物理計算ヘルパー ===
        triax_from_path_jax,
        equiv_strain_rate_jax,
        smooth_signal_jax,
        
        # === 損失関数 ===
        loss_flc_true_jax,
        loss_fn_jax,
        loss_single_exp_jax,
        
        # === パラメータ管理 ===
        init_edr_params_jax,
        
        # === データクラス ===
        MaterialParams,
        EDRParams,
        PressSchedule,
        ExpBinary,
        FLCPoint,
    )
    EDR_FIT_AVAILABLE = True
    print("✓ edr_fit.py完全統合: 全機能インポート成功")
except ImportError as e:
    EDR_FIT_AVAILABLE = False
    print(f"⚠️  edr_fit.pyのインポートエラー: {e}")
    print("   operation_marie_antoinette.pyはedr_fit.pyに依存します")
    print("   単体実行の場合は、edr_fit.pyと同じディレクトリに配置してください")

# optax（最適化ライブラリ）
try:
    import optax
    OPTAX_AVAILABLE = True
except ImportError:
    OPTAX_AVAILABLE = False
    print("⚠️  optaxが未インストール: pip install optax")

# =============================================================================
# Section 1: Gram行列とイベント表現
# =============================================================================

@jit
def compute_gram(x):
    """
    Gram行列計算（スケール不変版）
    
    Args:
        x: [time] または [batch, time]
    Returns:
        Gram行列 [time, time]
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
def gram_distance(G1, G2):
    """2つのGram行列間の距離"""
    return jnp.sum((G1 - G2)**2)

# =============================================================================
# Section 2: 正則化項（物理的異常検出）
# =============================================================================

@jit
def compute_tv(Lambda):
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

@jit
def compute_jump_penalty(Lambda, k=2.5):
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
        # [paths, time-1]
        d = jnp.abs(jnp.diff(Lambda, axis=1))
        threshold = jnp.mean(d, axis=1, keepdims=True) + k * jnp.std(d, axis=1, keepdims=True)
        return jnp.sum(jnp.maximum(0.0, d - threshold))

@jit
def compute_topo_penalty(Lambda):
    """
    位相連続性正則化
    Phase遷移の滑らかさを評価
    """
    if Lambda.ndim == 1:
        # phase_k = atan2(Lambda[k+1], Lambda[k])
        phase = jnp.arctan2(Lambda[1:], Lambda[:-1])
        # 位相差
        dphase = jnp.diff(phase)
        # [-π, π]に正規化
        dphase = (dphase + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.sum(dphase**2)
    else:
        # [paths, time-1]
        phase = jnp.arctan2(Lambda[:, 1:], Lambda[:, :-1])
        dphase = jnp.diff(phase, axis=1)
        dphase = (dphase + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.sum(dphase**2)

@jit
def compute_l1_norm(Lambda):
    """L1正則化（スパース性促進）"""
    return jnp.sum(jnp.abs(Lambda))

# =============================================================================
# Section 3: Phase 0 教師なし学習（物理制約のみ）
# =============================================================================

def phase0_unsupervised_learning(
    mat_dict: Dict,
    n_steps: int = 300,
    verbose: bool = True
):
    """
    Phase 0: Unsupervised FLC Pretraining
    実験データなしで、物理制約のみでEDRパラメータを事前学習
    
    Args:
        mat_dict: 材料パラメータ
        n_steps: 最適化ステップ数
        verbose: 進捗表示
    
    Returns:
        params_phase0: Phase 0で学習したパラメータ（raw）
        loss_history: 損失履歴
    """
    if not OPTAX_AVAILABLE:
        raise ImportError("optaxが必要です: pip install optax")
    
    if not EDR_FIT_AVAILABLE:
        raise ImportError("edr_fit.pyが必要です")
    
    if verbose:
        print("\n" + "="*60)
        print(" 🎂 Phase 0: Unsupervised FLC Manifold Learning")
        print("="*60)
        print("  物理制約のみでFLC面を事前学習")
        print("  実験データ不要！")
    
    # Phase 0用の安定版FLC予測
    def predict_flc_stable(path_ratio, edr_dict):
        """Phase 0用の安定版FLC予測（argmax使用）"""
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
        
        # 安定版：argmax使用
        exceed_mask = Lambda_smooth > edr_dict['Lambda_crit']
        first_exceed = jnp.argmax(exceed_mask)
        has_exceeded = jnp.any(exceed_mask)
        
        epsM_trimmed = epsM[:-1]
        Em = jnp.where(has_exceeded, epsM_trimmed[first_exceed], epsM_trimmed[-1])
        
        return Em
    
    # Phase 0損失関数
    def loss_phase0(raw_params):
        """物理制約のみでFLC面を学習"""
        edr_dict = transform_params_jax(raw_params)
        
        beta_grid = jnp.linspace(-0.8, 0.8, 13)
        
        # 各βでの仮想FLC限界を計算
        Em_grid = []
        for beta in beta_grid:
            Em = predict_flc_stable(beta, edr_dict)
            Em = jnp.where(jnp.isnan(Em), 0.3, Em)
            Em = jnp.clip(Em, 0.1, 0.8)
            Em_grid.append(Em)
        
        Em_array = jnp.array(Em_grid)
        Em_array = jnp.where(jnp.isnan(Em_array), 0.3, Em_array)
        
        # 物理制約1: 単調性
        monotonicity_loss = jnp.mean(jnp.maximum(0, -jnp.diff(jnp.abs(Em_array))))
        monotonicity_loss = jnp.where(jnp.isnan(monotonicity_loss), 0.0, monotonicity_loss)
        
        # 物理制約2: 凸性（V字形状）
        center = len(beta_grid) // 2
        left_branch = Em_array[:center]
        right_branch = Em_array[center:]
        
        convexity_loss = jnp.mean(jnp.maximum(0, jnp.diff(left_branch))) + \
                         jnp.mean(jnp.maximum(0, -jnp.diff(right_branch)))
        convexity_loss = jnp.where(jnp.isnan(convexity_loss), 0.0, convexity_loss)
        
        # 物理制約3: 対称性（破れを許容）
        asymmetry_factor = jnp.clip(edr_dict['beta_A_pos'] / (edr_dict['beta_A'] + 1e-8), 0.5, 2.0)
        symmetry_target = Em_array[::-1] * asymmetry_factor
        symmetry_loss = 0.1 * jnp.mean((Em_array - symmetry_target)**2)
        symmetry_loss = jnp.where(jnp.isnan(symmetry_loss), 0.0, symmetry_loss)
        
        # 物理制約4: 平滑性
        grad2 = jnp.diff(jnp.diff(Em_array))
        smoothness_loss = 0.05 * jnp.mean(grad2**2)
        smoothness_loss = jnp.where(jnp.isnan(smoothness_loss), 0.0, smoothness_loss)
        
        # 物理制約5: 合理的な範囲
        range_loss = jnp.mean(jnp.maximum(0, 0.1 - Em_array)**2) + \
                     jnp.mean(jnp.maximum(0, Em_array - 1.0)**2)
        range_loss = jnp.where(jnp.isnan(range_loss), 0.0, range_loss)
        
        total_loss = monotonicity_loss + convexity_loss + symmetry_loss + \
                    smoothness_loss + range_loss
        
        total_loss = jnp.where(jnp.isnan(total_loss), 1e10, total_loss)
        
        return total_loss
    
    # Phase 0初期化
    params_phase0 = init_edr_params_jax()
    
    # Phase 0最適化
    schedule_phase0 = optax.exponential_decay(
        init_value=3e-3,
        transition_steps=50,
        decay_rate=0.92
    )
    
    optimizer_phase0 = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_phase0)
    )
    
    opt_state_phase0 = optimizer_phase0.init(params_phase0)
    grad_fn_phase0 = jax.grad(loss_phase0)
    
    loss_history = []
    
    for step in range(n_steps):
        grads = grad_fn_phase0(params_phase0)
        updates, opt_state_phase0 = optimizer_phase0.update(grads, opt_state_phase0, params_phase0)
        params_phase0 = optax.apply_updates(params_phase0, updates)
        
        if step % 100 == 0:
            loss = float(loss_phase0(params_phase0))
            loss_history.append(loss)
            if verbose:
                print(f"  Step {step:3d}: Physics Loss = {loss:.6f}")
    
    final_loss = float(loss_phase0(params_phase0))
    loss_history.append(final_loss)
    
    if verbose:
        print(f"\n  ✅ Phase 0完了: Physics Loss = {final_loss:.6f}")
        print("  物理的に妥当なFLC面の初期化完了")
    
    return params_phase0, loss_history

# =============================================================================
# Section 4: 安全多様体の構築（edr_fit.py完全統合版）
# =============================================================================

def build_safe_manifold(
    mat_dict: Dict,
    edr_dict: Dict,
    simulate_fn=None,
    n_beta: int = 15,
    n_mu: int = 5,
    n_pN: int = 5,
    duration: float = 0.6,  # ← 引数として追加！デフォルトは0.6に
    safety_margin: float = 0.85,
    verbose: bool = True
):
    """
    安全なΛ(t)軌道を大量生成して多様体を構築
    
    Args:
        mat_dict: 材料パラメータ（JAX dict）
        edr_dict: EDRパラメータ（JAX dict）
        simulate_fn: シミュレーション関数（Noneならedr_fit.pyのものを使用）
        n_beta: β方向のサンプル数
        n_mu: 摩擦係数のサンプル数
        n_pN: 接触圧のサンプル数
        safety_margin: Lambda_crit * safety_margin 以下を安全と判定
        verbose: 進捗表示
    
    Returns:
        safe_manifold: {
            'lambdas': [n_safe, time],
            'grams': [n_safe, time, time],
            'conditions': [n_safe] の条件リスト
        }
    """
    # デフォルトシミュレータはedr_fit.pyのものを使用
    if simulate_fn is None:
        if not EDR_FIT_AVAILABLE:
            raise ImportError("simulate_lambda_jaxが利用できません。edr_fit.pyをインポートしてください")
        simulate_fn = simulate_lambda_jax
    
    if verbose:
        print("\n" + "="*60)
        print(" 🎂 Operation Marie Antoinette: 安全多様体構築")
        print("="*60)
        print(f"  パラメータ空間探索:")
        print(f"    β: {n_beta}点")
        print(f"    μ: {n_mu}点")
        print(f"    pN: {n_pN}点")
        print(f"    合計: {n_beta * n_mu * n_pN}軌道を生成")
    
    safe_lambdas = []
    safe_grams = []
    safe_conditions = []
    
    Lambda_crit = edr_dict['Lambda_crit']
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
                
                # 安全なスケジュール生成（低負荷）
                duration = 0.8
                major_rate = 0.4  # 低ひずみ速度
                dt = 1e-3
                N = int(duration/dt) + 1
                t = jnp.linspace(0, duration, N)
                epsM = major_rate * t
                epsm = beta * epsM
                
                # 三軸度計算（edr_fit.pyの関数を使用）
                triax_val = triax_from_path_jax(beta)
                
                schedule_dict = {
                    't': t,
                    'eps_maj': epsM,
                    'eps_min': epsm,
                    'triax': jnp.full(N, triax_val),
                    'mu': jnp.full(N, float(mu)),
                    'pN': jnp.full(N, float(pN)),
                    'vslip': jnp.full(N, 0.015),
                    'htc': jnp.full(N, 8000.0),
                    'Tdie': jnp.full(N, 293.15),
                    'contact': jnp.full(N, 1.0),
                    'T0': 293.15
                }
                
                # シミュレーション実行（edr_fit.pyの関数を使用）
                res = simulate_fn(schedule_dict, mat_dict, edr_dict)
                Lambda = res["Lambda"]
                
                # スムージング（edr_fit.pyの関数を使用）
                Lambda_smooth = smooth_signal_jax(Lambda, window_size=11)
                
                # 安全判定
                peak_Lambda = float(jnp.max(Lambda_smooth))
                
                if peak_Lambda < safety_threshold:
                    # 安全軌道として採用
                    safe_lambdas.append(np.array(Lambda_smooth))
                    
                    # Gram行列計算
                    G = compute_gram(Lambda_smooth)
                    safe_grams.append(np.array(G))
                    
                    # 条件記録
                    safe_conditions.append({
                        'beta': float(beta),
                        'mu': float(mu),
                        'pN': float(pN),
                        'peak_Lambda': peak_Lambda
                    })
                    
                    safe_count += 1
                
                if verbose and count % 20 == 0:
                    print(f"    進捗: {count}/{n_beta*n_mu*n_pN}, "
                          f"安全軌道: {safe_count}本")
    
    if verbose:
        print(f"\n  ✅ 安全多様体構築完了！")
        print(f"    生成軌道: {count}本")
        print(f"    安全軌道: {safe_count}本 ({safe_count/count*100:.1f}%)")
        print(f"    Gram行列: {safe_count} × [{len(safe_lambdas[0])} × {len(safe_lambdas[0])}]")
    
    return {
        'lambdas': jnp.array(safe_lambdas),
        'grams': jnp.array(safe_grams),
        'conditions': safe_conditions,
        'n_safe': safe_count
    }

# =============================================================================
# Section 4: 安全スコア計算
# =============================================================================

@jit
def compute_safety_score(
    Lambda: jnp.ndarray,
    safe_grams: jnp.ndarray,
    weights: Dict[str, float]
):
    """
    安全スコア計算（低いほど安全）
    
    Args:
        Lambda: テスト軌道 [time]
        safe_grams: 安全多様体のGram行列群 [n_safe, time, time]
        weights: 正則化の重み
    
    Returns:
        score: 安全スコア（低い=安全、高い=危険）
    """
    # スムージング適用（edr_fit.pyの関数を使用）
    Lambda_smooth = smooth_signal_jax(Lambda, window_size=11)
    
    # テストのGram行列
    G_test = compute_gram(Lambda_smooth)
    
    # 最も近い安全軌道との距離
    distances = vmap(lambda G_safe: gram_distance(G_test, G_safe))(safe_grams)
    min_dist = jnp.min(distances)
    
    # 正則化項
    tv = compute_tv(Lambda_smooth)
    jump = compute_jump_penalty(Lambda_smooth, k=2.5)
    topo = compute_topo_penalty(Lambda_smooth)
    l1 = compute_l1_norm(Lambda_smooth)
    
    # 総合スコア
    score = (min_dist + 
             weights['tv'] * tv + 
             weights['jump'] * jump + 
             weights['topo'] * topo + 
             weights['l1'] * l1)
    
    return score

# =============================================================================
# Section 5: 多様体ベースのバイナリ損失（edr_fit.py統合版）
# =============================================================================

def loss_binary_manifold(
    params,
    exps: List[ExpBinary],
    mat_dict: Dict,
    safe_manifold: Dict,
    simulate_fn=None,
    weights: Optional[Dict[str, float]] = None
):
    """
    安全多様体ベースのバイナリ損失関数
    
    Args:
        params: EDRパラメータ（raw）
        exps: 実験データリスト
        mat_dict: 材料パラメータ
        safe_manifold: 安全多様体
        simulate_fn: シミュレーション関数（Noneならedr_fit.pyのものを使用）
        weights: 正則化の重み
    
    Returns:
        loss: バイナリ損失
    """
    # デフォルト重み
    if weights is None:
        weights = {
            'tv': 0.1,
            'jump': 0.5,
            'topo': 0.1,
            'l1': 1e-3
        }
    
    # デフォルトシミュレータ
    if simulate_fn is None:
        if not EDR_FIT_AVAILABLE:
            raise ImportError("simulate_lambda_jaxが利用できません")
        simulate_fn = simulate_lambda_jax
    
    # パラメータ変換（edr_fit.pyの関数を使用）
    edr_dict = transform_params_jax(params)
    safe_grams = safe_manifold['grams']
    
    total_loss = 0.0
    
    for exp in exps:
        # スケジュール変換（edr_fit.pyの関数を使用）
        schedule_dict = schedule_to_jax_dict(exp.schedule)
        res = simulate_fn(schedule_dict, mat_dict, edr_dict)
        Lambda = res["Lambda"]
        
        # 安全スコア計算
        score = compute_safety_score(Lambda, safe_grams, weights)
        
        # 閾値設定（調整可能）
        safe_threshold = 0.3
        danger_threshold = 0.5
        
        if exp.failed == 1:
            # 破断サンプル: スコアが高いべき（danger_threshold以上）
            loss = jnp.maximum(0.0, danger_threshold - score)**2
        else:
            # 安全サンプル: スコアが低いべき（safe_threshold以下）
            loss = jnp.maximum(0.0, score - safe_threshold)**2
        
        total_loss += loss
    
    return total_loss / len(exps)

# =============================================================================
# Section 6: Phase 1.5Bへの統合（制約付き多様体最適化）
# =============================================================================

def phase_15b_manifold_optimization(
    params_init,
    flc_pts_data: Dict,
    exps: List[ExpBinary],
    mat_dict: Dict,
    safe_manifold: Dict,
    simulate_fn=None,
    flc_target: float = None,
    n_steps: int = 500,
    verbose: bool = True
):
    """
    Phase 1.5B: FLC制約付き多様体ベースBinary最適化
    
    Args:
        params_init: 初期パラメータ
        flc_pts_data: FLCデータ
        exps: バイナリ実験データ
        mat_dict: 材料パラメータ
        safe_manifold: 安全多様体
        simulate_fn: シミュレーション関数（Noneならedr_fit.pyのものを使用）
        flc_target: FLC目標値（Noneなら現在値を使用）
        n_steps: ステップ数
        verbose: 進捗表示
    
    Returns:
        params_final: 最適化後パラメータ
        history: 最適化履歴
    """
    if not OPTAX_AVAILABLE:
        raise ImportError("optaxが必要です: pip install optax")
    
    # デフォルトシミュレータ
    if simulate_fn is None:
        if not EDR_FIT_AVAILABLE:
            raise ImportError("simulate_lambda_jaxが利用できません")
        simulate_fn = simulate_lambda_jax
    
    # FLC目標値の設定
    if flc_target is None:
        # 現在のFLC損失を目標値として設定
        flc_target = float(loss_flc_true_jax(params_init, flc_pts_data, mat_dict))
    
    if verbose:
        print("\n" + "="*60)
        print(" 🎂 Phase 1.5B: 制約付き多様体最適化")
        print("="*60)
        print(f"  FLC制約: < {flc_target * 1.03:.6f} (3%許容)")
        print(f"  目的: Binary最小化（多様体ベース）")
        print(f"  ステップ数: {n_steps}")
    
    # 正則化の重み
    manifold_weights = {
        'tv': 0.1,
        'jump': 0.5,
        'topo': 0.1,
        'l1': 1e-3
    }
    
    # 制約付き損失関数
    def loss_constrained(params):
        # FLC損失（edr_fit.pyの関数を使用）
        flc_loss = loss_flc_true_jax(params, flc_pts_data, mat_dict)
        
        # 多様体ベースのバイナリ損失
        bin_loss = loss_binary_manifold(
            params, exps, mat_dict, safe_manifold, 
            simulate_fn, manifold_weights
        )
        
        # FLC閾値
        flc_threshold = flc_target * 1.03
        
        # 動的重み付け
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
    
    # 勾配関数
    grad_fn = jax.grad(lambda p: loss_constrained(p)[0])
    
    # 履歴
    history = {
        'total': [],
        'flc': [],
        'binary': []
    }
    
    # 最適化ループ
    for step in range(n_steps):
        # 勾配計算
        grads = grad_fn(params)
        
        # パラメータ更新
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # 進捗表示
        if step % 100 == 0 and verbose:
            total_loss, flc_loss, bin_loss = loss_constrained(params)
            history['total'].append(float(total_loss))
            history['flc'].append(float(flc_loss))
            history['binary'].append(float(bin_loss))
            
            print(f"  Step {step:3d}: Total = {float(total_loss):.6f} "
                  f"(FLC: {float(flc_loss):.6f}, Binary: {float(bin_loss):.6f})")
    
    if verbose:
        print(f"\n  ✅ Phase 1.5B完了！")
    
    return params, history

# =============================================================================
# Section 8: 完全パイプライン（Phase 0 → 多様体 → Binary最適化）
# =============================================================================

def marie_antoinette_pipeline(
    mat_dict: Dict,
    exps: Optional[List[ExpBinary]] = None,
    flc_pts_data: Optional[Dict] = None,
    use_phase0: bool = True,
    phase0_steps: int = 300,
    manifold_params: Optional[Dict] = None,
    phase15b_steps: int = 500,
    verbose: bool = True
):
    """
    Operation Marie Antoinette 完全パイプライン
    
    Phase 0（オプション）: 教師なし物理制約学習
      ↓
    安全多様体構築: 大量の安全軌道生成
      ↓
    Phase 1.5B（オプション）: FLC制約付きBinary最適化
    
    Args:
        mat_dict: 材料パラメータ
        exps: バイナリ実験データ（Phase 1.5B用、オプション）
        flc_pts_data: FLCデータ（Phase 1.5B用、オプション）
        use_phase0: Phase 0を実行するか
        phase0_steps: Phase 0のステップ数
        manifold_params: 多様体構築パラメータ（n_beta, n_mu, n_pN）
        phase15b_steps: Phase 1.5Bのステップ数
        verbose: 進捗表示
    
    Returns:
        results: {
            'params': 最終パラメータ,
            'safe_manifold': 安全多様体,
            'phase0_history': Phase 0履歴,
            'phase15b_history': Phase 1.5B履歴
        }
    """
    if verbose:
        print("="*80)
        print(" 🎂 Operation Marie Antoinette - Complete Pipeline")
        print("="*80)
        print("\n「データがないなら作ればいいじゃない！」")
        print("  Phase 0: 教師なし学習 → 多様体構築 → Binary最適化\n")
    
    results = {}
    
    # ===========================
    # Phase 0: 教師なし学習
    # ===========================
    if use_phase0:
        params_phase0, phase0_history = phase0_unsupervised_learning(
            mat_dict,
            n_steps=phase0_steps,
            verbose=verbose
        )
        edr_dict = transform_params_jax(params_phase0)
        results['phase0_history'] = phase0_history
        results['params_phase0'] = params_phase0
        
        if verbose:
            print(f"\n✅ Phase 0完了！物理的に妥当なパラメータ獲得")
    else:
        # デフォルトパラメータ使用
        params_phase0 = init_edr_params_jax()
        edr_dict = transform_params_jax(params_phase0)
        results['phase0_history'] = None
        
        if verbose:
            print(f"\n⏭️  Phase 0スキップ: デフォルトパラメータ使用")
    
    # ===========================
    # 安全多様体構築
    # ===========================
    if manifold_params is None:
        manifold_params = {
            'n_beta': 15,
            'n_mu': 5,
            'n_pN': 5
        }
    
    if verbose:
        print(f"\n{'='*60}")
        print(" 安全多様体構築")
        print(f"{'='*60}")
    
    safe_manifold = build_safe_manifold(
        mat_dict, edr_dict,
        n_beta=manifold_params['n_beta'],
        n_mu=manifold_params['n_mu'],
        n_pN=manifold_params['n_pN'],
        duration=0.6,
        safety_margin=0.85,
        verbose=verbose
    )
    
    results['safe_manifold'] = safe_manifold
    results['params_manifold'] = params_phase0  # 多様体構築時のパラメータ
    
    if verbose:
        print(f"\n✅ 安全多様体構築完了！{safe_manifold['n_safe']}本の安全軌道")
    
    # ===========================
    # Phase 1.5B: Binary最適化（オプション）
    # ===========================
    if exps is not None and flc_pts_data is not None:
        if verbose:
            print(f"\n{'='*60}")
            print(" Phase 1.5B: 制約付きBinary最適化")
            print(f"{'='*60}")
        
        params_final, phase15b_history = phase_15b_manifold_optimization(
            params_phase0,
            flc_pts_data,
            exps,
            mat_dict,
            safe_manifold,
            n_steps=phase15b_steps,
            verbose=verbose
        )
        
        results['params_final'] = params_final
        results['phase15b_history'] = phase15b_history
        
        if verbose:
            print(f"\n✅ Phase 1.5B完了！Binary最適化成功")
    else:
        results['params_final'] = params_phase0
        results['phase15b_history'] = None
        
        if verbose:
            print(f"\n⏭️  Phase 1.5Bスキップ: 実験データなし")
    
    # ===========================
    # サマリー
    # ===========================
    if verbose:
        print(f"\n{'='*80}")
        print(" 🎂 Pipeline Complete!")
        print(f"{'='*80}")
        
        if use_phase0:
            print(f"\n✅ Phase 0: Physics Loss = {results['phase0_history'][-1]:.6f}")
        
        print(f"✅ 安全多様体: {safe_manifold['n_safe']}本の軌道")
        
        if results['phase15b_history'] is not None:
            print(f"✅ Phase 1.5B: Binary最適化完了")
            print(f"   FLC Loss: {results['phase15b_history']['flc'][-1]:.6f}")
            print(f"   Binary Loss: {results['phase15b_history']['binary'][-1]:.6f}")
        
        print(f"\n🎉 Operation Marie Antoinette大成功！")
    
    return results

# =============================================================================
# Section 9: ユーティリティ（edr_fit.py統合版）
# =============================================================================

def visualize_safe_manifold(safe_manifold: Dict, output_path: str = None):
    """安全多様体の可視化"""
    import matplotlib.pyplot as plt
    
    lambdas = safe_manifold['lambdas']
    conditions = safe_manifold['conditions']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (1) 全軌道プロット
    ax = axes[0, 0]
    for i, lam in enumerate(lambdas[:50]):  # 最初の50本
        ax.plot(lam, alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Λ(t)')
    ax.set_title(f'Safe Trajectories (n={len(lambdas)})')
    ax.grid(True, alpha=0.3)
    
    # (2) βごとの分布
    ax = axes[0, 1]
    betas = [c['beta'] for c in conditions]
    peaks = [c['peak_Lambda'] for c in conditions]
    sc = ax.scatter(betas, peaks, c=peaks, cmap='viridis', alpha=0.6)
    ax.set_xlabel('β (path ratio)')
    ax.set_ylabel('Peak Λ')
    ax.set_title('Peak Λ vs β')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Peak Λ')
    
    # (3) Gram行列のサンプル
    ax = axes[1, 0]
    G_sample = safe_manifold['grams'][0]
    im = ax.imshow(G_sample, cmap='RdBu_r', aspect='auto')
    ax.set_xlabel('Time')
    ax.set_ylabel('Time')
    ax.set_title('Sample Gram Matrix')
    plt.colorbar(im, ax=ax)
    
    # (4) 条件分布
    ax = axes[1, 1]
    mus = [c['mu'] for c in conditions]
    pNs = [c['pN']/1e6 for c in conditions]  # MPa
    sc = ax.scatter(mus, pNs, c=peaks, cmap='viridis', alpha=0.6)
    ax.set_xlabel('μ (friction)')
    ax.set_ylabel('pN (MPa)')
    ax.set_title('Condition Space Coverage')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Peak Λ')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  可視化保存: {output_path}")
    
    plt.show()
    
    return fig

def analyze_safety_scores(
    exps: List[ExpBinary],
    mat_dict: Dict,
    edr_dict: Dict,
    safe_manifold: Dict,
    simulate_fn=None,
    weights: Optional[Dict[str, float]] = None
):
    """
    全実験データの安全スコア分析
    """
    # デフォルト重み
    if weights is None:
        weights = {
            'tv': 0.1,
            'jump': 0.5,
            'topo': 0.1,
            'l1': 1e-3
        }
    
    # デフォルトシミュレータ
    if simulate_fn is None:
        if not EDR_FIT_AVAILABLE:
            raise ImportError("simulate_lambda_jaxが利用できません")
        simulate_fn = simulate_lambda_jax
    
    print("\n" + "="*60)
    print(" 🎂 Safety Score Analysis")
    print("="*60)
    
    safe_grams = safe_manifold['grams']
    
    for i, exp in enumerate(exps):
        # スケジュール変換（edr_fit.pyの関数を使用）
        schedule_dict = schedule_to_jax_dict(exp.schedule)
        res = simulate_fn(schedule_dict, mat_dict, edr_dict)
        Lambda = res["Lambda"]
        
        # 安全スコア計算
        score = float(compute_safety_score(Lambda, safe_grams, weights))
        
        # 各成分の寄与
        Lambda_smooth = smooth_signal_jax(Lambda, window_size=11)
        G_test = compute_gram(Lambda_smooth)
        distances = [float(gram_distance(G_test, G_safe)) for G_safe in safe_grams]
        min_dist = min(distances)
        
        tv = float(compute_tv(Lambda_smooth))
        jump = float(compute_jump_penalty(Lambda_smooth))
        topo = float(compute_topo_penalty(Lambda_smooth))
        l1 = float(compute_l1_norm(Lambda_smooth))
        
        status = "破断" if exp.failed == 1 else "安全"
        
        print(f"\nExp{i} ({exp.label}, {status}):")
        print(f"  Total Score: {score:.4f}")
        print(f"    Gram距離: {min_dist:.4f}")
        print(f"    TV      : {tv:.4f} (× {weights['tv']:.2f})")
        print(f"    Jump    : {jump:.4f} (× {weights['jump']:.2f})")
        print(f"    Topo    : {topo:.4f} (× {weights['topo']:.2f})")
        print(f"    L1      : {l1:.4f} (× {weights['l1']:.4f})")

# =============================================================================
# メイン実行例（完全パイプライン版）
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" 🎂 Operation Marie Antoinette v2.0")
    print(" 「データがないなら作ればいいじゃない！」")
    print(" （edr_fit.py完全統合版 + Phase 0パイプライン）")
    print("="*80)
    
    print("\n逆問題×多様体学習による破断判定システム")
    print("  - Phase 0: 教師なし物理制約学習")
    print("  - 安全多様体の自動構築")
    print("  - Gram行列ベースのパターン認識")
    print("  - 物理的異常検出（TV, Jump, Topo正則化）")
    print("  - ゼロショット運用可能")
    print("  ✅ edr_fit.pyの全機能を活用")
    
    if EDR_FIT_AVAILABLE:
        print("\n✅ edr_fit.py統合完了")
        print("  利用可能な機能:")
        print("    - simulate_lambda_jax")
        print("    - smooth_signal_jax")
        print("    - triax_from_path_jax")
        print("    - schedule_to_jax_dict")
        print("    - mat_to_jax_dict")
        print("    - loss_flc_true_jax")
        print("    - Phase 0教師なし学習")
        print("    - その他全機能")
        
        print("\n" + "="*60)
        print(" デモ実行")
        print("="*60)
        
        # 材料パラメータ
        from edr_fit import generate_demo_experiments, generate_demo_flc
        
        mat = MaterialParams()
        mat_dict = mat_to_jax_dict(mat)
        
        # デモデータ
        exps = generate_demo_experiments()
        flc_data_list = generate_demo_flc()
        
        # FLCデータをdict化
        flc_pts_data = {
            'path_ratios': jnp.array([p.path_ratio for p in flc_data_list]),
            'major_limits': jnp.array([p.major_limit for p in flc_data_list]),
            'minor_limits': jnp.array([p.minor_limit for p in flc_data_list])
        }
        
        print(f"\n✓ Material params: OK")
        print(f"✓ Binary experiments: {len(exps)}")
        print(f"✓ FLC points: {len(flc_data_list)}")
        
        # 完全パイプライン実行
        results = marie_antoinette_pipeline(
            mat_dict=mat_dict,
            exps=exps,
            flc_pts_data=flc_pts_data,
            use_phase0=True,
            phase0_steps=300,
            manifold_params={'n_beta': 15, 'n_mu': 5, 'n_pN': 5},
            phase15b_steps=500,
            verbose=True
        )
        
        # 結果可視化
        print("\n" + "="*60)
        print(" 結果可視化")
        print("="*60)
        
        visualize_safe_manifold(
            results['safe_manifold'],
            output_path='safe_manifold_demo.png'
        )
        
        # 安全スコア分析
        edr_dict_final = transform_params_jax(results['params_final'])
        
        analyze_safety_scores(
            exps, mat_dict, edr_dict_final,
            results['safe_manifold']
        )
        
        print("\n" + "="*80)
        print(" 🎉 デモ実行完了！")
        print("="*80)
        
    else:
        print("\n⚠️  edr_fit.pyが見つかりません")
        print("  同じディレクトリにedr_fit.pyを配置してください")
