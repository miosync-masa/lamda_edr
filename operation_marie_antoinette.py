"""
=============================================================================
Operation Marie Antoinette: Inverse Problem Data Augmentation
"データがないなら作ればいいじゃない！"

逆問題×多様体学習×イベントグラムによるゼロショット破断判定
=============================================================================
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

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
# Section 3: 安全多様体の構築
# =============================================================================

def build_safe_manifold(
    mat_dict: Dict,
    edr_dict: Dict,
    simulate_fn,
    n_beta: int = 15,
    n_mu: int = 5,
    n_pN: int = 5,
    safety_margin: float = 0.85,
    verbose: bool = True
):
    """
    安全なΛ(t)軌道を大量生成して多様体を構築
    
    Args:
        mat_dict: 材料パラメータ（JAX dict）
        edr_dict: EDRパラメータ（JAX dict）
        simulate_fn: シミュレーション関数
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
                
                # 三軸度計算
                from edr_fit_fixed import triax_from_path_jax
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
                
                # シミュレーション実行
                res = simulate_fn(schedule_dict, mat_dict, edr_dict)
                Lambda = res["Lambda"]
                
                # 安全判定
                peak_Lambda = float(jnp.max(Lambda))
                
                if peak_Lambda < safety_threshold:
                    # 安全軌道として採用
                    safe_lambdas.append(np.array(Lambda))
                    
                    # Gram行列計算
                    G = compute_gram(Lambda)
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
    # テストのGram行列
    G_test = compute_gram(Lambda)
    
    # 最も近い安全軌道との距離
    distances = vmap(lambda G_safe: gram_distance(G_test, G_safe))(safe_grams)
    min_dist = jnp.min(distances)
    
    # 正則化項
    tv = compute_tv(Lambda)
    jump = compute_jump_penalty(Lambda, k=2.5)
    topo = compute_topo_penalty(Lambda)
    l1 = compute_l1_norm(Lambda)
    
    # 総合スコア
    score = (min_dist + 
             weights['tv'] * tv + 
             weights['jump'] * jump + 
             weights['topo'] * topo + 
             weights['l1'] * l1)
    
    return score

# =============================================================================
# Section 5: 多様体ベースのバイナリ損失
# =============================================================================

def loss_binary_manifold(
    params,
    exps,
    mat_dict,
    safe_manifold: Dict,
    simulate_fn,
    weights: Dict[str, float]
):
    """
    安全多様体ベースのバイナリ損失関数
    
    Args:
        params: EDRパラメータ（raw）
        exps: 実験データリスト
        mat_dict: 材料パラメータ
        safe_manifold: 安全多様体
        simulate_fn: シミュレーション関数
        weights: 正則化の重み
    
    Returns:
        loss: バイナリ損失
    """
    from edr_fit_fixed import transform_params_jax, schedule_to_jax_dict
    
    edr_dict = transform_params_jax(params)
    safe_grams = safe_manifold['grams']
    
    total_loss = 0.0
    
    for exp in exps:
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
    flc_pts_data,
    exps,
    mat_dict,
    safe_manifold: Dict,
    simulate_fn,
    flc_target: float,
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
        simulate_fn: シミュレーション関数
        flc_target: FLC目標値
        n_steps: ステップ数
        verbose: 進捗表示
    
    Returns:
        params_final: 最適化後パラメータ
        history: 最適化履歴
    """
    import optax
    from edr_fit_fixed import loss_flc_true_jax
    
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
        flc_loss = loss_flc_true_jax(params, flc_pts_data, mat_dict)
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
# Section 7: ユーティリティ
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
    exps,
    mat_dict,
    edr_dict,
    safe_manifold: Dict,
    simulate_fn,
    weights: Dict[str, float]
):
    """
    全実験データの安全スコア分析
    """
    from edr_fit_fixed import schedule_to_jax_dict
    
    print("\n" + "="*60)
    print(" 🎂 Safety Score Analysis")
    print("="*60)
    
    safe_grams = safe_manifold['grams']
    
    for i, exp in enumerate(exps):
        schedule_dict = schedule_to_jax_dict(exp.schedule)
        res = simulate_fn(schedule_dict, mat_dict, edr_dict)
        Lambda = res["Lambda"]
        
        # 安全スコア計算
        score = float(compute_safety_score(Lambda, safe_grams, weights))
        
        # 各成分の寄与
        G_test = compute_gram(Lambda)
        distances = [float(gram_distance(G_test, G_safe)) for G_safe in safe_grams]
        min_dist = min(distances)
        
        tv = float(compute_tv(Lambda))
        jump = float(compute_jump_penalty(Lambda))
        topo = float(compute_topo_penalty(Lambda))
        l1 = float(compute_l1_norm(Lambda))
        
        status = "破断" if exp.failed == 1 else "安全"
        
        print(f"\nExp{i} ({exp.label}, {status}):")
        print(f"  Total Score: {score:.4f}")
        print(f"    Gram距離: {min_dist:.4f}")
        print(f"    TV      : {tv:.4f} (× {weights['tv']:.2f})")
        print(f"    Jump    : {jump:.4f} (× {weights['jump']:.2f})")
        print(f"    Topo    : {topo:.4f} (× {weights['topo']:.2f})")
        print(f"    L1      : {l1:.4f} (× {weights['l1']:.4f})")

# =============================================================================
# メイン実行例
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" 🎂 Operation Marie Antoinette")
    print(" 「データがないなら作ればいいじゃない！」")
    print("="*80)
    
    print("\n逆問題×多様体学習による破断判定システム")
    print("  - 安全多様体の自動構築")
    print("  - Gram行列ベースのパターン認識")
    print("  - 物理的異常検出（TV, Jump, Topo正則化）")
    print("  - ゼロショット運用可能")
    
    print("\n✅ モジュールのインポート完了")
    print("  edr_fit_fixed.py から以下を利用:")
    print("    - simulate_lambda_jax()")
    print("    - transform_params_jax()")
    print("    - その他のヘルパー関数")
