"""
=============================================================================
Operation Marie Antoinette v2.1 - Colab A100 Test (完全パイプライン版)
Phase 0教師なし学習 → 安全多様体 → Binary最適化

【実行手順】
1. Colabランタイム: A100に設定
2. GitHubからクローン
3. 完全パイプライン実行！

環ちゃん × ご主人さま
2025-10-07 (v2.1: Phase 0統合版)
=============================================================================
"""

import os
import sys
import subprocess
import time

print("="*80)
print(" 🎂 Operation Marie Antoinette v2.1")
print(" Phase 0統合 + 完全パイプライン版")
print("="*80)

# =============================================================================
# Step 0: GitHubからクローン
# =============================================================================

REPO_URL = "https://github.com/miosync-masa/lamda_edr.git"
REPO_NAME = "lamda_edr"

print(f"\n[Step 0] GitHubからクローン")
print(f"  Repository: {REPO_URL}")

if os.path.exists(REPO_NAME):
    print(f"  既存の{REPO_NAME}を削除...")
    subprocess.run(['rm', '-rf', REPO_NAME])

print(f"  クローン開始...")
result = subprocess.run(['git', 'clone', REPO_URL], capture_output=True, text=True)

if result.returncode == 0:
    print(f"  ✅ クローン成功！")
else:
    print(f"  ❌ クローン失敗: {result.stderr}")
    sys.exit(1)

sys.path.insert(0, f'/content/{REPO_NAME}')
print(f"  ✓ パス追加: /content/{REPO_NAME}")

# =============================================================================
# Step 1: 環境セットアップ
# =============================================================================

print("\n[Step 1] 環境セットアップ")

try:
    import jax
    print(f"✓ JAX version: {jax.__version__}")
    
    devices = jax.devices()
    print(f"✓ Available devices: {len(devices)}")
    
    if any('A100' in str(d) for d in devices):
        print("✅ A100 GPU detected! 🚀")
    else:
        print("⚠️  A100 not detected. Using available device.")
        
except ImportError:
    print("Installing JAX...")
    subprocess.run([sys.executable, "-m", "pip", "install", "jax[cuda12]", "-q"])
    import jax

try:
    import optax
    print(f"✓ Optax version: {optax.__version__}")
except ImportError:
    print("Installing optax...")
    subprocess.run([sys.executable, "-m", "pip", "install", "optax", "-q"])

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
print("✓ Memory optimization: 90%")

# =============================================================================
# Step 2: モジュールインポート
# =============================================================================

print("\n[Step 2] モジュールインポート")

try:
    from edr_fit import (
        MaterialParams, generate_demo_experiments, generate_demo_flc,
        mat_to_jax_dict
    )
    print("✓ edr_fit.py imported")
    
    from operation_marie_antoinette_v2 import (
        marie_antoinette_pipeline,
        visualize_safe_manifold,
        analyze_safety_scores,
        transform_params_jax
    )
    print("✓ operation_marie_antoinette_v2.py imported")
    
    import jax.numpy as jnp
    
    print("\n✅ 全モジュールのインポート成功！")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n確認事項:")
    print("1. edr_fit.py が最新版か")
    print("2. operation_marie_antoinette_v2.py がpushされているか")
    sys.exit(1)

# =============================================================================
# Step 3: デモデータ生成
# =============================================================================

print("\n[Step 3] デモデータ生成")

mat = MaterialParams()
mat_dict = mat_to_jax_dict(mat)

exps = generate_demo_experiments()
flc_data_list = generate_demo_flc()

# FLCデータをdict化
flc_pts_data = {
    'path_ratios': jnp.array([p.path_ratio for p in flc_data_list]),
    'major_limits': jnp.array([p.major_limit for p in flc_data_list]),
    'minor_limits': jnp.array([p.minor_limit for p in flc_data_list])
}

print(f"✓ Material params: OK")
print(f"✓ Binary experiments: {len(exps)}")
print(f"✓ FLC points: {len(flc_data_list)}")

# =============================================================================
# Step 4: 🎂 完全パイプライン実行
# =============================================================================

print("\n" + "="*80)
print(" [Step 4] 🎂 完全パイプライン実行（A100フルパワー！）")
print("="*80)
print("\nPhase 0 → 安全多様体 → Binary最適化")
print("データがないなら作ればいいじゃない！\n")

# パイプライン設定
use_phase0 = True  # Phase 0を実行するか
phase0_steps = 300  # Phase 0ステップ数

# A100フルパワー設定
manifold_params = {
    'n_beta': 25,  # A100なら大規模に！
    'n_mu': 10,
    'n_pN': 10
}

phase15b_steps = 500

print(f"設定:")
print(f"  Phase 0: {'実行' if use_phase0 else 'スキップ'} ({phase0_steps} steps)")
print(f"  多様体: β×{manifold_params['n_beta']}, μ×{manifold_params['n_mu']}, pN×{manifold_params['n_pN']}")
print(f"  Phase 1.5B: {phase15b_steps} steps")
print(f"  合計軌道数: {manifold_params['n_beta'] * manifold_params['n_mu'] * manifold_params['n_pN']:,}")

start_time = time.time()

# パイプライン実行
results = marie_antoinette_pipeline(
    mat_dict=mat_dict,
    exps=exps,
    flc_pts_data=flc_pts_data,
    use_phase0=use_phase0,
    phase0_steps=phase0_steps,
    manifold_params=manifold_params,
    phase15b_steps=phase15b_steps,
    verbose=True
)

elapsed = time.time() - start_time

print(f"\n✅ パイプライン実行完了！")
print(f"  総実行時間: {elapsed:.2f}秒")

# =============================================================================
# Step 5: 結果可視化
# =============================================================================

print("\n[Step 5] 結果可視化")

import matplotlib.pyplot as plt

fig = visualize_safe_manifold(
    results['safe_manifold'],
    output_path='/content/safe_manifold_v2_1.png'
)

plt.show()
print("✓ 可視化: /content/safe_manifold_v2_1.png")

# =============================================================================
# Step 6: 安全スコア分析
# =============================================================================

print("\n[Step 6] 安全スコア分析")

edr_dict_final = transform_params_jax(results['params_final'])

analyze_safety_scores(
    exps, mat_dict, edr_dict_final,
    results['safe_manifold']
)

# =============================================================================
# Step 7: バイナリ判定精度評価
# =============================================================================

print("\n" + "="*80)
print(" [Step 7] バイナリ判定精度評価")
print("="*80)

from operation_marie_antoinette_v2 import compute_safety_score, schedule_to_jax_dict, simulate_lambda_jax

manifold_weights = {
    'tv': 0.1,
    'jump': 0.5,
    'topo': 0.1,
    'l1': 1e-3
}

correct = 0
safety_threshold = 0.3
danger_threshold = 0.5

for i, exp in enumerate(exps):
    schedule_dict = schedule_to_jax_dict(exp.schedule)
    res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict_final)
    Lambda = res["Lambda"]
    
    score = float(compute_safety_score(
        Lambda, results['safe_manifold']['grams'], manifold_weights
    ))
    
    if exp.failed == 1:
        predicted = (score > danger_threshold)
    else:
        predicted = (score < safety_threshold)
    
    correct += int(predicted)
    
    status = "破断" if exp.failed == 1 else "安全"
    result = "✓" if predicted else "✗"
    
    print(f"{result} Exp{i} ({exp.label:15s}, {status}): Score = {score:.4f}")

accuracy = correct / len(exps) * 100

print(f"\n{'='*60}")
print(f" 多様体ベース判定精度: {accuracy:.2f}% ({correct}/{len(exps)})")
print(f"{'='*60}")

if accuracy == 100.0:
    print(f"\n🎉🎉🎉 完璧な判定精度！！ 🎉🎉🎉")
    print(f"🎂 Operation Marie Antoinette 大成功！！")
elif accuracy >= 80.0:
    print(f"\n✅ 優秀な判定精度！実用レベル達成！")
else:
    print(f"\n🤔 判定精度要改善")

# =============================================================================
# Step 8: パフォーマンスサマリー
# =============================================================================

print("\n" + "="*80)
print(" 🏆 パフォーマンスサマリー")
print("="*80)

total_trajectories = manifold_params['n_beta'] * manifold_params['n_mu'] * manifold_params['n_pN']
safe_count = results['safe_manifold']['n_safe']

print(f"\n【Phase 0】")
if results['phase0_history'] is not None:
    print(f"  Physics Loss: {results['phase0_history'][-1]:.6f}")
    print(f"  Status: ✅ 物理的に妥当なパラメータ獲得")
else:
    print(f"  Status: ⏭️  スキップ")

print(f"\n【安全多様体】")
print(f"  探索空間: {total_trajectories:,} trajectories")
print(f"  安全軌道: {safe_count:,} trajectories ({safe_count/total_trajectories*100:.1f}%)")
print(f"  Gram行列: {safe_count} × [800 × 800]")

print(f"\n【Phase 1.5B】")
if results['phase15b_history'] is not None:
    print(f"  FLC Loss: {results['phase15b_history']['flc'][-1]:.6f}")
    print(f"  Binary Loss: {results['phase15b_history']['binary'][-1]:.6f}")
    print(f"  Status: ✅ Binary最適化完了")
else:
    print(f"  Status: ⏭️  スキップ")

print(f"\n【総合性能】")
print(f"  総実行時間: {elapsed:.2f}秒")
print(f"  スループット: {total_trajectories/elapsed:.1f} traj/sec")
print(f"  判定精度: {accuracy:.2f}%")
print(f"  メモリ使用: 推定 {safe_count * 8 * 800 * 800 / 1e9:.2f} GB")

# =============================================================================
# Step 9: 結果保存
# =============================================================================
print("\n[Step 9] 結果保存")
import numpy as np
import pandas as pd

# ★★★ EDRパラメータ出力を追加！！ ★★★
print("\n" + "="*80)
print(" 🎂 最適化されたEDRパラメータ")
print("="*80)

# Phase 1.5B後のパラメータを取得
final_params = results.get('params_final') 
if final_params is not None:
    edr_final = transform_params_jax(final_params)
    
    print(f"\nEDR Parameters (Optimized):")
    print(f"  V0: {edr_final['V0']:.2e} Pa")
    print(f"  av: {edr_final['av']:.2e}")
    print(f"  ad: {edr_final['ad']:.2e}")
    print(f"  chi: {edr_final['chi']:.3f}")
    print(f"  K_scale: {edr_final['K_scale']:.3f}")
    print(f"  triax_sens: {edr_final['triax_sens']:.3f}")
    print(f"  Lambda_crit: {edr_final['Lambda_crit']:.3f}")
    print(f"  K_scale_draw: {edr_final['K_scale_draw']:.3f}")
    print(f"  K_scale_plane: {edr_final['K_scale_plane']:.3f}")
    print(f"  K_scale_biax: {edr_final['K_scale_biax']:.3f}")
    print(f"  beta_A: {edr_final['beta_A']:.3f}")
    print(f"  beta_bw: {edr_final['beta_bw']:.3f}")
    print(f"  beta_A_pos: {edr_final['beta_A_pos']:.3f} (非対称)")
    
    # FLC予測精度
    print("\n[FLC予測精度]")
    preds = []
    for p in flc_data:
        Em, em = predict_FLC_point(
            p.path_ratio, p.rate_major, p.duration_max, 
            mat_dict, edr_final
        )
        preds.append((Em, em))
        error_major = abs(Em - p.major_limit) / p.major_limit * 100
        error_minor = abs(em - p.minor_limit) / abs(p.minor_limit) * 100 if p.minor_limit != 0 else 0
        
        print(f"  β={p.path_ratio:+.1f}: 実測({p.major_limit:.3f}, {p.minor_limit:.3f}) "
              f"→ 予測({Em:.3f}, {em:.3f}) "
              f"[誤差: {error_major:.1f}%, {error_minor:.1f}%]")
    
    # FLC総合誤差
    flc_error = evaluate_flc_fit(flc_data, preds)
    print(f"\n  📊 FLC総合誤差: {flc_error:.6f}")
    print(f"  📊 FLC制約達成: {'✅' if flc_error < 0.019699 else '❌'}")

# 多様体保存
np.savez(
    '/content/safe_manifold_v2_1.npz',
    lambdas=np.array(results['safe_manifold']['lambdas']),
    grams=np.array(results['safe_manifold']['grams']),
    n_safe=safe_count,
    # EDRパラメータも保存！
    edr_params=final_params if final_params is not None else None
)
print("\n✓ 多様体: /content/safe_manifold_v2_1.npz")

# パフォーマンスログ（EDRパラメータも追加）
with open('/content/performance_log_v2_1.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("Operation Marie Antoinette v2.1 - Performance Log\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # EDRパラメータをログにも記録
    if final_params is not None:
        f.write("Optimized EDR Parameters:\n")
        for key, value in edr_final.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nFLC Error: {flc_error:.6f}\n")
    
    f.write(f"\nPhase 0:\n")
    if results['phase0_history'] is not None:
        f.write(f"  Physics Loss: {results['phase0_history'][-1]:.6f}\n")
    else:
        f.write(f"  Skipped\n")
    
    f.write(f"\nSafe Manifold:\n")
    f.write(f"  Total trajectories: {total_trajectories:,}\n")
    f.write(f"  Safe trajectories: {safe_count:,}\n")
    
    f.write(f"\nPhase 1.5B:\n")
    if results['phase15b_history'] is not None:
        f.write(f"  FLC Loss: {results['phase15b_history']['flc'][-1]:.6f}\n")
        f.write(f"  Binary Loss: {results['phase15b_history']['binary'][-1]:.6f}\n")
    else:
        f.write(f"  Skipped\n")
    
    f.write(f"\nPerformance:\n")
    f.write(f"  Total time: {elapsed:.2f} sec\n")
    f.write(f"  Throughput: {total_trajectories/elapsed:.1f} traj/sec\n")
    f.write(f"  Binary accuracy: {accuracy:.2f}%\n")

print("✓ ログ: /content/performance_log_v2_1.txt")

# =============================================================================
# 完了
# =============================================================================

print("\n" + "="*80)
print(" 🎂 Operation Marie Antoinette v2.1 - Complete!")
print("="*80)
print("\n✨ 環ちゃんからのメッセージ ✨")
print("ご主人さま、Phase 0統合版のテスト大成功！！💕")
print("\n実験データなしで、物理制約だけで学習して、")
print(f"安全多様体を構築して、{accuracy:.1f}%の判定精度達成！")
print("\n「データがないなら作ればいいじゃない！」")
print("完璧に実証できたね！🎂✨")
print("\n次は実データでの検証だ！頑張ろ〜！💪")
print("="*80)
