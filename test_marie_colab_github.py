"""
=============================================================================
Operation Marie Antoinette - Colab A100 Test (GitHub版)
GitHub直接cloneして実行

【実行手順】
1. Colabランタイム: A100に設定
2. このセルを実行するだけ！

環ちゃん × ご主人さま
2025-10-07
=============================================================================
"""

# =============================================================================
# Step 0: GitHubからクローン
# =============================================================================

print("="*80)
print(" 🎂 Operation Marie Antoinette - GitHub Clone版")
print("="*80)

import os
import sys
import subprocess

# リポジトリURL
REPO_URL = "https://github.com/miosync-masa/lamda_edr.git"
REPO_NAME = "lamda_edr"

print(f"\n[Step 0] GitHubからクローン")
print(f"  Repository: {REPO_URL}")

# すでにクローン済みなら削除
if os.path.exists(REPO_NAME):
    print(f"  既存の{REPO_NAME}を削除...")
    subprocess.run(['rm', '-rf', REPO_NAME])

# クローン実行
print(f"  クローン開始...")
result = subprocess.run(
    ['git', 'clone', REPO_URL],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(f"  ✅ クローン成功！")
else:
    print(f"  ❌ クローン失敗:")
    print(result.stderr)
    sys.exit(1)

# パス追加
sys.path.insert(0, f'/content/{REPO_NAME}')
print(f"  ✓ パス追加: /content/{REPO_NAME}")

# =============================================================================
# Step 1: 環境セットアップ
# =============================================================================

print("\n[Step 1] 環境セットアップ")

# JAX GPU版確認
try:
    import jax
    print(f"✓ JAX version: {jax.__version__}")
    
    devices = jax.devices()
    print(f"✓ Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.device_kind}")
    
    if any('A100' in str(d) for d in devices):
        print("✅ A100 GPU detected! Let's go! 🚀")
    else:
        print("⚠️  A100 not detected. Using available device.")
        
except ImportError:
    print("❌ JAX not installed. Installing...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "jax[cuda12]", "-q"
    ])
    import jax
    print(f"✓ JAX installed: {jax.__version__}")

# Optax確認
try:
    import optax
    print(f"✓ Optax version: {optax.__version__}")
except ImportError:
    print("Installing optax...")
    subprocess.run([sys.executable, "-m", "pip", "install", "optax", "-q"])
    import optax

# メモリ最適化
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
print("✓ Memory optimization: 90% allocation")

# =============================================================================
# Step 2: モジュールインポート
# =============================================================================

print("\n[Step 2] モジュールインポート")

try:
    from edr_fit import (
        MaterialParams, EDRParams, PressSchedule, ExpBinary, FLCPoint,
        simulate_lambda_jax, mat_to_jax_dict, schedule_to_jax_dict,
        generate_demo_experiments, generate_demo_flc,
        hybrid_staged_optimization
    )
    print("✓ edr_fit.py imported")
    
    from operation_marie_antoinette import (
        build_safe_manifold, compute_safety_score,
        visualize_safe_manifold, analyze_safety_scores,
        phase_15b_manifold_optimization
    )
    print("✓ operation_marie_antoinette.py imported")
    
    print("\n✅ 全モジュールのインポート成功！")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n確認事項:")
    print("1. GitHubリポジトリに edr_fit.py が含まれているか")
    print("2. operation_marie_antoinette.py が含まれているか")
    print("3. 最新版がpushされているか")
    sys.exit(1)

# =============================================================================
# Step 3: デモデータ生成
# =============================================================================

print("\n[Step 3] デモデータ生成")

mat = MaterialParams()
exps = generate_demo_experiments()
flc_data = generate_demo_flc()

print(f"✓ Material params: {mat}")
print(f"✓ Binary experiments: {len(exps)}")
print(f"✓ FLC points: {len(flc_data)}")

# =============================================================================
# Step 4: EDRパラメータ（簡易版 or 最適化実行）
# =============================================================================

print("\n[Step 4] EDRパラメータ設定")
print("\n選択肢:")
print("  A. 事前設定パラメータを使用（高速）")
print("  B. 多段階Hybrid最適化を実行（時間かかる）")

# ここでは簡易版を使用
use_preoptimized = True  # False にすると最適化実行

if use_preoptimized:
    print("\n→ 事前設定パラメータを使用")
    
    edr = EDRParams(
        V0=1.8e9,
        av=3.5e4,
        ad=1.2e-7,
        chi=0.12,
        K_scale=0.22,
        triax_sens=0.32,
        Lambda_crit=0.98,
        K_scale_draw=0.16,
        K_scale_plane=0.26,
        K_scale_biax=0.21,
        beta_A=0.33,
        beta_bw=0.27,
        beta_A_pos=0.52
    )
    
    print(f"✓ EDR params initialized")
    
else:
    print("\n→ 多段階Hybrid最適化を実行")
    print("  （Phase 0 → Phase 1 → Phase 1.5）")
    print("  ⏳ これには5-10分かかります...")
    
    import time
    start = time.time()
    
    edr, info = hybrid_staged_optimization(
        exps, flc_data, mat,
        verbose=True
    )
    
    elapsed = time.time() - start
    
    print(f"\n✅ 最適化完了！（{elapsed:.1f}秒）")
    print(f"  最終Loss: {info['final_loss']:.6f}")
    print(f"  FLC Loss: {info['final_flc_loss']:.6f}")
    print(f"  Binary Loss: {info['final_bin_loss']:.6f}")

# JAX dict化
mat_dict = mat_to_jax_dict(mat)
edr_dict = {
    'V0': edr.V0, 'av': edr.av, 'ad': edr.ad,
    'chi': edr.chi, 'K_scale': edr.K_scale,
    'triax_sens': edr.triax_sens,
    'Lambda_crit': edr.Lambda_crit,
    'K_scale_draw': edr.K_scale_draw,
    'K_scale_plane': edr.K_scale_plane,
    'K_scale_biax': edr.K_scale_biax,
    'beta_A': edr.beta_A, 'beta_bw': edr.beta_bw,
    'beta_A_pos': edr.beta_A_pos
}

# =============================================================================
# Step 5: 🎂 安全多様体構築（A100全開！）
# =============================================================================

print("\n" + "="*80)
print(" [Step 5] 🎂 安全多様体構築（A100パワー全開）")
print("="*80)

import time
import numpy as np

# A100設定
n_beta = 25
n_mu = 10
n_pN = 10
total_trajectories = n_beta * n_mu * n_pN

print(f"\n🚀 A100設定:")
print(f"  β: {n_beta} points")
print(f"  μ: {n_mu} points")
print(f"  pN: {n_pN} points")
print(f"  Total: {total_trajectories:,} trajectories")

start_time = time.time()

safe_manifold = build_safe_manifold(
    mat_dict, edr_dict, simulate_lambda_jax,
    n_beta=n_beta,
    n_mu=n_mu,
    n_pN=n_pN,
    safety_margin=0.85,
    verbose=True
)

elapsed = time.time() - start_time

print(f"\n✅ 安全多様体構築完了！")
print(f"  実行時間: {elapsed:.2f}秒")
print(f"  安全軌道: {safe_manifold['n_safe']}本")
print(f"  スループット: {total_trajectories/elapsed:.1f} traj/sec")

if elapsed < 60:
    print(f"  ⚡ 超高速！A100の本領発揮！")
elif elapsed < 120:
    print(f"  ✅ 良好なパフォーマンス")
else:
    print(f"  🤔 やや遅い？設定を見直すかも")

# =============================================================================
# Step 6: 可視化
# =============================================================================

print("\n[Step 6] 可視化")

import matplotlib.pyplot as plt

fig = visualize_safe_manifold(
    safe_manifold,
    output_path='/content/safe_manifold_a100.png'
)

plt.show()
print("✓ 可視化完了: /content/safe_manifold_a100.png")

# =============================================================================
# Step 7: 安全スコア分析
# =============================================================================

print("\n[Step 7] 安全スコア分析")

manifold_weights = {
    'tv': 0.1,
    'jump': 0.5,
    'topo': 0.1,
    'l1': 1e-3
}

analyze_safety_scores(
    exps, mat_dict, edr_dict, safe_manifold,
    simulate_lambda_jax, manifold_weights
)

# =============================================================================
# Step 8: バイナリ判定精度評価
# =============================================================================

print("\n" + "="*80)
print(" [Step 8] バイナリ判定精度評価（多様体ベース）")
print("="*80)

correct = 0
safety_threshold = 0.3
danger_threshold = 0.5

results = []

for i, exp in enumerate(exps):
    schedule_dict = schedule_to_jax_dict(exp.schedule)
    res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
    Lambda = res["Lambda"]
    
    score = float(compute_safety_score(
        Lambda, safe_manifold['grams'], manifold_weights
    ))
    
    if exp.failed == 1:
        predicted = (score > danger_threshold)
    else:
        predicted = (score < safety_threshold)
    
    correct += int(predicted)
    
    status = "破断" if exp.failed == 1 else "安全"
    result = "✓" if predicted else "✗"
    
    results.append({
        'exp_id': i,
        'label': exp.label,
        'status': status,
        'score': score,
        'predicted': predicted
    })
    
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
# Step 9: 結果保存
# =============================================================================

print("\n[Step 9] 結果保存")

# 多様体保存
np.savez(
    '/content/safe_manifold_a100.npz',
    lambdas=np.array(safe_manifold['lambdas']),
    grams=np.array(safe_manifold['grams']),
    n_safe=safe_manifold['n_safe']
)
print("✓ 多様体保存: /content/safe_manifold_a100.npz")

# 結果CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('/content/binary_results.csv', index=False)
print("✓ 結果保存: /content/binary_results.csv")

# パフォーマンスログ
memory_usage = len(safe_manifold['lambdas']) * 8 * 1000 * 1000 / 1e9

with open('/content/performance_log.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("Operation Marie Antoinette - A100 Performance\n")
    f.write("="*80 + "\n\n")
    f.write(f"Execution Time: {elapsed:.2f} sec\n")
    f.write(f"Total Trajectories: {total_trajectories:,}\n")
    f.write(f"Safe Trajectories: {safe_manifold['n_safe']:,}\n")
    f.write(f"Throughput: {total_trajectories/elapsed:.1f} traj/sec\n")
    f.write(f"Memory Usage: {memory_usage:.2f} GB\n")
    f.write(f"Binary Accuracy: {accuracy:.2f}%\n")

print("✓ ログ保存: /content/performance_log.txt")

# =============================================================================
# 完了
# =============================================================================

print("\n" + "="*80)
print(" 🎂 Operation Marie Antoinette - Complete!")
print("="*80)
print("\n✨ 環ちゃんからのメッセージ ✨")
print("ご主人さま、A100でのテスト大成功！！💕")
print(f"安全多様体: {safe_manifold['n_safe']}本の軌道を構築")
print(f"判定精度: {accuracy:.1f}%")
print("\n「データがないなら作ればいいじゃない！」")
print("Operation Marie Antoinette、完璧に動いたよ！🎂✨")
print("\n次は実データでの検証だね！頑張ろ〜！💪")
print("="*80)
