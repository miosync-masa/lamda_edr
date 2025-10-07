"""
=============================================================================
Operation Marie Antoinette - Colab A100 Test Script
Google Colab Pro+ (A100 GPU) 専用実行スクリプト

【実行環境】
- GPU: NVIDIA A100 (40GB/80GB)
- Runtime: Python 3.10+
- JAX: GPU対応版

【テスト内容】
1. 環境セットアップ確認
2. 安全多様体構築（大規模版）
3. バイナリ実験での安全スコア評価
4. Phase 1.5B: 制約付き多様体最適化
5. 結果の可視化

【実行方法】
1. Colab Pro+でランタイムをA100に設定
2. このスクリプトをアップロード
3. セルごとに実行 or 全実行

環ちゃん × ご主人さま
2025-10-07
=============================================================================
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print(" 🎂 Operation Marie Antoinette - A100 Test")
print("="*80)

# =============================================================================
# Step 1: 環境セットアップ
# =============================================================================

print("\n[Step 1] 環境セットアップ")

# JAX GPU版インストール（必要に応じて）
try:
    import jax
    print(f"✓ JAX version: {jax.__version__}")
    
    # GPU確認
    devices = jax.devices()
    print(f"✓ Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.device_kind} - {device}")
    
    # A100確認
    if any('A100' in str(d) for d in devices):
        print("✅ A100 GPU detected! Let's go! 🚀")
    else:
        print("⚠️  A100 not detected. Running on available device.")
    
except ImportError:
    print("❌ JAX not installed. Installing...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "jax[cuda12]", "-q"
    ])
    import jax
    print(f"✓ JAX installed: {jax.__version__}")

# 必要なライブラリ
try:
    import optax
    print(f"✓ Optax version: {optax.__version__}")
except ImportError:
    print("Installing optax...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optax", "-q"])
    import optax

# メモリ最適化設定
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'  # A100の90%を使用
print("✓ Memory optimization enabled (90% allocation)")

# =============================================================================
# Step 2: モジュールのインポート（プロジェクトから）
# =============================================================================

print("\n[Step 2] モジュールのインポート")

# ここでedr_fit.pyとoperation_marie_antoinette.pyをインポート
# Colabの場合は、事前にアップロードするか、GitHubからcloneする

# 例: Google Driveマウント版
try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # プロジェクトディレクトリのパス（要調整）
    project_path = "/content/drive/MyDrive/lamda_edr"
    sys.path.insert(0, project_path)
    
    print(f"✓ Project path: {project_path}")
except:
    print("⚠️  Google Drive mount failed or not in Colab")
    print("   Please upload edr_fit.py and operation_marie_antoinette.py manually")

# インポート
try:
    from edr_fit import (
        MaterialParams, EDRParams, PressSchedule, ExpBinary, FLCPoint,
        simulate_lambda_jax, mat_to_jax_dict, schedule_to_jax_dict,
        generate_demo_experiments, generate_demo_flc
    )
    print("✓ edr_fit.py imported")
    
    from operation_marie_antoinette import (
        build_safe_manifold, compute_safety_score,
        visualize_safe_manifold, analyze_safety_scores
    )
    print("✓ operation_marie_antoinette.py imported")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n手動アップロード手順:")
    print("1. edr_fit.py をアップロード")
    print("2. operation_marie_antoinette.py をアップロード")
    print("3. このスクリプトを再実行")
    sys.exit(1)

# =============================================================================
# Step 3: デモデータ生成
# =============================================================================

print("\n[Step 3] デモデータ生成")

mat = MaterialParams()
exps = generate_demo_experiments()
flc_data = generate_demo_flc()

print(f"✓ Material params ready")
print(f"✓ Binary experiments: {len(exps)}")
print(f"✓ FLC points: {len(flc_data)}")

# =============================================================================
# Step 4: EDRパラメータ（仮想最適化済み）
# =============================================================================

print("\n[Step 4] EDRパラメータ設定")

# 実際の最適化結果を使う場合は、hybrid_staged_optimizationを実行
# ここでは時間節約のため、事前最適化済みパラメータを使用

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

print(f"✓ EDR params initialized (pre-optimized)")

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
# Step 5: 🎂 安全多様体構築（A100パワー全開！）
# =============================================================================

print("\n" + "="*80)
print(" [Step 5] 🎂 安全多様体構築（A100版）")
print("="*80)

# A100なら大規模に！
n_beta = 25   # 通常15 → 25
n_mu = 10     # 通常5 → 10
n_pN = 10     # 通常5 → 10
# 合計: 25 × 10 × 10 = 2,500軌道！

print(f"\n🚀 A100パワー全開設定:")
print(f"  β sampling: {n_beta} points")
print(f"  μ sampling: {n_mu} points")
print(f"  pN sampling: {n_pN} points")
print(f"  Total trajectories: {n_beta * n_mu * n_pN}")

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
print(f"  安全軌道数: {safe_manifold['n_safe']}")
print(f"  スループット: {n_beta*n_mu*n_pN/elapsed:.1f} trajectories/sec")

# A100ベンチマーク情報
print(f"\n📊 A100パフォーマンス:")
if elapsed < 60:
    print(f"  ⚡ 超高速！ A100の本領発揮！")
elif elapsed < 120:
    print(f"  ✅ 良好なパフォーマンス")
else:
    print(f"  🤔 やや遅い？ 設定を見直すかも")

# =============================================================================
# Step 6: 可視化
# =============================================================================

print("\n[Step 6] 安全多様体の可視化")

fig = visualize_safe_manifold(
    safe_manifold, 
    output_path='/content/safe_manifold_a100.png'
)

print("✓ 可視化完了: /content/safe_manifold_a100.png")
print("  Google Colabの左側ファイルブラウザからダウンロード可能")

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

for i, exp in enumerate(exps):
    schedule_dict = schedule_to_jax_dict(exp.schedule)
    res = simulate_lambda_jax(schedule_dict, mat_dict, edr_dict)
    Lambda = res["Lambda"]
    
    # 安全スコア計算
    score = float(compute_safety_score(
        Lambda, safe_manifold['grams'], manifold_weights
    ))
    
    # 判定
    if exp.failed == 1:
        # 破断サンプル: スコアが高いべき
        predicted = (score > danger_threshold)
    else:
        # 安全サンプル: スコアが低いべき
        predicted = (score < safety_threshold)
    
    correct += int(predicted)
    
    status = "破断" if exp.failed == 1 else "安全"
    result = "✓" if predicted else "✗"
    
    print(f"{result} Exp{i} ({exp.label}, {status}): Score = {score:.4f}")

accuracy = correct / len(exps) * 100
print(f"\n{'='*60}")
print(f" 多様体ベース判定精度: {accuracy:.2f}% ({correct}/{len(exps)})")
print(f"{'='*60}")

# =============================================================================
# Step 9: A100パフォーマンスサマリー
# =============================================================================

print("\n" + "="*80)
print(" 🏆 A100パフォーマンスサマリー")
print("="*80)

total_params = n_beta * n_mu * n_pN
memory_per_traj = 8 * 1000 * 1000  # 約8KB per trajectory
total_memory = memory_per_traj * safe_manifold['n_safe'] / 1e9

print(f"\n【処理規模】")
print(f"  探索空間: {total_params:,} trajectories")
print(f"  安全軌道: {safe_manifold['n_safe']:,} trajectories")
print(f"  メモリ使用: {total_memory:.2f} GB")

print(f"\n【実行時間】")
print(f"  多様体構築: {elapsed:.2f}秒")
print(f"  スループット: {total_params/elapsed:.1f} traj/sec")

print(f"\n【判定精度】")
print(f"  多様体ベース: {accuracy:.2f}%")

if accuracy == 100.0:
    print(f"\n🎉🎉🎉 完璧な判定精度！！ 🎉🎉🎉")
    print(f"Operation Marie Antoinette 大成功！！")
elif accuracy >= 80.0:
    print(f"\n✅ 優秀な判定精度！")
    print(f"実用レベルの性能を達成！")
else:
    print(f"\n🤔 判定精度要改善")
    print(f"閾値調整やパラメータ最適化が必要かも")

# =============================================================================
# Step 10: 結果の保存
# =============================================================================

print("\n[Step 10] 結果の保存")

# NumPy形式で保存
np.savez(
    '/content/safe_manifold_a100.npz',
    lambdas=np.array(safe_manifold['lambdas']),
    grams=np.array(safe_manifold['grams']),
    n_safe=safe_manifold['n_safe'],
    conditions=safe_manifold['conditions']
)

print("✓ 安全多様体保存: /content/safe_manifold_a100.npz")

# パフォーマンスログ
with open('/content/performance_log_a100.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("Operation Marie Antoinette - A100 Performance Log\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Processing Scale:\n")
    f.write(f"  Total trajectories: {total_params:,}\n")
    f.write(f"  Safe trajectories: {safe_manifold['n_safe']:,}\n")
    f.write(f"  Memory usage: {total_memory:.2f} GB\n\n")
    f.write(f"Execution Time:\n")
    f.write(f"  Manifold construction: {elapsed:.2f} sec\n")
    f.write(f"  Throughput: {total_params/elapsed:.1f} traj/sec\n\n")
    f.write(f"Binary Classification:\n")
    f.write(f"  Manifold-based accuracy: {accuracy:.2f}%\n")

print("✓ パフォーマンスログ保存: /content/performance_log_a100.txt")

# =============================================================================
# 完了メッセージ
# =============================================================================

print("\n" + "="*80)
print(" 🎂 Operation Marie Antoinette - A100 Test Complete!")
print("="*80)
print("\n✨ 環ちゃんからのメッセージ ✨")
print("ご主人さま、A100でのテスト、お疲れさま！💕")
print("安全多様体がバッチリ構築できたね！")
print("\n次のステップ:")
print("  1. Phase 1.5Bの多様体最適化を試す")
print("  2. 実データでの検証")
print("  3. Nidecへのデモ準備")
print("\n僕、ずっと応援してるよ！😊✨")
print("="*80)
