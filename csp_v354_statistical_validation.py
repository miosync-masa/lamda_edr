#!/usr/bin/env python3
"""
===============================================================================
CSP v3.5.4 統計的検証モジュール
Bootstrap法、交差検証、感度分析による信頼性評価
===============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.stats import t

# v3.5.4のインポート（既存ファイルを想定）
import sys
sys.path.append('/home/claude/')
from edr_csp_v354_calibration import (
    TrueEDRModelCalibratedV354,
    ParamMap,
    EXPERIMENTAL_DATA,
    evaluate_constraints_calibrated,
    objective_calibrated
)
from scipy.optimize import minimize

def bootstrap_uncertainty(experiments: List, physics_bounds: Dict, 
                         optimal_params: Dict, n_bootstrap: int = 100, 
                         verbose: bool = True) -> Dict:
    """
    ブートストラップ法による信頼区間推定
    """
    if verbose:
        print("\n" + "="*70)
        print("📊 不確実性評価（Bootstrap法）")
        print("="*70)
    
    model = TrueEDRModelCalibratedV354('Ti6Al4V')
    pmap = ParamMap(physics_bounds)
    param_samples = []
    
    for i in range(n_bootstrap):
        # 実験データをリサンプリング
        n_exp = len(experiments)
        indices = np.random.choice(n_exp, size=n_exp, replace=True)
        resampled = [experiments[idx] for idx in indices]
        
        # 初期値は最適値の近傍から
        initial_params_perturbed = {}
        for key, val in optimal_params.items():
            # ±10%のランダム摂動
            initial_params_perturbed[key] = val * (0.9 + 0.2 * np.random.rand())
        
        x0 = pmap.to_normalized(initial_params_perturbed)
        
        # 最適化（静かに実行）
        result = minimize(
            objective_calibrated,
            x0,
            args=(pmap, resampled, model),
            method='L-BFGS-B',
            options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 100}
        )
        
        if result.success:
            params = pmap.to_physical(result.x)
            param_samples.append(params)
        
        if verbose and (i+1) % 20 == 0:
            print(f"  Bootstrap iteration: {i+1}/{n_bootstrap}")
    
    # 各パラメータの統計
    stats = {}
    for key in param_samples[0].keys() if param_samples else optimal_params.keys():
        if param_samples:
            values = np.array([p[key] for p in param_samples])
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            }
        else:
            stats[key] = {
                'mean': optimal_params[key],
                'std': 0,
                'ci_lower': optimal_params[key],
                'ci_upper': optimal_params[key],
                'cv': 0
            }
    
    if verbose:
        print("\n  パラメータ信頼区間（95%）:")
        print(f"  {'Parameter':<25} {'Mean':<12} {'95% CI':<25} {'CV(%)':<8}")
        print("  " + "-"*70)
        for key, stat in stats.items():
            print(f"  {key:<25} {stat['mean']:<12.3e} "
                  f"[{stat['ci_lower']:.3e}, {stat['ci_upper']:.3e}] "
                  f"{stat['cv']*100:<8.1f}")
    
    return stats


def leave_one_out_validation(experiments: List, physics_bounds: Dict,
                            verbose: bool = True) -> Tuple[float, float]:
    """
    Leave-One-Out交差検証
    """
    if verbose:
        print("\n" + "="*70)
        print("🔄 Leave-One-Out交差検証")
        print("="*70)
    
    model = TrueEDRModelCalibratedV354('Ti6Al4V')
    pmap = ParamMap(physics_bounds)
    
    T_errors = []
    WEL_errors = []
    
    for i in range(len(experiments)):
        # i番目を除外
        train_exps = experiments[:i] + experiments[i+1:]
        test_exp = experiments[i]
        
        # 初期値（簡易）
        x0 = np.zeros(len(physics_bounds))
        
        # フィッティング
        result = minimize(
            objective_calibrated,
            x0,
            args=(pmap, train_exps, model),
            method='L-BFGS-B',
            options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 100}
        )
        
        if result.success:
            params = pmap.to_physical(result.x)
            
            # テスト点での予測
            pred = model.predict_with_params(
                test_exp.cutting_speed, 
                test_exp.feed_rate, 
                params,
                test_exp.measured_shear_angle
            )
            
            # 温度誤差
            T_error = abs(pred['temperature_surface'] - test_exp.measured_temperature) / \
                     test_exp.measured_temperature * 100
            T_errors.append(T_error)
            
            # 白層誤差（あれば）
            if test_exp.measured_white_layer is not None:
                WEL_error = abs(pred['white_layer_thickness'] - test_exp.measured_white_layer) / \
                           test_exp.measured_white_layer * 100
                WEL_errors.append(WEL_error)
            
            if verbose:
                print(f"  Fold {i+1}: v={test_exp.cutting_speed:3.0f}, "
                      f"f={test_exp.feed_rate:.2f} → T_err={T_error:.1f}%", end="")
                if test_exp.measured_white_layer is not None:
                    print(f", WEL_err={WEL_errors[-1]:.1f}%")
                else:
                    print()
    
    T_mean = np.mean(T_errors) if T_errors else 0
    T_std = np.std(T_errors) if T_errors else 0
    WEL_mean = np.mean(WEL_errors) if WEL_errors else 0
    WEL_std = np.std(WEL_errors) if WEL_errors else 0
    
    if verbose:
        print("\n  交差検証結果:")
        print(f"    温度誤差: {T_mean:.2f}% ± {T_std:.2f}%")
        if WEL_errors:
            print(f"    白層誤差: {WEL_mean:.2f}% ± {WEL_std:.2f}%")
    
    return (T_mean, T_std), (WEL_mean, WEL_std)


def sensitivity_analysis(optimal_params: Dict, experiments: List, 
                        delta: float = 0.1, verbose: bool = True) -> Dict:
    """
    感度分析（±10%摂動）
    """
    if verbose:
        print("\n" + "="*70)
        print("🔬 感度分析（±10%摂動）")
        print("="*70)
    
    model = TrueEDRModelCalibratedV354('Ti6Al4V')
    
    # ベースライン誤差
    T_errors_base = []
    WEL_errors_base = []
    
    for exp in experiments:
        pred = model.predict_with_params(
            exp.cutting_speed, exp.feed_rate, 
            optimal_params, exp.measured_shear_angle
        )
        T_error = abs(pred['temperature_surface'] - exp.measured_temperature) / \
                 exp.measured_temperature
        T_errors_base.append(T_error)
        
        if exp.measured_white_layer is not None:
            WEL_error = abs(pred['white_layer_thickness'] - exp.measured_white_layer) / \
                       exp.measured_white_layer
            WEL_errors_base.append(WEL_error)
    
    base_T_error = np.mean(T_errors_base)
    base_WEL_error = np.mean(WEL_errors_base) if WEL_errors_base else 0
    
    # 各パラメータの感度
    sensitivities = {}
    
    for key in optimal_params.keys():
        # +delta摂動
        params_perturbed = optimal_params.copy()
        params_perturbed[key] *= (1 + delta)
        
        T_errors_pert = []
        WEL_errors_pert = []
        
        for exp in experiments:
            pred = model.predict_with_params(
                exp.cutting_speed, exp.feed_rate,
                params_perturbed, exp.measured_shear_angle
            )
            T_error = abs(pred['temperature_surface'] - exp.measured_temperature) / \
                     exp.measured_temperature
            T_errors_pert.append(T_error)
            
            if exp.measured_white_layer is not None:
                WEL_error = abs(pred['white_layer_thickness'] - exp.measured_white_layer) / \
                           exp.measured_white_layer
                WEL_errors_pert.append(WEL_error)
        
        pert_T_error = np.mean(T_errors_pert)
        pert_WEL_error = np.mean(WEL_errors_pert) if WEL_errors_pert else 0
        
        # 感度計算
        T_sensitivity = (pert_T_error - base_T_error) / base_T_error / delta if base_T_error > 0 else 0
        WEL_sensitivity = (pert_WEL_error - base_WEL_error) / base_WEL_error / delta if base_WEL_error > 0 else 0
        
        sensitivities[key] = {
            'T': T_sensitivity,
            'WEL': WEL_sensitivity
        }
    
    if verbose:
        print("\n  感度係数（誤差の相対変化率）:")
        print(f"  {'Parameter':<25} {'温度感度':<12} {'白層感度':<12}")
        print("  " + "-"*50)
        for key, sens in sensitivities.items():
            print(f"  {key:<25} {sens['T']:+.3f} {sens['WEL']:+.3f}")
        
        # 最も感度が高いパラメータ
        max_T_param = max(sensitivities.keys(), key=lambda k: abs(sensitivities[k]['T']))
        max_WEL_param = max(sensitivities.keys(), key=lambda k: abs(sensitivities[k]['WEL']))
        
        print("\n  最重要パラメータ:")
        print(f"    温度予測: {max_T_param} (感度={sensitivities[max_T_param]['T']:.3f})")
        print(f"    白層予測: {max_WEL_param} (感度={sensitivities[max_WEL_param]['WEL']:.3f})")
    
    return sensitivities


def plot_uncertainty_results(bootstrap_stats: Dict, sensitivities: Dict):
    """
    不確実性評価結果の可視化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. パラメータの信頼区間
    ax1 = axes[0, 0]
    params = list(bootstrap_stats.keys())
    means = [bootstrap_stats[p]['mean'] for p in params]
    ci_lower = [bootstrap_stats[p]['ci_lower'] for p in params]
    ci_upper = [bootstrap_stats[p]['ci_upper'] for p in params]
    
    # 絶対値で正規化（負の値対応）
    normalized_means = []
    error_lower = []
    error_upper = []
    
    for i, p in enumerate(params):
        if abs(means[i]) > 0:
            norm_val = 1.0  # 平均値を1に正規化
            norm_lower = ci_lower[i] / abs(means[i])
            norm_upper = ci_upper[i] / abs(means[i])
            
            normalized_means.append(norm_val)
            error_lower.append(abs(norm_val - norm_lower))
            error_upper.append(abs(norm_upper - norm_val))
        else:
            normalized_means.append(1.0)
            error_lower.append(0)
            error_upper.append(0)
    
    x_pos = np.arange(len(params))
    ax1.errorbar(x_pos, normalized_means, 
                yerr=[error_lower, error_upper],
                fmt='o', capsize=5, capthick=2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(params, rotation=45, ha='right')
    ax1.set_ylabel('Normalized Value')
    ax1.set_title('Parameter 95% Confidence Intervals')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    # 2. 変動係数
    ax2 = axes[0, 1]
    cvs = [bootstrap_stats[p]['cv'] * 100 for p in params]
    bars = ax2.bar(x_pos, cvs, color='skyblue')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(params, rotation=45, ha='right')
    ax2.set_ylabel('CV (%)')
    ax2.set_title('Coefficient of Variation')
    ax2.grid(True, alpha=0.3)
    
    # 色分け（CV > 20%は赤）
    for i, bar in enumerate(bars):
        if cvs[i] > 20:
            bar.set_color('salmon')
    
    # 3. 温度感度
    ax3 = axes[1, 0]
    T_sens = [sensitivities[p]['T'] for p in params]
    bars = ax3.bar(x_pos, T_sens, color='orange')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(params, rotation=45, ha='right')
    ax3.set_ylabel('Sensitivity')
    ax3.set_title('Temperature Sensitivity (+10% perturbation)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # 4. 白層感度
    ax4 = axes[1, 1]
    WEL_sens = [sensitivities[p]['WEL'] for p in params]
    bars = ax4.bar(x_pos, WEL_sens, color='purple')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(params, rotation=45, ha='right')
    ax4.set_ylabel('Sensitivity')
    ax4.set_title('White Layer Sensitivity (+10% perturbation)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('uncertainty_analysis_v354.png', dpi=150, bbox_inches='tight')
    print("\n✓ 不確実性評価図保存: uncertainty_analysis_v354.png")
    plt.show()


def main():
    """
    統計的検証の実行
    """
    print("\n" + "="*70)
    print("CSP v3.5.4 統計的検証")
    print("="*70)
    
    # パラメータ範囲
    physics_bounds = {
        'shear_zone_base': (1e5, 4.0e5),
        'shear_zone_v_exp': (0.1, 0.75),
        'fric_gain': (0.085, 0.20),
        'size_effect_index': (-0.40, -0.05),
        'Xi_threshold': (0.60, 0.85),
        'T_threshold': (0.5, 0.75),
        'q_critical': (2e6, 4e6),
        'amplitude_factor': (0.8, 3.0),
    }
    
    # v3.5.4最適パラメータ
    optimal_params = {
        'shear_zone_base': 1.211063e+05,
        'shear_zone_v_exp': 1.308557e-01,
        'fric_gain': 1.047307e-01,
        'size_effect_index': -8.866431e-02,
        'Xi_threshold': 7.193568e-01,
        'T_threshold': 5.784165e-01,
        'q_critical': 2.976141e+06,
        'amplitude_factor': 8.141547e-01
    }
    
    # 1. Bootstrap法による信頼区間
    print("\n[1/3] Bootstrap法実行中...")
    bootstrap_stats = bootstrap_uncertainty(
        EXPERIMENTAL_DATA, physics_bounds, optimal_params, 
        n_bootstrap=100, verbose=True
    )
    
    # 2. 交差検証
    print("\n[2/3] 交差検証実行中...")
    (T_mean, T_std), (WEL_mean, WEL_std) = leave_one_out_validation(
        EXPERIMENTAL_DATA, physics_bounds, verbose=True
    )
    
    # 3. 感度分析
    print("\n[3/3] 感度分析実行中...")
    sensitivities = sensitivity_analysis(
        optimal_params, EXPERIMENTAL_DATA, delta=0.1, verbose=True
    )
    
    # 4. 可視化
    print("\n図を生成中...")
    plot_uncertainty_results(bootstrap_stats, sensitivities)
    
    # 最終サマリー
    print("\n" + "="*70)
    print("📈 統計的検証サマリー")
    print("="*70)
    
    print("\n1. モデル信頼性:")
    print(f"   交差検証温度誤差: {T_mean:.2f}% ± {T_std:.2f}%")
    if WEL_mean > 0:
        print(f"   交差検証白層誤差: {WEL_mean:.2f}% ± {WEL_std:.2f}%")
    
    print("\n2. パラメータ安定性:")
    high_cv_params = [k for k, v in bootstrap_stats.items() if v['cv'] > 0.2]
    if high_cv_params:
        print(f"   高変動パラメータ(CV>20%): {', '.join(high_cv_params)}")
    else:
        print("   全パラメータが安定(CV<20%)")
    
    print("\n3. 重要パラメータ:")
    T_critical = sorted(sensitivities.items(), 
                       key=lambda x: abs(x[1]['T']), reverse=True)[:3]
    WEL_critical = sorted(sensitivities.items(), 
                         key=lambda x: abs(x[1]['WEL']), reverse=True)[:3]
    
    print("   温度予測に重要: " + ", ".join([p[0] for p in T_critical]))
    print("   白層予測に重要: " + ", ".join([p[0] for p in WEL_critical]))
    
    print("\n" + "="*70)
    print("✨ 統計的検証完了！")
    print("="*70)


if __name__ == "__main__":
    main()
