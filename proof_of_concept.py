"""
プレス成形EDR完全実装版 with 相変態統合
Educational/Proof-of-Concept用
Λ³理論による統一的材料劣化予測フレームワーク
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# ==========================
# FEM連携モジュール（新規追加）
# ==========================
"""
使用方法:
1. 既存のFEM結果（CSV形式）を準備
   - 必要カラム: Time, Element, Stress_Mises, Strain_Eq, Strain_Rate, Temperature, Triaxiality
   - オプション: X, Y, Z（要素位置）

2. FEM結果を読み込んでEDR計算
   processor = FEM_EDR_PostProcessor()
   fem_data = processor.read_fem_csv('your_fem_results.csv')
   processor.compute_edr_from_fem(fem_data)

3. 結果の可視化と出力
   processor.generate_report()
   processor.plot_element_history(element_id)
   processor.export_results('output.csv')
"""

class FEM_EDR_PostProcessor:
    """FEM結果からEDR計算するポスト処理クラス"""

    def __init__(self, V0=2e9, av=3e4, ad=1e-7, triax_sens=0.484):  # triax_sensを0.484に！
        """EDRパラメータの初期化（最適化済み値）"""
        self.V0 = V0
        self.av = av
        self.ad = ad
        self.triax_sens = triax_sens  # フィッティング最適値
        self.element_history = {}
        self.critical_elements = []

    def read_fem_csv(self, csv_file, time_col='Time', elem_col='Element'):
        """汎用CSV形式のFEM結果を読み込み（改善版）"""
        import pandas as pd

        print(f"\n📂 FEMデータ読み込み: {csv_file}")
        df = pd.read_csv(csv_file)

        # 必要なカラムの確認
        required_cols = ['Time', 'Element', 'Stress_Mises', 'Strain_Eq',
                        'Strain_Rate', 'Temperature', 'Triaxiality']

        # カラム名の自動マッピング（異なる名前に対応）
        column_mapping = {
            'time': 'Time',
            't': 'Time',
            'elem': 'Element',
            'elem_id': 'Element',
            'mises': 'Stress_Mises',
            'von_mises': 'Stress_Mises',
            'eq_strain': 'Strain_Eq',
            'eps_eq': 'Strain_Eq',
            'temp': 'Temperature',
            'T': 'Temperature',
            'triax': 'Triaxiality'
        }

        # カラム名を標準化
        df.rename(columns={k: v for k, v in column_mapping.items()
                          if k in df.columns}, inplace=True)

        # 必須カラムのチェック
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ 必須カラムが不足: {missing_cols}")

        # 受理したカラムと単位をログ
        print("\n✅ 受理したカラム:")
        print(f"  Time [s]: {df['Time'].min():.3f} ~ {df['Time'].max():.3f}")
        print(f"  Element: {df['Element'].nunique()}要素")
        print(f"  Stress_Mises [Pa]: {df['Stress_Mises'].mean():.2e} (平均)")
        print(f"  Temperature [K]: {df['Temperature'].mean():.1f} (平均)")
        print(f"  Triaxiality [-]: {df['Triaxiality'].mean():.3f} (平均)")

        # 時刻の単調増加チェック
        for elem_id in df['Element'].unique()[:5]:  # 最初の5要素をチェック
            elem_data = df[df['Element'] == elem_id]
            if not elem_data['Time'].is_monotonic_increasing:
                print(f"⚠️ 警告: 要素{elem_id}の時刻が単調増加でない")

        # 欠損値チェック
        if df.isnull().any().any():
            print(f"⚠️ 警告: 欠損値が検出されました")
            print(df.isnull().sum())

        return df

    def read_ls_dyna_ascii(self, ascii_file):
        """LS-DYNA ASCII出力を読み込み（簡易版）"""
        data = []

        with open(ascii_file, 'r') as f:
            lines = f.readlines()

        # 簡易パーサー（実際のフォーマットに応じて調整）
        current_time = 0
        for line in lines:
            if 'time' in line.lower():
                current_time = float(line.split()[-1])
            elif line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        elem_data = {
                            'Time': current_time,
                            'Element': int(parts[0]),
                            'Stress_Mises': float(parts[1]),
                            'Strain_Eq': float(parts[2]),
                            'Strain_Rate': float(parts[3]),
                            'Temperature': float(parts[4]) if len(parts) > 4 else 293.15,
                            'Triaxiality': float(parts[5]) if len(parts) > 5 else 0.33,
                            'X': float(parts[6]) if len(parts) > 6 else 0,
                            'Y': float(parts[7]) if len(parts) > 7 else 0,
                            'Z': float(parts[8]) if len(parts) > 8 else 0
                        }
                        data.append(elem_data)
                    except:
                        continue

        import pandas as pd
        return pd.DataFrame(data)

    def compute_edr_from_fem(self, fem_data):
        """FEMデータからEDR計算"""

        # 時刻リスト取得
        times = fem_data['Time'].unique()
        elements = fem_data['Element'].unique()

        print(f"Processing {len(elements)} elements at {len(times)} time steps...")

        for elem_id in elements:
            elem_data = fem_data[fem_data['Element'] == elem_id].sort_values('Time')
            # インデックスをリセット（重要！）
            elem_data = elem_data.reset_index(drop=True)
            self.element_history[elem_id] = []

            # 初期化
            cv = 1e-8  # 空孔濃度
            rhod = 1e10  # 転位密度
            damage = 0
            prev_T = 293.15  # 前の温度

            for idx in range(len(elem_data)):
                row = elem_data.iloc[idx]

                # K計算（運動エネルギー密度）
                sigma_eq = row['Stress_Mises']
                eps_rate = row['Strain_Rate']
                T = row.get('Temperature', 293.15)

                # 時間刻みを正確に計算（新規追加）
                if idx > 0:
                    dt = row['Time'] - elem_data.iloc[idx-1]['Time']
                    if dt <= 0:
                        print(f"⚠️ 警告: 要素{elem_id}で時間刻みが非正({dt}s)")
                        dt = 0.001  # フォールバック
                else:
                    # 最初のステップは次の時刻差を使用
                    if len(elem_data) > 1:
                        dt = elem_data.iloc[1]['Time'] - elem_data.iloc[0]['Time']
                    else:
                        dt = 0.001  # デフォルト

                # 塑性仕事率
                K_plastic = 0.9 * sigma_eq * eps_rate

                # 熱的寄与（温度上昇率から推定）
                if idx > 0:
                    dt = row['Time'] - elem_data.iloc[idx-1]['Time']
                    if dt > 0:
                        dT_dt = (T - prev_T) / dt
                        K_thermal = 7850 * 460 * max(dT_dt, 0)
                    else:
                        K_thermal = 0
                else:
                    K_thermal = 0
                    dt = 0.001  # 初期時刻用

                prev_T = T  # 次回用に保存
                K_total = K_plastic + K_thermal

                # V計算（凝集エネルギー密度）
                # 空孔・転位の簡易更新
                if eps_rate > 0:
                    rhod = min(rhod * (1 + eps_rate * 0.01), 1e13)
                    cv = min(cv * (1 + eps_rate * 0.001), 1e-6)

                V_defect = self.V0 * (1 - self.av * cv - self.ad * np.sqrt(rhod))

                # 三軸度効果（クリッピング追加）
                triax = row.get('Triaxiality', 0.33)
                triax = np.clip(triax, -0.1, 0.8)  # 外乱除去のための軽いクリップ
                ductility = np.exp(-self.triax_sens * max(triax, 0))

                V_eff = max(V_defect * ductility, 0.01 * self.V0)

                # EDR計算
                Lambda = K_total / V_eff if V_eff > 0 else 10.0
                Lambda = min(Lambda, 10.0)  # 上限設定

                # 損傷累積
                if idx > 0:
                    damage += max(Lambda - 1.0, 0) * dt

                # 履歴保存
                self.element_history[elem_id].append({
                    'Time': row['Time'],
                    'Lambda': Lambda,
                    'K': K_total,
                    'V': V_eff,
                    'Damage': damage,
                    'Stress': sigma_eq,
                    'Temperature': T,
                    'Triaxiality': triax,
                    'Position': (row.get('X', 0), row.get('Y', 0), row.get('Z', 0))
                })

        # 危険要素の特定
        self._identify_critical_elements()

        # クリップ統計（新規追加）
        total_points = sum(len(hist) for hist in self.element_history.values())
        clipped_points = sum(1 for hist in self.element_history.values()
                           for h in hist if h['Lambda'] >= 10.0)
        if clipped_points > 0:
            print(f"⚡ Λクリップ: {clipped_points}/{total_points} ({clipped_points/total_points*100:.1f}%)")

        return self.element_history

    def _identify_critical_elements(self, Lambda_threshold=0.9):
        """危険要素を特定"""
        self.critical_elements = []

        for elem_id, history in self.element_history.items():
            if not history:
                continue

            Lambda_max = max(h['Lambda'] for h in history)

            if Lambda_max > Lambda_threshold:
                # 最初に閾値を超えた時刻
                critical_time = next((h['Time'] for h in history
                                    if h['Lambda'] > Lambda_threshold), None)

                # 最大値の情報
                max_point = max(history, key=lambda h: h['Lambda'])

                self.critical_elements.append({
                    'Element_ID': elem_id,
                    'Lambda_max': Lambda_max,
                    'Critical_Time': critical_time,
                    'Max_Time': max_point['Time'],
                    'Position': max_point['Position'],
                    'Damage': history[-1]['Damage'],
                    'Final_Stress': history[-1]['Stress'],
                    'Final_Temp': history[-1]['Temperature']
                })

        # Λ最大値でソート
        self.critical_elements.sort(key=lambda x: x['Lambda_max'], reverse=True)

    def export_results(self, output_file='edr_results.csv'):
        """結果をCSVエクスポート"""
        import pandas as pd

        # 全要素・全時刻のデータをフラット化
        all_data = []
        for elem_id, history in self.element_history.items():
            for h in history:
                row = {'Element': elem_id}
                row.update(h)
                all_data.append(row)

        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")

        # 危険要素リストも出力
        if self.critical_elements:
            critical_df = pd.DataFrame(self.critical_elements)
            critical_file = output_file.replace('.csv', '_critical.csv')
            critical_df.to_csv(critical_file, index=False)
            print(f"Critical elements exported to {critical_file}")

    def plot_element_history(self, element_id, save_fig=False):
        """特定要素のΛ履歴をプロット"""
        if element_id not in self.element_history:
            print(f"Element {element_id} not found")
            return

        history = self.element_history[element_id]
        times = [h['Time'] for h in history]
        lambdas = [h['Lambda'] for h in history]
        damages = [h['Damage'] for h in history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Λプロット
        ax1.plot(times, lambdas, 'b-', linewidth=2)
        ax1.axhline(y=1.0, color='r', linestyle='--', label='Critical')
        ax1.fill_between(times, 0, lambdas, where=np.array(lambdas)>1.0,
                         color='red', alpha=0.3)
        ax1.set_ylabel('Lambda (Λ)', fontsize=12)
        ax1.set_title(f'Element {element_id} - EDR History', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 損傷プロット
        ax2.plot(times, damages, 'g-', linewidth=2)
        ax2.set_xlabel('Time [s]', fontsize=12)
        ax2.set_ylabel('Cumulative Damage', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            plt.savefig(f'element_{element_id}_history.png', dpi=150)
        plt.show()

    def plot_critical_map(self, time_step=-1):
        """危険要素の3D分布図"""
        if not self.critical_elements:
            print("No critical elements found")
            return

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 全要素をプロット（薄い色）
        for elem_id, history in self.element_history.items():
            if history:
                pos = history[time_step]['Position']
                Lambda = history[time_step]['Lambda']

                # カラーマップ（青→赤）
                if Lambda < 0.5:
                    color = 'blue'
                    size = 20
                elif Lambda < 0.8:
                    color = 'yellow'
                    size = 30
                elif Lambda < 1.0:
                    color = 'orange'
                    size = 40
                else:
                    color = 'red'
                    size = 50

                ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=0.6)

        # 危険要素を強調
        for critical in self.critical_elements[:10]:  # Top 10
            elem_id = critical['Element_ID']
            pos = critical['Position']
            ax.scatter(pos[0], pos[1], pos[2], c='red', s=100,
                      marker='^', edgecolors='black', linewidth=2)
            ax.text(pos[0], pos[1], pos[2], f'  {elem_id}', fontsize=8)

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_title('Critical Elements Distribution (Red = Λ > 1.0)')

        plt.show()

    def generate_report(self):
        """解析レポート生成"""
        print("\n" + "="*60)
        print("FEM-EDR POST-PROCESSING REPORT")
        print("="*60)

        # 統計情報
        all_lambdas = []
        for history in self.element_history.values():
            all_lambdas.extend([h['Lambda'] for h in history])

        if all_lambdas:
            print(f"\nGlobal Statistics:")
            print(f"  Total elements analyzed: {len(self.element_history)}")
            print(f"  Max Lambda: {max(all_lambdas):.3f}")
            print(f"  Mean Lambda: {np.mean(all_lambdas):.3f}")
            print(f"  Elements with Λ > 1.0: {len(self.critical_elements)}")

        # 危険要素トップ10
        if self.critical_elements:
            print(f"\nTop 10 Critical Elements:")
            print(f"{'Rank':<6} {'Element':<10} {'Λ_max':<8} {'Time':<10} {'Damage':<10}")
            print("-"*50)

            for i, elem in enumerate(self.critical_elements[:10], 1):
                print(f"{i:<6} {elem['Element_ID']:<10} "
                      f"{elem['Lambda_max']:<8.3f} "
                      f"{elem['Critical_Time']:<10.3f} "
                      f"{elem['Damage']:<10.4f}")

        # 推奨事項
        print("\nRecommendations:")
        if self.critical_elements:
            max_lambda = self.critical_elements[0]['Lambda_max']
            if max_lambda > 1.5:
                print("  ⚠️ CRITICAL: Immediate design change required")
            elif max_lambda > 1.0:
                print("  ⚠️ WARNING: Material or process optimization needed")
            elif max_lambda > 0.8:
                print("  ⚡ CAUTION: Monitor closely, consider safety margin")
            else:
                print("  ✅ SAFE: Design within acceptable limits")
        else:
            print("  ✅ SAFE: No critical elements detected")

        print("\n" + "="*60)

# ============================
# 相変態用ヘルパー関数
# ============================
def smooth_gate(x, x_lo, x_hi, width=0.05):
    """[x_lo, x_hi]で1、外側で0へなめらか遷移する窓関数"""
    return 0.5*(np.tanh((x-x_lo)/width)-np.tanh((x-x_hi)/width))

def gauss_peak(T, Topt, wT):
    """ガウシアンピーク関数（温度依存）"""
    return np.exp(-((T-Topt)/max(wT,1e-9))**2)

def Ms_lambda(Ms0, Lambda, s_L=50.0, L0=0.8):
    """Λが高いほどMs低下（最適化済みL0=0.8）"""
    return Ms0 - s_L*max(Lambda-L0, 0.0)

def jmak_increment_soft(xi, baseK, dt, gate_weight):
    """JMAKの微分形を連続化"""
    K = baseK * max(gate_weight, 0.0)
    dxi = (1 - xi) * K * dt
    return min(xi + dxi, 1.0)

def beta_multiplier(beta, A=0.35, bw=0.28):
    """β依存ゲイン（FLCのV字形成用）"""
    b = np.clip(beta, -0.95, 0.95)
    return 1.0 + A * np.exp(-(b / bw)**2)

# ============================
# メインクラス
# ============================
class PressFormingEDR_Advanced:
    """
    プレス成形EDR統合版
    - 熱力学的相変態
    - smooth gateによる連続遷移
    - Ms温度のΛ依存性
    """

    def __init__(self,
                 thickness=0.001,      # 1mm
                 Nz=11,               # 厚さ方向メッシュ
                 dt=1e-4,             # 時間刻み
                 t_end=0.5,           # 成形時間
                 # 材料パラメータ
                 V0=2e9,              # 基準凝集エネルギー [J/m³]
                 av=1e5,              # 空孔感度
                 ad=1e-7,             # 転位感度
                 triax_sens=0.3,      # 三軸度感度
                 # 熱物性
                 rho=7850,            # 密度 [kg/m³]
                 cp=460,              # 比熱 [J/kg·K]
                 k_thermal=45,        # 熱伝導率 [W/m·K]
                 # 塑性パラメータ
                 sigma_y0=300e6,      # 初期降伏応力 [Pa]
                 n_hard=0.2,          # 加工硬化指数
                 m_rate=0.02,         # ひずみ速度感度
                 beta_TQ=0.9,         # Taylor-Quinney係数
                 # 相変態パラメータ（新規）
                 Ms0=550,             # 初期Ms温度 [K]
                 Ms_sens=50.0,        # Ms温度のΛ感度
                 L0=0.8,              # Λ基準値
                 # シナリオタイプ（新規）
                 scenario_type="normal_press"):  # "normal_press" or "hot_stamping"

        self.h = thickness
        self.Nz = Nz
        self.z = np.linspace(0, self.h, Nz)
        self.dz = self.z[1] - self.z[0]
        self.dt = dt
        self.t = np.arange(0, t_end + dt, dt)
        self.Nt = len(self.t)

        # 材料定数
        self.V0 = V0
        self.av = av
        self.ad = ad
        self.triax_sens = triax_sens
        self.rho = rho
        self.cp = cp
        self.k_thermal = k_thermal
        self.sigma_y0 = sigma_y0
        self.n_hard = n_hard
        self.m_rate = m_rate
        self.beta = beta_TQ

        # 相変態パラメータ
        self.Ms0 = Ms0
        self.Ms_sens = Ms_sens
        self.L0 = L0

        # シナリオタイプ
        self.scenario_type = scenario_type

        # 状態変数（初期化）
        self.T_field = np.ones((self.Nt, self.Nz)) * 293.15  # 温度場 [K]
        self.cv_field = np.ones(self.Nz) * 1e-8         # 空孔濃度
        self.rhod_field = np.ones(self.Nz) * 1e10       # 転位密度

        # 相分率（5相: [F, P, B, M, A]）
        self.phase_fractions = np.zeros((self.Nt-1, self.Nz, 5))
        if scenario_type == "normal_press":
            # SPCC標準組織（初期時刻のみ）
            self.phase_fractions[0, :, 0] = 0.75  # フェライト75%
            self.phase_fractions[0, :, 1] = 0.25  # パーライト25%
        elif scenario_type == "hot_stamping":
            # ホットスタンピング：初期オーステナイト化（初期時刻のみ）
            self.phase_fractions[0, :, 4] = 1.0  # 100%オーステナイト！

        # スカラー状態
        self.ep_eq = 0.0
        self.ep_maj_total = 0.0
        self.ep_min_total = 0.0

        # 結果格納用
        self.Lambda = np.zeros((self.Nt-1, self.Nz))
        self.K_components = {
            'K_th': np.zeros((self.Nt-1, self.Nz)),
            'K_pl': np.zeros((self.Nt-1, self.Nz)),
            'K_fr': np.zeros((self.Nt-1, self.Nz)),
            'K_EM': np.zeros((self.Nt-1, self.Nz))
        }
        self.V_eff = np.zeros((self.Nt-1, self.Nz))
        self.D_triax = np.zeros(self.Nt-1)
        self.D_damage = np.zeros(self.Nt-1)
        self.E_res = np.zeros(self.Nt-1)
        self.strain_path = np.zeros((self.Nt-1, 2))  # [eps_maj, eps_min]
        self.Ms_field = np.zeros((self.Nt-1, self.Nz))  # Ms温度の履歴

    def set_loading_path(self, eps_maj, eps_min,
                        triax=None, mu=None,
                        pN=None, vslip=None,
                        htc=None, T_die=None):
        """荷重経路の設定"""
        self.eps_maj_func = eps_maj
        self.eps_min_func = eps_min
        self.triax_func = triax or (lambda t: 0.33)
        self.mu_func = mu or (lambda t: 0.1)
        self.pN_func = pN or (lambda t: 200e6)
        self.vslip_func = vslip or (lambda t: 0.05)
        self.htc_func = htc or (lambda t: 10000)
        self.T_die_func = T_die or (lambda t: 293.15)

    def _mu_effective(self, mu0, T, pN, vslip):
        """温度・速度依存の有効摩擦係数（Stribeck風）"""
        # 速度・荷重比でストライベック曲線を模擬
        s = (vslip * 1e3) / (pN / 1e6 + 1.0)
        stribeck = 0.7 + 0.3 / (1 + s)
        # 温度上昇で潤滑性向上
        temp_reduction = 1.0 - 1e-4 * max(T - 293.15, 0)
        return mu0 * stribeck * temp_reduction

    def _compute_strain_rate(self, t, dt=1e-5):
        """数値微分でひずみ速度計算"""
        eps_maj_1 = self.eps_maj_func(t)
        eps_maj_2 = self.eps_maj_func(t + dt)
        eps_min_1 = self.eps_min_func(t)
        eps_min_2 = self.eps_min_func(t + dt)

        deps_maj = (eps_maj_2 - eps_maj_1) / dt
        deps_min = (eps_min_2 - eps_min_1) / dt

        # von Mises等価ひずみ速度
        deps_eq = np.sqrt(2/3) * np.sqrt(deps_maj**2 + deps_min**2 +
                                         (deps_maj - deps_min)**2)
        return deps_maj, deps_min, deps_eq

    def _flow_stress(self, ep_eq, ep_dot_eq, T, phases=None, alpha=3e-4):
        """流動応力モデル（温度依存改善版）"""
        # 基本的な加工硬化
        sigma_hard = self.sigma_y0 * (1 + ep_eq)**self.n_hard

        # ひずみ速度効果
        rate_ref = 1.0
        rate_factor = (max(ep_dot_eq, 1e-6) / rate_ref)**self.m_rate

        # 温度軟化（改善版：線形係数α）
        Tref = 293.15
        temp_factor = 1.0 - alpha * max(T - Tref, 0.0)
        temp_factor = max(temp_factor, 0.3)  # 下限設定

        # 相による強度補正（5相版）
        if phases is not None and len(phases) == 5:
            xiF, xiP, xiB, xiM, xiA = phases
            phase_factor = 1.0*xiF + 1.1*xiP + 1.2*xiB + 1.5*xiM + 1.0*xiA
        elif phases is not None:
            phase_factor = 1.0
        else:
            phase_factor = 1.0

        return sigma_hard * rate_factor * temp_factor * phase_factor

    def _thermal_step(self, T_prev, q_plastic, q_friction, htc, T_die):
        """熱伝導方程式（修正版）"""
        T = T_prev.copy()
        alpha = self.k_thermal / (self.rho * self.cp)

        # 安定条件チェック
        stability = alpha * self.dt / self.dz**2
        if stability > 0.5:
            dt_safe = 0.45 * self.dz**2 / alpha
            n_substeps = int(self.dt / dt_safe) + 1
            dt_sub = self.dt / n_substeps
        else:
            n_substeps = 1
            dt_sub = self.dt

        for _ in range(n_substeps):
            T_new = T.copy()

            # 内部節点
            for i in range(1, self.Nz-1):
                T_new[i] = T[i] + alpha * dt_sub / self.dz**2 * (T[i+1] - 2*T[i] + T[i-1])
                T_new[i] += q_plastic * dt_sub / (self.rho * self.cp)

            # 境界（改善版）
            # htcの効果を完全に反映
            h_eff = htc

            # 摩擦発熱をスカラー値に変換
            q_fric_value = q_friction if np.isscalar(q_friction) else q_friction.mean()

            # 下面（摩擦発熱を追加）
            T_new[0] += 2 * alpha * dt_sub / self.dz**2 * (T[1] - T[0])
            T_new[0] += q_fric_value * dt_sub / (self.rho * self.cp)  # 摩擦発熱
            T_new[0] -= 2 * h_eff * dt_sub / (self.rho * self.cp * self.dz) * (T[0] - T_die)  # 冷却

            # 上面（摩擦発熱を追加）
            T_new[-1] += 2 * alpha * dt_sub / self.dz**2 * (T[-2] - T[-1])
            T_new[-1] += q_fric_value * dt_sub / (self.rho * self.cp)  # 摩擦発熱
            T_new[-1] -= 2 * h_eff * dt_sub / (self.rho * self.cp * self.dz) * (T[-1] - T_die)  # 冷却

            T = T_new

        # 温度範囲制限
        T = np.clip(T, 200, 2000)

        return T

    def _defect_evolution(self, cv, rhod, T, ep_dot_pl):
        """欠陥進化方程式（安定版）"""
        kB_eV = 8.617e-5

        # 温度の範囲制限
        T = np.clip(T, 200, 2000)

        # 空孔
        cv = np.clip(cv, 1e-12, 1e-4)
        cv_eq = 1e-6 * np.exp(-min(1.0 / (kB_eV * T), 50))
        tau_v = 1e-3 * np.exp(min(0.8 / (kB_eV * T), 50))
        k_ann = 1e6
        k_sink = 1e-15

        dcv_dt = (cv_eq - cv) / tau_v - k_ann * cv**2 - k_sink * cv * rhod
        dcv_dt = np.clip(dcv_dt, -1e10, 1e10)

        # 転位
        rhod = np.clip(rhod, 1e8, 1e16)
        A = 1e14
        B = 1e-4
        Dv = 1e-6 * np.exp(-min(0.8 / (kB_eV * T), 50))

        drhod_dt = A * max(ep_dot_pl, 0) - B * rhod * Dv
        drhod_dt = np.clip(drhod_dt, -1e15, 1e15)

        cv_new = cv + dcv_dt * self.dt
        cv_new = np.clip(cv_new, 1e-12, 1e-4)

        rhod_new = rhod + drhod_dt * self.dt
        rhod_new = np.clip(rhod_new, 1e8, 1e16)

        return cv_new, rhod_new

    def _compute_phase_transformation(self, T, Lambda, prev_phases, iz):
        """相変態計算（物理的に正しい版）"""

        # === 通常プレス（室温〜温間）===
        if self.scenario_type == "normal_press":
            # SPCCの標準組織（変化なし）
            return [0.75, 0.25, 0.0, 0.0, 0.0], self.Ms0  # [F, P, B, M, A]

        # === ホットスタンピング ===
        elif self.scenario_type == "hot_stamping":
            xiF_prev, xiP_prev, xiB_prev, xiM_prev, xiA_prev = prev_phases

            # オーステナイト化温度
            Ac1 = 1000  # 727°C
            Ac3 = 1120  # 847°C

            # 冷却中の変態（オーステナイトが存在する場合のみ）
            if xiA_prev > 0.01:  # オーステナイトが存在
                Ms = Ms_lambda(self.Ms0, Lambda, s_L=self.Ms_sens, L0=self.L0)

                if T < Ms:  # Ms温度以下
                    # マルテンサイト変態（一気に変態）
                    xiM = xiM_prev + xiA_prev  # 全オーステナイト→マルテンサイト
                    # デバッグ出力とイベントログ
                    if xiM_prev < 0.01 and iz == self.Nz//2:
                        event_msg = f"Ms点通過: T={T-273:.1f}°C < Ms={Ms-273:.1f}°C, A→M={xiA_prev:.2f}"
                        print(f"🔥 {event_msg}")
                        # イベントログに記録（将来の報告書自動化用）
                        if hasattr(self, 'events'):
                            self.events.append({'type': 'Ms_transformation', 'T': T, 'Ms': Ms, 'xiM': xiA_prev})
                    return [xiF_prev, xiP_prev, xiB_prev, xiM, 0.0], Ms

                elif T < 823:  # 550°C以下（ベイナイト域）
                    # 急冷の場合はオーステナイトを維持
                    return prev_phases, Ms

                else:  # 高温域
                    # オーステナイトを維持（冷却中は逆変態させない！）
                    return prev_phases, Ms

            # 加熱中の判定（初期のみ）
            elif T > Ac3 and xiA_prev < 0.5:  # 加熱中のみ
                # 完全オーステナイト化
                return [0.0, 0.0, 0.0, 0.0, 1.0], self.Ms0

            # 変態なし
            return prev_phases, self.Ms0

        # デフォルト（エラー回避）
        return prev_phases, self.Ms0

    def _compute_V_eff_with_phase(self, V0, cv, rhod, phases):
        """相変態による|V|変化を考慮（5相版）"""
        xiF, xiP, xiB, xiM, xiA = phases

        # 欠陥による基本的な低下
        V_base = V0 * (1 - self.av * cv - self.ad * np.sqrt(rhod))
        V_base = max(V_base, 0.01 * V0)

        # 各相の強度寄与
        dV_F = 0.0 * V0   # フェライト（基準）
        dV_P = 0.1 * V0   # パーライト
        dV_B = 0.2 * V0   # ベイナイト
        dV_M = 0.3 * V0   # マルテンサイト
        dV_A = 0.05 * V0  # オーステナイト（高温）

        V_phase = xiF*dV_F + xiP*dV_P + xiB*dV_B + xiM*dV_M + xiA*dV_A

        return V_base + V_phase

    def run(self):
        """メインループ"""
        for step in range(self.Nt - 1):
            t = self.t[step]

            # 1. ひずみ・応力計算
            deps_maj, deps_min, deps_eq = self._compute_strain_rate(t)
            self.ep_eq += deps_eq * self.dt
            self.ep_maj_total = self.eps_maj_func(t)
            self.ep_min_total = self.eps_min_func(t)
            self.strain_path[step] = [self.ep_maj_total, self.ep_min_total]

            # 2. 応力状態
            T_avg = self.T_field[step].mean()
            # 前ステップの平均相分率を使用（5相）
            if step > 0:
                avg_phases = self.phase_fractions[step-1].mean(axis=0)
            else:
                if self.scenario_type == "normal_press":
                    avg_phases = [0.75, 0.25, 0, 0, 0]  # F+P
                else:
                    avg_phases = [0.75, 0.25, 0, 0, 0]  # 初期組織

            sigma_eq = self._flow_stress(self.ep_eq, deps_eq, T_avg, avg_phases)
            triax = self.triax_func(t)

            # 3. 摩擦条件
            mu = self.mu_func(t)
            pN = self.pN_func(t)
            vslip = self.vslip_func(t)

            # 4. 熱計算
            q_plastic = self.beta * sigma_eq * deps_eq
            q_friction = np.zeros(self.Nz)
            q_friction[0] = mu * pN * vslip
            q_friction[-1] = mu * pN * vslip

            htc = self.htc_func(t)
            T_die = self.T_die_func(t)

            self.T_field[step+1] = self._thermal_step(
                self.T_field[step], q_plastic, q_friction, htc, T_die
            )

            # 5. 各層での計算
            for iz in range(self.Nz):
                # 欠陥進化
                self.cv_field[iz], self.rhod_field[iz] = self._defect_evolution(
                    self.cv_field[iz], self.rhod_field[iz],
                    self.T_field[step+1, iz], deps_eq
                )

                # K成分
                if step > 0:
                    dT_dt = (self.T_field[step+1, iz] - self.T_field[step, iz]) / self.dt
                    # 温度変化の絶対値を使う（加熱も冷却もエネルギー変化）
                    self.K_components['K_th'][step, iz] = self.rho * self.cp * abs(dT_dt) * 0.01
                else:
                    self.K_components['K_th'][step, iz] = 0

                self.K_components['K_pl'][step, iz] = sigma_eq * deps_eq

                if iz == 0 or iz == self.Nz-1:
                    self.K_components['K_fr'][step, iz] = mu * pN * vslip / self.h
                else:
                    decay = np.exp(-2 * abs(iz - self.Nz//2) / self.Nz)
                    self.K_components['K_fr'][step, iz] = mu * pN * vslip / self.h * decay * 0.1

                self.K_components['K_EM'][step, iz] = 0

                # 仮Λ計算（相変態前）
                t_char = 1.0 / max(deps_eq, 1e-6)
                K_energy = self.K_components['K_pl'][step, iz] * t_char

                if self.ep_maj_total > 1e-6:
                    beta_current = self.ep_min_total / self.ep_maj_total
                else:
                    beta_current = 0.0

                # β依存ゲインの適用
                beta_gain = beta_multiplier(beta_current, A=0.35, bw=0.28)
                K_energy = K_energy * beta_gain

                V_temp = self.V0 * (1 - self.av * self.cv_field[iz] -
                                   self.ad * np.sqrt(self.rhod_field[iz]))
                V_temp = max(V_temp, 1e7)
                Lambda_temp = K_energy / V_temp
                Lambda_temp = min(Lambda_temp, 10.0)

                # 相変態計算
                if step > 0:
                    prev_phases = self.phase_fractions[step-1, iz]
                else:
                    if self.scenario_type == "normal_press":
                        prev_phases = [0.75, 0.25, 0, 0, 0]  # F+P
                    else:  # hot_stamping
                        # 初期温度で判定
                        if self.T_field[0, iz] > 1120:  # 847°C以上
                            prev_phases = [0, 0, 0, 0, 1.0]  # 100%オーステナイト！
                            if iz == self.Nz//2 and step == 0:  # デバッグ
                                print(f"DEBUG: 初期オーステナイト設定 T={self.T_field[0, iz]-273:.1f}°C")
                        else:
                            prev_phases = [0.75, 0.25, 0, 0, 0]  # 常温開始の場合
                            if iz == self.Nz//2 and step == 0:  # デバッグ
                                print(f"DEBUG: F+P設定 T={self.T_field[0, iz]-273:.1f}°C")

                phases, Ms = self._compute_phase_transformation(
                    self.T_field[step+1, iz], Lambda_temp, prev_phases, iz
                )

                # デバッグ：相変態の詳細（中心部のみ）
                if iz == self.Nz//2 and step < 5:  # 最初の5ステップ
                    print(f"DEBUG step={step}: T={self.T_field[step+1, iz]-273:.1f}°C, "
                          f"prev=[{prev_phases[0]:.2f},{prev_phases[1]:.2f},{prev_phases[2]:.2f},"
                          f"{prev_phases[3]:.2f},{prev_phases[4]:.2f}] → "
                          f"new=[{phases[0]:.2f},{phases[1]:.2f},{phases[2]:.2f},"
                          f"{phases[3]:.2f},{phases[4]:.2f}]")

                self.phase_fractions[step, iz] = phases
                self.Ms_field[step, iz] = Ms

                # V_eff（相考慮）
                self.V_eff[step, iz] = self._compute_V_eff_with_phase(
                    self.V0, self.cv_field[iz], self.rhod_field[iz], phases
                )

                # 最終Λ計算
                self.Lambda[step, iz] = K_energy / (self.V_eff[step, iz] *
                                                   np.exp(-self.triax_sens * max(triax, 0)))
                self.Lambda[step, iz] = min(self.Lambda[step, iz], 10.0)

            # 6. 損傷積算
            Lambda_avg = self.Lambda[step].mean()
            if step > 0:
                self.D_damage[step] = self.D_damage[step-1] + max(Lambda_avg - 1, 0) * self.dt
            else:
                self.D_damage[step] = max(Lambda_avg - 1, 0) * self.dt

            # 7. 残留エネルギー
            if step > 0:
                self.E_res[step] = self.E_res[step-1] + q_plastic * self.dt
            else:
                self.E_res[step] = q_plastic * self.dt

            # 8. D_triax（全体）
            self.D_triax[step] = np.exp(-self.triax_sens * max(triax, 0))

    def plot_results(self):
        """統合可視化（6パネル）"""
        fig = plt.figure(figsize=(18, 12))

        # 1. Λの時空間マップ
        ax1 = fig.add_subplot(231)
        im1 = ax1.imshow(self.Lambda.T, aspect='auto', origin='lower',
                        extent=[0, self.t[-2], 0, self.h*1000],
                        cmap='hot', vmin=0, vmax=1.5)
        ax1.contour(self.t[:-1], self.z*1000, self.Lambda.T,
                   levels=[0.5, 0.8, 1.0], colors='white', linewidths=2)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Thickness position [mm]')
        ax1.set_title('EDR (Λ) Space-Time Map')
        plt.colorbar(im1, ax=ax1, label='Λ')

        # 2. 相分率マップ（新規）
        ax2 = fig.add_subplot(232)
        phase_rgb = np.zeros((self.Nt-1, self.Nz, 3))
        phase_rgb[:,:,0] = self.phase_fractions[:,:,1]  # 赤=パーライト
        phase_rgb[:,:,1] = self.phase_fractions[:,:,2]  # 緑=ベイナイト
        phase_rgb[:,:,2] = self.phase_fractions[:,:,3]  # 青=マルテンサイト

        ax2.imshow(phase_rgb.transpose(1,0,2), aspect='auto', origin='lower',
                  extent=[0, self.t[-2], 0, self.h*1000])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Thickness [mm]')
        ax2.set_title('Phase Evolution (R=P, G=B, B=M)')

        # 3. FLC
        ax3 = fig.add_subplot(233)
        eps_maj = self.strain_path[:, 0]
        eps_min = self.strain_path[:, 1]
        Lambda_mean = self.Lambda.mean(axis=1)

        scatter = ax3.scatter(eps_min, eps_maj, c=Lambda_mean,
                            cmap='RdYlGn_r', vmin=0, vmax=1.5, s=10)
        ax3.set_xlabel('Minor strain ε₂')
        ax3.set_ylabel('Major strain ε₁')
        ax3.set_title('Forming Limit Diagram with EDR')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Λ')

        # 4. K成分
        ax4 = fig.add_subplot(234)
        K_th_mean = self.K_components['K_th'].mean(axis=1)
        K_pl_mean = self.K_components['K_pl'].mean(axis=1)
        K_fr_mean = self.K_components['K_fr'].mean(axis=1)

        ax4.fill_between(self.t[:-1], 0, K_th_mean, alpha=0.5, label='K_thermal')
        ax4.fill_between(self.t[:-1], K_th_mean, K_th_mean+K_pl_mean,
                        alpha=0.5, label='K_plastic')
        ax4.fill_between(self.t[:-1], K_th_mean+K_pl_mean,
                        K_th_mean+K_pl_mean+K_fr_mean,
                        alpha=0.5, label='K_friction')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Energy density rate [W/m³]')
        ax4.set_title('K Components Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 温度プロファイル
        ax5 = fig.add_subplot(235)
        times_plot = [0, int(self.Nt/4), int(self.Nt/2), int(3*self.Nt/4), -1]
        for idx in times_plot:
            if idx >= 0 and idx < self.Nt:
                ax5.plot(self.z*1000, self.T_field[idx]-273.15,
                        label=f't={self.t[idx]:.2f}s')
        ax5.set_xlabel('Thickness [mm]')
        ax5.set_ylabel('Temperature [°C]')
        ax5.set_title('Temperature Profile Evolution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Ms温度とΛの関係（新規）
        ax6 = fig.add_subplot(236)
        Lambda_mean_time = self.Lambda.mean(axis=1)
        Ms_mean_time = self.Ms_field.mean(axis=1)

        ax6_twin = ax6.twinx()
        line1 = ax6.plot(self.t[:-1], Lambda_mean_time, 'b-', label='Λ mean')
        ax6.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Λ critical')
        line2 = ax6_twin.plot(self.t[:-1], Ms_mean_time-273.15, 'g-', label='Ms temp')

        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('EDR (Λ)', color='b')
        ax6_twin.set_ylabel('Ms Temperature [°C]', color='g')
        ax6.set_title('Λ-driven Ms Evolution')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_phase_analysis(self):
        """相変態解析専用プロット"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 相分率の時間変化（厚さ平均）
        ax = axes[0, 0]
        phase_mean = self.phase_fractions.mean(axis=1)

        # 5相の場合 [F, P, B, M, A]
        if self.phase_fractions.shape[2] == 5:
            ax.fill_between(self.t[:-1], 0, phase_mean[:, 0],
                           alpha=0.7, label='Ferrite', color='lightgray')
            ax.fill_between(self.t[:-1], phase_mean[:, 0],
                           phase_mean[:, 0] + phase_mean[:, 1],
                           alpha=0.7, label='Pearlite', color='orange')
            ax.fill_between(self.t[:-1], phase_mean[:, 0] + phase_mean[:, 1],
                           phase_mean[:, 0] + phase_mean[:, 1] + phase_mean[:, 2],
                           alpha=0.7, label='Bainite', color='green')
            ax.fill_between(self.t[:-1], phase_mean[:, 0] + phase_mean[:, 1] + phase_mean[:, 2],
                           phase_mean[:, 0] + phase_mean[:, 1] + phase_mean[:, 2] + phase_mean[:, 3],
                           alpha=0.7, label='Martensite', color='blue')
            if phase_mean[:, 4].max() > 0.01:  # オーステナイトがある場合のみ
                ax.fill_between(self.t[:-1],
                               phase_mean[:, 0] + phase_mean[:, 1] + phase_mean[:, 2] + phase_mean[:, 3],
                               1.0, alpha=0.7, label='Austenite', color='yellow')
        else:
            # 4相の場合（旧版）
            ax.fill_between(self.t[:-1], 0, phase_mean[:, 0],
                           alpha=0.7, label='Austenite', color='yellow')
            ax.fill_between(self.t[:-1], phase_mean[:, 0],
                           phase_mean[:, 0] + phase_mean[:, 1],
                           alpha=0.7, label='Pearlite', color='red')
            ax.fill_between(self.t[:-1], phase_mean[:, 0] + phase_mean[:, 1],
                           phase_mean[:, 0] + phase_mean[:, 1] + phase_mean[:, 2],
                           alpha=0.7, label='Bainite', color='green')
            ax.fill_between(self.t[:-1], phase_mean[:, 0] + phase_mean[:, 1] + phase_mean[:, 2],
                           1.0, alpha=0.7, label='Martensite', color='blue')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Phase fraction')
        ax.set_title('Average Phase Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 最終相分率の厚さ分布
        ax = axes[0, 1]
        final_phases = self.phase_fractions[-1]
        width = self.h * 1000 / self.Nz * 0.8
        x = self.z * 1000

        if self.phase_fractions.shape[2] == 5:
            # 5相 [F, P, B, M, A]
            bottom = np.zeros(self.Nz)
            if final_phases[:, 0].max() > 0.01:  # フェライト
                ax.bar(x, final_phases[:, 0], width, bottom=bottom,
                       label='Ferrite', color='lightgray', alpha=0.7)
                bottom += final_phases[:, 0]
            if final_phases[:, 1].max() > 0.01:  # パーライト
                ax.bar(x, final_phases[:, 1], width, bottom=bottom,
                       label='Pearlite', color='orange', alpha=0.7)
                bottom += final_phases[:, 1]
            if final_phases[:, 2].max() > 0.01:  # ベイナイト
                ax.bar(x, final_phases[:, 2], width, bottom=bottom,
                       label='Bainite', color='green', alpha=0.7)
                bottom += final_phases[:, 2]
            if final_phases[:, 3].max() > 0.01:  # マルテンサイト
                ax.bar(x, final_phases[:, 3], width, bottom=bottom,
                       label='Martensite', color='blue', alpha=0.7)
                bottom += final_phases[:, 3]
            if final_phases[:, 4].max() > 0.01:  # オーステナイト
                ax.bar(x, final_phases[:, 4], width, bottom=bottom,
                       label='Austenite', color='yellow', alpha=0.7)
        else:
            # 4相（旧版）
            ax.bar(x, final_phases[:, 1], width, label='Pearlite', color='red', alpha=0.7)
            ax.bar(x, final_phases[:, 2], width, bottom=final_phases[:, 1],
                   label='Bainite', color='green', alpha=0.7)
            ax.bar(x, final_phases[:, 3], width,
                   bottom=final_phases[:, 1] + final_phases[:, 2],
                   label='Martensite', color='blue', alpha=0.7)

        ax.set_xlabel('Thickness [mm]')
        ax.set_ylabel('Phase fraction')
        ax.set_title('Final Microstructure Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. CCT風プロット（改善版）
        ax = axes[1, 0]
        T_history = self.T_field[:, self.Nz//2] - 273.15  # コア温度[°C]
        time_array = self.t[:len(T_history)]

        # 温度履歴をプロット
        ax.plot(time_array, T_history, 'b-', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Temperature [°C]', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_title('Thermal History')
        ax.grid(True, alpha=0.3)

        # 冷却速度を右軸に
        ax2 = ax.twinx()
        cooling_rate = -np.gradient(T_history, self.dt)
        ax2.plot(time_array[1:], cooling_rate[1:], 'r--', alpha=0.6, linewidth=1)
        ax2.set_ylabel('Cooling rate [°C/s]', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(bottom=0)  # 負の値を表示しない

        # ホットスタンピングの場合のみ相変態点をプロット
        if self.scenario_type == "hot_stamping" and self.phase_fractions.shape[2] == 5:
            phases = self.phase_fractions[:, self.Nz//2]
            # マルテンサイト開始
            M_idx = np.where(phases[:, 3] > 0.01)[0]
            if len(M_idx) > 0:
                ax.scatter(time_array[M_idx[0]], T_history[M_idx[0]],
                         c='blue', s=150, marker='v', label=f'Ms={T_history[M_idx[0]]:.0f}°C',
                         zorder=5, edgecolors='black', linewidth=2)

        ax.legend(loc='upper right')

        # 温度範囲の表示
        T_max = T_history.max()
        T_min = T_history.min()
        ax.annotate(f'Max: {T_max:.1f}°C\nMin: {T_min:.1f}°C',
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 4. 硬さ予測（相分率ベース）
        ax = axes[1, 1]
        # 各相の硬さ[HV]（仮定値）
        HV_A = 200
        HV_P = 250
        HV_B = 350
        HV_M = 600

        hardness = (self.phase_fractions[:, :, 0] * HV_A +
                   self.phase_fractions[:, :, 1] * HV_P +
                   self.phase_fractions[:, :, 2] * HV_B +
                   self.phase_fractions[:, :, 3] * HV_M)

        im = ax.imshow(hardness.T, aspect='auto', origin='lower',
                      extent=[0, self.t[-2], 0, self.h*1000],
                      cmap='copper', vmin=200, vmax=600)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Thickness [mm]')
        ax.set_title('Predicted Hardness Evolution [HV]')
        plt.colorbar(im, ax=ax, label='Hardness [HV]')

        plt.tight_layout()
        plt.show()

    def generate_FLC(self):
        """FLC曲線生成"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ひずみ比のバリエーション
        strain_ratios = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
        critical_strains = []

        for ratio in strain_ratios:
            for eps1 in np.linspace(0, 0.8, 50):
                eps2 = ratio * eps1
                eps_eq = np.sqrt(2/3) * np.sqrt(eps1**2 + eps2**2 + (eps1-eps2)**2)
                triax = (eps1 + eps2) / (eps1 - eps2 + 1e-6)
                triax = np.clip(triax, -1, 1)

                K_approx = 1e8 * eps_eq
                V_approx = self.V0 * np.exp(-triax * 0.3)
                Lambda_approx = K_approx / V_approx

                if Lambda_approx > 1.0:
                    critical_strains.append([eps2, eps1])
                    break

        if critical_strains:
            critical_strains = np.array(critical_strains)
            ax1.plot(critical_strains[:, 0], critical_strains[:, 1],
                    'r-', linewidth=3, label='EDR-based FLC (Λ=1)')

        # 安全領域の色分け
        eps2_grid, eps1_grid = np.meshgrid(np.linspace(-0.3, 0.3, 50),
                                          np.linspace(0, 0.6, 50))
        Lambda_grid = np.zeros_like(eps1_grid)

        for i in range(50):
            for j in range(50):
                e1, e2 = eps1_grid[i,j], eps2_grid[i,j]
                e_eq = np.sqrt(2/3) * np.sqrt(e1**2 + e2**2 + (e1-e2)**2)
                Lambda_grid[i,j] = 1e8 * e_eq / self.V0

        im = ax1.contourf(eps2_grid, eps1_grid, Lambda_grid,
                         levels=[0, 0.5, 0.8, 1.0, 1.5],
                         colors=['green', 'yellow', 'orange', 'red'],
                         alpha=0.3)

        ax1.set_xlabel('Minor strain ε₂')
        ax1.set_ylabel('Major strain ε₁')
        ax1.set_title('EDR-based Forming Limit Diagram')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 3D表示
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(eps2_grid, eps1_grid, Lambda_grid,
                               cmap='RdYlGn_r', alpha=0.8)
        ax2.contour(eps2_grid, eps1_grid, Lambda_grid,
                   levels=[1.0], colors='red', linewidths=3)
        ax2.set_xlabel('Minor strain ε₂')
        ax2.set_ylabel('Major strain ε₁')
        ax2.set_zlabel('EDR (Λ)')
        ax2.set_title('3D EDR Landscape')

        plt.tight_layout()
        plt.show()

    def export_results(self):
        """結果のエクスポート（辞書形式）"""
        return {
            'time': self.t[:-1],
            'Lambda': self.Lambda,
            'phase_fractions': self.phase_fractions,
            'temperature': self.T_field,
            'damage': self.D_damage,
            'residual_energy': self.E_res,
            'strain_path': self.strain_path,
            'Ms_field': self.Ms_field,
            'K_components': self.K_components,
            'V_eff': self.V_eff
        }

# ==========================
# デモシナリオ
# ==========================
def demo_actuator_scenario():
    """ニデック向けアクチュエータ部品シナリオ（品質管理版）"""

    # === ひずみ経路（実測値ベース） ===
    def eps_maj(t):
        # 3段階成形プロセス
        if t < 0.15:  # 初期成形
            return 0.15 * (t / 0.15)**1.2
        elif t < 0.3:  # 中間成形
            return 0.15 + 0.15 * ((t - 0.15) / 0.15)**1.0
        else:  # 最終成形
            return 0.3 + 0.1 * ((t - 0.3) / 0.2)**0.8

    def eps_min(t):
        # 絞り比 r = -0.5（典型的な深絞り）
        return -0.5 * eps_maj(t)

    # === 三軸度（成形モードの遷移） ===
    def triax(t):
        if t < 0.1:
            return 0.67  # 等二軸張出し
        elif t < 0.3:
            return 0.33  # 平面ひずみ
        else:
            return 0.0   # 単軸引張

    # === 摩擦管理（潤滑システム） ===
    def mu(t):
        base = 0.12  # 基準摩擦係数（実測値）

        # 潤滑管理ポイント
        if 0.2 < t < 0.22:  # 警告域（軽微な潤滑不足）
            return base * 1.5
        elif 0.35 < t < 0.37:  # 危険域（要注意）
            return base * 2.0
        else:
            return base

    # === 接触圧力（実機データ） ===
    def pN(t):
        # パンチ荷重プロファイル
        if t < 0.05:  # 初期接触
            return 50e6 * (t / 0.05)
        elif t < 0.4:  # 主成形
            return 200e6 + 50e6 * np.sin(2 * np.pi * t / 0.1)  # 振動成分
        else:  # ホールド
            return 150e6

    # === すべり速度（プレス速度） ===
    def vslip(t):
        # SPM（Strokes Per Minute）= 30相当
        v_punch = 0.1  # m/s（100mm/s）
        if t < 0.05:
            return v_punch * (t / 0.05)**0.5  # ソフトスタート
        elif t < 0.4:
            return v_punch
        else:
            return v_punch * 0.1  # 保持速度

    # === 熱管理（金型温度制御） ===
    def htc(t):
        # 熱伝達係数 [W/m²K]
        if t < 0.3:
            return 8000  # 成形中（潤滑油膜あり）
        else:
            return 15000  # 保持中（金型密着）

    def T_die(t):
        # 金型温度管理 [K]
        T_die_set = 353.15  # 80°C（温間成形）

        # PID制御を模擬
        if t < 0.1:
            return 293.15 + (T_die_set - 293.15) * (t / 0.1)
        else:
            # ±5°Cの制御精度
            variation = 5 * np.sin(2 * np.pi * t / 0.2)
            return T_die_set + variation

    return {
        'eps_maj': eps_maj,
        'eps_min': eps_min,
        'triax': triax,
        'mu': mu,
        'pN': pN,
        'vslip': vslip,
        'htc': htc,
        'T_die': T_die
    }

def demo_hot_stamping_scenario():
    """ホットスタンピングシナリオ（発熱抑制版）"""

    def eps_maj(t):
        return 0.3 * (t / 0.3)**0.7 if t < 0.3 else 0.3

    def eps_min(t):
        return -0.15 * (t / 0.3)**1.0 if t < 0.3 else -0.15

    def triax(t):
        return 0.33

    def mu(t):
        # ホットスタンピングは潤滑良好
        return 0.01 if t < 0.3 else 0.0  # 成形後は摩擦ゼロ

    def pN(t):
        # 成形圧を低く
        return 20e6 if t < 0.3 else 0  # 成形後は圧力解放

    def vslip(t):
        # すべり速度を最小化
        return 0.005 if t < 0.3 else 0  # 成形後は完全停止

    def T_die(t):
        # 金型は常温で急冷
        return 293  # 20°C

    def htc(t):
        # 超強力冷却
        if t < 0.3:
            return 20000  # 成形中
        else:
            return 200000  # 成形後は超急冷！

    return {
        'eps_maj': eps_maj,
        'eps_min': eps_min,
        'triax': triax,
        'mu': mu,
        'pN': pN,
        'vslip': vslip,
        'htc': htc,
        'T_die': T_die
    }

def calculate_hardness(phases, eps_eq):
    """現実的な硬さ計算（5相対応）"""
    if len(phases) == 5:
        xiF, xiP, xiB, xiM, xiA = phases
    else:
        # 互換性のため
        xiF, xiP, xiB, xiM = 0.75, 0.25, 0, 0
        xiA = 0

    # 各相の基本硬さ [HV]
    HV_F = 120   # フェライト
    HV_P = 200   # パーライト
    HV_B = 350   # ベイナイト
    HV_M = 600   # マルテンサイト
    HV_A = 150   # オーステナイト（高温）

    # 相分率による加重平均
    HV_base = (HV_F * xiF + HV_P * xiP + HV_B * xiB +
               HV_M * xiM + HV_A * xiA)

    # 加工硬化の寄与（最大50HV増加）
    HV_work = 50 * (1 - np.exp(-3 * eps_eq))

    return HV_base + HV_work

def quality_control_limits():
    """品質管理限界値（マージン考慮版）"""
    return {
        'Lambda_safe': 0.97,       # 安全側マージン
        'Lambda_warning': 0.7,     # 警告レベル
        'Lambda_critical': 0.9,    # 危険レベル
        'Lambda_reject': 1.03,     # 不良品レベル（マージン付き）
        'T_max': 473.15,          # 200°C（最高温度）
        'damage_limit': 0.01,      # 損傷許容値
        'hardness_range': (150, 250)  # HV硬さ範囲（通常プレス）
    }

def evaluate_quality(sim_results):
    """品質判定関数"""
    limits = quality_control_limits()

    Lambda_max = sim_results.Lambda.max()
    T_max = sim_results.T_field.max()
    damage = sim_results.D_damage[-1]

    # 硬さ予測（相分率ベース）
    final_phases = sim_results.phase_fractions[-1].mean(axis=0)
    HV = calculate_hardness(final_phases, sim_results.ep_eq)

    # 判定
    status = "OK"
    issues = []

    if Lambda_max > limits['Lambda_reject']:
        status = "NG"
        issues.append(f"Λ超過: {Lambda_max:.2f}")
    elif Lambda_max > limits['Lambda_critical']:
        status = "要注意"
        issues.append(f"Λ高め: {Lambda_max:.2f}")

    if T_max > limits['T_max']:
        status = "NG"
        issues.append(f"温度超過: {T_max-273.15:.1f}°C")

    if damage > limits['damage_limit']:
        status = "NG"
        issues.append(f"損傷: {damage:.4f}")

    if sim_results.scenario_type == "normal_press":
        if not (limits['hardness_range'][0] <= HV <= limits['hardness_range'][1]):
            status = "要調査"
            issues.append(f"硬さ異常: {HV:.0f}HV")
    elif sim_results.scenario_type == "hot_stamping":
        # ホットスタンピングでは高硬度が正常
        if not (400 <= HV <= 650):
            status = "要調査"
            issues.append(f"硬さ異常: {HV:.0f}HV")

    return {
        'status': status,
        'Lambda_max': Lambda_max,
        'T_max': T_max - 273.15,
        'damage': damage,
        'hardness': HV,
        'issues': issues
    }

# ==========================
# メイン実行
# ==========================
if __name__ == "__main__":
    print("="*60)
    print("プレス成形EDR統合シミュレーション")
    print("Λ³理論による材料劣化・相変態予測")
    print("="*60)

    # シナリオ選択
    print("\nシナリオ選択:")
    print("1: 通常プレス成形（品質管理版）")
    print("2: ホットスタンピング")

    scenario_choice = 1  # ホットスタンピングに変更

    if scenario_choice == 1:
        # 通常プレス
        sim = PressFormingEDR_Advanced(
            scenario_type="normal_press",  # 重要！
            thickness=0.001,
            Nz=11,
            dt=1e-3,
            t_end=0.5,
            V0=1.8e9,     # SPCC相当
            av=3e4,       # 品質管理用に調整
            ad=1e-7,
            triax_sens=0.4,
            Ms0=723,      # 450°C
            Ms_sens=30.0
        )
        scenario = demo_actuator_scenario()
        print("→ 通常プレス成形シナリオを実行")
    else:
        # ホットスタンピング
        sim = PressFormingEDR_Advanced(
            scenario_type="hot_stamping",  # 重要！
            thickness=0.001,
            Nz=11,
            dt=5e-3,      # 時間刻みを大きく
            t_end=3.0,    # 冷却時間を延長
            V0=2.2e9,     # 高強度鋼
            av=2e4,
            ad=5e-8,
            triax_sens=0.3,
            beta_TQ=0.3,  # 熱変換率を下げる
            Ms0=673,      # 400°C
            Ms_sens=40.0
        )

        # 初期温度を設定（重要！！）
        sim.T_field[:] = 1173.15  # 900°C（オーステナイト化温度）

        scenario = demo_hot_stamping_scenario()
        print("→ ホットスタンピングシナリオを実行")
        print(f"  初期温度: {sim.T_field[0,0]-273.15:.1f}°C")

    sim.set_loading_path(**scenario)

    # 実行
    print("\n🔬 シミュレーション実行中...")
    sim.run()

    # ========== デバッグ情報（ホットスタンピング用） ==========
    if scenario_choice == 2:
        print("\n" + "="*50)
        print("デバッグ情報（ホットスタンピング）")
        print("="*50)

        # 温度履歴の確認
        print(f"初期温度（表面）: {sim.T_field[0, 0]-273:.1f}°C")
        print(f"初期温度（中心）: {sim.T_field[0, sim.Nz//2]-273:.1f}°C")
        print(f"最終温度（表面）: {sim.T_field[-1, 0]-273:.1f}°C")
        print(f"最終温度（中心）: {sim.T_field[-1, sim.Nz//2]-273:.1f}°C")
        print(f"Ms温度設定値: {sim.Ms0-273:.1f}°C")

        # 中間時点の温度も確認
        mid_step = len(sim.T_field)//2
        print(f"中間時点温度（中心）: {sim.T_field[mid_step, sim.Nz//2]-273:.1f}°C")

        # 初期相分率の確認（新規追加）
        print(f"\n初期相分率（step=0）:")
        print(f"  F={sim.phase_fractions[0, sim.Nz//2, 0]:.2f}, "
              f"P={sim.phase_fractions[0, sim.Nz//2, 1]:.2f}, "
              f"B={sim.phase_fractions[0, sim.Nz//2, 2]:.2f}, "
              f"M={sim.phase_fractions[0, sim.Nz//2, 3]:.2f}, "
              f"A={sim.phase_fractions[0, sim.Nz//2, 4]:.2f}")

        # 最終相分率の確認
        print(f"\n最終相分率（step={len(sim.phase_fractions)-1}）:")
        if sim.phase_fractions[-1, sim.Nz//2, 4] > 0.9:  # オーステナイト
            print("⚠️ 問題：オーステナイトのまま！")
        elif sim.phase_fractions[-1, sim.Nz//2, 3] > 0.5:  # マルテンサイト
            print("✅ 正常：マルテンサイト変態完了")
        else:
            print(f"  F={sim.phase_fractions[-1, sim.Nz//2, 0]:.2f}, "
                  f"P={sim.phase_fractions[-1, sim.Nz//2, 1]:.2f}, "
                  f"B={sim.phase_fractions[-1, sim.Nz//2, 2]:.2f}, "
                  f"M={sim.phase_fractions[-1, sim.Nz//2, 3]:.2f}, "
                  f"A={sim.phase_fractions[-1, sim.Nz//2, 4]:.2f}")

        print("="*50)
    # ========== デバッグ終了 ==========

    # 可視化
    print("📊 結果可視化中...")
    sim.plot_results()
    sim.plot_phase_analysis()
    sim.generate_FLC()

    # サマリー
    print("\n" + "="*50)
    print("シミュレーション結果サマリー")
    print("="*50)
    print(f"最大Λ（平均）: {sim.Lambda.mean(axis=1).max():.3f}")
    print(f"最大Λ（表面）: {sim.Lambda[:, [0,-1]].max():.3f}")
    print(f"最終損傷度: {sim.D_damage[-1]:.4f}")
    print(f"最高温度: {(sim.T_field.max()-273.15):.1f}°C")
    print(f"残留エネルギー: {sim.E_res[-1]/1e6:.2f} MJ/m³")

    # 最終相分率（5相表示）
    final_phases = sim.phase_fractions[-1].mean(axis=0)
    print(f"\n最終相分率（平均）:")
    print(f"  フェライト: {final_phases[0]:.1%}")
    print(f"  パーライト: {final_phases[1]:.1%}")
    print(f"  ベイナイト: {final_phases[2]:.1%}")
    print(f"  マルテンサイト: {final_phases[3]:.1%}")
    print(f"  オーステナイト: {final_phases[4]:.1%}")

    # 品質判定
    qc_result = evaluate_quality(sim)

    print("\n" + "="*50)
    print("品質管理レポート")
    print("="*50)
    print(f"判定: {qc_result['status']}")
    print(f"最大Λ: {qc_result['Lambda_max']:.3f}")
    print(f"最高温度: {qc_result['T_max']:.1f}°C")
    print(f"損傷度: {qc_result['damage']:.5f}")
    print(f"予測硬さ: {qc_result['hardness']:.0f}HV")

    if qc_result['issues']:
        print(f"問題点: {', '.join(qc_result['issues'])}")
    else:
        print("問題なし")

    # 危険度判定
    Lambda_max = sim.Lambda.max()
    print(f"\n安全性評価:")
    if Lambda_max < 0.5:
        print("✅ 安全 - 生産性向上可能")
    elif Lambda_max < 0.8:
        print("🟡 最適 - バランス良好")
    elif Lambda_max < 1.0:
        print("🟠 注意 - 臨界接近")
    else:
        print("🔴 危険 - 白層形成リスク！")

    print("\n" + "="*50)
    print("シミュレーション完了")
    print("="*50)

    # ==========================
    # FEM連携デモ（新規追加）
    # ==========================
    print("\n" + "="*60)
    print("FEM-EDR連携デモ")
    print("="*60)

    # サンプルFEMデータ生成（実際はCSVファイルから読み込む）
    def generate_sample_fem_data():
        """デモ用のFEMデータを生成"""
        import pandas as pd

        # 時系列データの生成
        times = np.linspace(0, 0.5, 50)
        elements = [1001, 1002, 1003, 1004, 1005]

        data = []
        for elem_id in elements:
            for i, t in enumerate(times):
                # 要素ごとに異なる応力履歴
                stress_factor = 1.0 + 0.2 * (elem_id - 1003)

                data.append({
                    'Time': t,
                    'Element': elem_id,
                    'Stress_Mises': 300e6 * (1 + 0.5*t) * stress_factor,
                    'Strain_Eq': 0.4 * t,
                    'Strain_Rate': 0.8,
                    'Temperature': 293 + 100*t,
                    'Triaxiality': 0.33 + 0.1*np.sin(10*t),
                    'X': (elem_id - 1003) * 10,
                    'Y': 0,
                    'Z': 0
                })

        return pd.DataFrame(data)

    # FEM連携実行
    print("\n1. FEMデータの準備...")
    fem_data = generate_sample_fem_data()
    print(f"   サンプルデータ生成: {len(fem_data)} records")

    print("\n2. FEM-EDRポスト処理の実行...")
    fem_processor = FEM_EDR_PostProcessor(
        V0=sim.V0,
        av=sim.av,
        ad=sim.ad,
        triax_sens=sim.triax_sens
    )

    # EDR計算
    element_history = fem_processor.compute_edr_from_fem(fem_data)

    # レポート生成
    fem_processor.generate_report()

    # 最も危険な要素のプロット
    if fem_processor.critical_elements:
        most_critical = fem_processor.critical_elements[0]['Element_ID']
        print(f"\n3. 最も危険な要素 ({most_critical}) のΛ履歴をプロット...")
        fem_processor.plot_element_history(most_critical)

    # CSVエクスポート
    print("\n4. 結果をCSVファイルにエクスポート...")
    fem_processor.export_results('fem_edr_results.csv')

    print("\n" + "="*60)
    print("FEM連携デモ完了")
    print("="*60)
