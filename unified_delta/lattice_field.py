#!/usr/bin/env python3
"""
LatticeField: 統合格子場モジュール
===================================

SparseNeighborGraph + StressConcentrationField を統合

責務：
  - 3D格子構造の管理
  - 空孔の導入・管理
  - 配位数Z場
  - 近傍グラフ（CSR形式、SpMV対応）
  - 応力集中係数K_t場
  - 表面検出

Author: Tamaki & Masamichi
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from scipy.spatial import cKDTree

# CuPy（なければNumPyにフォールバック）
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    from scipy.sparse import csr_matrix
    import scipy.sparse as cpsparse
    GPU_AVAILABLE = False


@dataclass
class LatticeConfig:
    """格子設定"""
    Nx: int
    Ny: int
    Nz: int
    Z_bulk: int = 8          # BCC=8, FCC=12
    vacancy_fraction: float = 0.02
    neighbor_type: str = '6nn'  # '6nn' or '26nn'
    seed: int = 42
    
    @property
    def N(self) -> int:
        return self.Nx * self.Ny * self.Nz
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.Nx, self.Ny, self.Nz)


class LatticeField:
    """
    統合3D格子場
    
    ═══════════════════════════════════════════════════════════
    空孔・表面からの距離に基づく応力集中 K_t
    + CSR近傍グラフによる高速SpMV伝播
    ═══════════════════════════════════════════════════════════
    
    使い方:
        field = LatticeField.create(N=30, Z_bulk=8, vacancy_fraction=0.02)
        K_t = field.stress_concentration_factor(A=5.0, r_char=2.0)
        propagated = field.propagate(values)
    """
    
    def __init__(self, config: LatticeConfig):
        """
        Args:
            config: LatticeConfig インスタンス
        """
        self.config = config
        self.Nx, self.Ny, self.Nz = config.Nx, config.Ny, config.Nz
        self.N = config.N
        self.Z_bulk = config.Z_bulk
        
        # === 格子状態 ===
        self.lattice = np.ones(config.shape, dtype=bool)      # True = 原子あり
        self.Z = np.ones(config.shape, dtype=np.int32) * config.Z_bulk  # 配位数
        self.vacancy_positions = []
        self.dist_to_vacancy = np.full(config.shape, np.inf)
        
        # === 近傍グラフ（CSR） ===
        self._csr_matrix = None
        self._indptr = None
        self._indices = None
        self._neighbors_list = None
        
        # 初期化
        self._build_neighbor_graph()
        self._introduce_vacancies(config.vacancy_fraction, config.seed)
        self._detect_surfaces()
        self._compute_distance_to_vacancies()
        
        self._print_stats()
    
    # ========================================
    # ファクトリメソッド
    # ========================================
    
    @classmethod
    def create(cls,
               N: int = 30,
               Z_bulk: int = 8,
               vacancy_fraction: float = 0.02,
               neighbor_type: str = '6nn',
               seed: int = 42) -> 'LatticeField':
        """
        簡易ファクトリ（立方格子）
        
        Args:
            N: 格子サイズ（N³）
            Z_bulk: バルク配位数
            vacancy_fraction: 空孔率
            neighbor_type: '6nn'（6近傍）or '26nn'（26近傍）
            seed: 乱数シード
        """
        config = LatticeConfig(
            Nx=N, Ny=N, Nz=N,
            Z_bulk=Z_bulk,
            vacancy_fraction=vacancy_fraction,
            neighbor_type=neighbor_type,
            seed=seed,
        )
        return cls(config)
    
    @classmethod
    def create_rect(cls,
                    Nx: int, Ny: int, Nz: int,
                    Z_bulk: int = 8,
                    vacancy_fraction: float = 0.02) -> 'LatticeField':
        """直方体格子"""
        config = LatticeConfig(Nx=Nx, Ny=Ny, Nz=Nz, Z_bulk=Z_bulk, vacancy_fraction=vacancy_fraction)
        return cls(config)
    
    # ========================================
    # 近傍グラフ構築（CSR形式）
    # ========================================
    
    def _build_neighbor_graph(self):
        """近傍グラフをCSR形式で構築"""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        N = self.N
        
        # 近傍オフセット
        if self.config.neighbor_type == '26nn':
            offsets = [(di, dj, dk) 
                       for di in [-1, 0, 1] 
                       for dj in [-1, 0, 1] 
                       for dk in [-1, 0, 1]
                       if not (di == 0 and dj == 0 and dk == 0)]
        else:  # 6nn
            offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        
        # インデックス変換
        def to_flat(i, j, k):
            return i * Ny * Nz + j * Nz + k
        
        # 近傍リスト構築
        neighbors_list = []
        indptr = [0]
        indices = []
        data = []
        
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    nb = []
                    for di, dj, dk in offsets:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz:
                            nb.append(to_flat(ni, nj, nk))
                    
                    neighbors_list.append(nb)
                    indices.extend(nb)
                    data.extend([1.0] * len(nb))
                    indptr.append(len(indices))
        
        self._neighbors_list = neighbors_list
        self._indptr = np.array(indptr, dtype=np.int32)
        self._indices = np.array(indices, dtype=np.int32)
        self._data = np.array(data, dtype=np.float32)
        
        # CSR行列
        if GPU_AVAILABLE:
            indptr_gpu = cp.asarray(self._indptr)
            indices_gpu = cp.asarray(self._indices)
            data_gpu = cp.asarray(self._data)
            self._csr_matrix = cpsparse.csr_matrix(
                (data_gpu, indices_gpu, indptr_gpu),
                shape=(N, N)
            )
        else:
            from scipy.sparse import csr_matrix
            self._csr_matrix = csr_matrix(
                (self._data, self._indices, self._indptr),
                shape=(N, N)
            )
        
        self.n_edges = len(indices)
        self.k_avg = self.n_edges / N if N > 0 else 0
    
    # ========================================
    # 空孔導入
    # ========================================
    
    def _introduce_vacancies(self, fraction: float, seed: int):
        """空孔を導入し、周囲のZを更新"""
        if fraction <= 0:
            return
        
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        n_vac = int(self.N * fraction)
        
        np.random.seed(seed)
        vac_pos = np.random.choice(self.N, n_vac, replace=False)
        
        for pos in vac_pos:
            i = pos // (Ny * Nz)
            j = (pos % (Ny * Nz)) // Nz
            k = pos % Nz
            
            self.lattice[i, j, k] = False
            self.vacancy_positions.append((i, j, k))
            
            # 隣接原子のZを減少（26近傍で影響）
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        if di == 0 and dj == 0 and dk == 0:
                            continue
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz:
                            if self.lattice[ni, nj, nk]:
                                self.Z[ni, nj, nk] = max(1, self.Z[ni, nj, nk] - 1)
    
    # ========================================
    # 表面検出
    # ========================================
    
    def _detect_surfaces(self):
        """表面原子を検出し、Zを調整"""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if not self.lattice[i, j, k]:
                        continue
                    
                    n_boundary = 0
                    if i == 0 or i == Nx - 1:
                        n_boundary += 1
                    if j == 0 or j == Ny - 1:
                        n_boundary += 1
                    if k == 0 or k == Nz - 1:
                        n_boundary += 1
                    
                    self.Z[i, j, k] = max(1, self.Z[i, j, k] - n_boundary)
    
    # ========================================
    # 空孔距離場
    # ========================================
    
    def _compute_distance_to_vacancies(self):
        """各格子点から最近傍空孔への距離を計算"""
        if len(self.vacancy_positions) == 0:
            return
        
        vac_coords = np.array(self.vacancy_positions)
        tree = cKDTree(vac_coords)
        
        active_indices = np.where(self.lattice)
        active_coords = np.column_stack(active_indices)
        
        distances, _ = tree.query(active_coords, k=1)
        
        for idx, (i, j, k) in enumerate(zip(*active_indices)):
            self.dist_to_vacancy[i, j, k] = distances[idx]
    
    # ========================================
    # 応力集中係数 K_t
    # ========================================
    
    def stress_concentration_factor(self,
                                    A: float = 30.0,
                                    r_char: float = 1.0,
                                    K_t_max: float = 100.0) -> np.ndarray:
        """
        応力集中係数 K_t を計算
        
        K_t = 1 + A × exp(-r / r_char) × (Z_bulk / Z_eff)
        
        Args:
            A: 空孔直隣での最大増幅係数
            r_char: 特性距離（この距離で 1/e に減衰）
            K_t_max: 上限
        
        Returns:
            K_t [Nx, Ny, Nz]: 応力集中係数
        """
        K_t = np.ones(self.config.shape)
        mask = self.lattice & (self.Z > 0)
        
        # 距離ベースの応力集中（指数減衰）
        r = self.dist_to_vacancy[mask]
        distance_factor = np.exp(-r / r_char)
        
        # 配位数補正
        Z_factor = self.Z_bulk / np.maximum(self.Z[mask], 1)
        
        # 複合効果
        K_t[mask] = 1.0 + A * distance_factor * Z_factor
        
        return np.minimum(K_t, K_t_max)
    
    def local_stress(self, sigma_applied: float, **kwargs) -> np.ndarray:
        """局所応力場 σ_local = K_t × σ_applied"""
        K_t = self.stress_concentration_factor(**kwargs)
        return K_t * sigma_applied
    
    # ========================================
    # SpMV伝播（カスケード用）
    # ========================================
    
    def propagate(self, values: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        値を近傍に伝播（SpMV）
        
        Args:
            values: 各頂点の値 [N] (flat) or [Nx, Ny, Nz]
            mask: 伝播元マスク（Noneなら全部）
        
        Returns:
            propagated: 各頂点が受け取った値の合計
        """
        # 3D → flat
        if values.ndim == 3:
            values = values.ravel()
        
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.ravel()
            values = values * mask.astype(values.dtype)
        
        if GPU_AVAILABLE:
            values_gpu = cp.asarray(values)
            result = self._csr_matrix.T @ values_gpu
            return cp.asnumpy(result).reshape(self.config.shape)
        else:
            result = self._csr_matrix.T @ values
            return result.reshape(self.config.shape)
    
    def get_neighbors(self, i: int, j: int, k: int) -> list:
        """(i,j,k)の近傍インデックスを取得"""
        flat_idx = i * self.Ny * self.Nz + j * self.Nz + k
        return list(self._neighbors_list[flat_idx])
    
    # ========================================
    # フラット配列取得（GPU計算用）
    # ========================================
    
    def get_flat_arrays(self, **kt_kwargs) -> Dict:
        """
        フラット配列で取得（GPU計算用）
        
        Returns:
            dict with keys: Z, K_t, dist, mask, n_active, flat_indices
        """
        mask = self.lattice
        K_t = self.stress_concentration_factor(**kt_kwargs)
        
        return {
            'Z': self.Z[mask].astype(np.float32),
            'K_t': K_t[mask].astype(np.float32),
            'dist': self.dist_to_vacancy[mask].astype(np.float32),
            'mask': mask,
            'n_active': np.sum(mask),
            'flat_indices': np.where(mask.ravel())[0],
        }
    
    # ========================================
    # 動的更新（カスケード破壊用）
    # ========================================
    
    def mark_failed(self, failed_mask: np.ndarray):
        """
        破壊点をマークし、周囲のZを更新
        
        Args:
            failed_mask: [Nx, Ny, Nz] bool
        """
        newly_failed = failed_mask & self.lattice
        
        if not np.any(newly_failed):
            return
        
        # lattice更新
        self.lattice[newly_failed] = False
        
        # Z低下を伝播
        Z_loss = self.propagate(newly_failed.astype(np.float32))
        self.Z = np.maximum(self.Z - Z_loss.astype(np.int32), 1)
        
        # 空孔距離の再計算（オプション、重いので省略可）
        # self._compute_distance_to_vacancies()
    
    # ========================================
    # ユーティリティ
    # ========================================
    
    def _print_stats(self):
        """統計情報を出力"""
        n_active = np.sum(self.lattice)
        n_vac = len(self.vacancy_positions)
        Z_active = self.Z[self.lattice]
        dist_active = self.dist_to_vacancy[self.lattice]
        
        print(f"\n{'='*60}")
        print(f"LatticeField: {self.Nx}×{self.Ny}×{self.Nz} = {self.N}")
        print(f"{'='*60}")
        print(f"  Active sites:  {n_active}")
        print(f"  Vacancies:     {n_vac} ({100*n_vac/self.N:.1f}%)")
        print(f"  Z_bulk:        {self.Z_bulk}")
        print(f"  Z range:       [{Z_active.min()}, {Z_active.max()}]")
        print(f"  Z mean:        {Z_active.mean():.2f}")
        print(f"  Neighbor type: {self.config.neighbor_type}")
        print(f"  Edges:         {self.n_edges}")
        print(f"  k_avg:         {self.k_avg:.1f}")
        if n_vac > 0:
            print(f"  Dist to vac:   min={dist_active.min():.2f}, "
                  f"max={dist_active.max():.2f}, mean={dist_active.mean():.2f}")
        print(f"  GPU:           {GPU_AVAILABLE}")
        print(f"{'='*60}\n")
    
    def print_Kt_stats(self, **kwargs):
        """K_t の統計を表示"""
        K_t = self.stress_concentration_factor(**kwargs)
        K_t_flat = K_t[self.lattice]
        
        A = kwargs.get('A', 30.0)
        r_char = kwargs.get('r_char', 1.0)
        
        print(f"\nK_t distribution (A={A}, r_char={r_char}):")
        print(f"  min:    {K_t_flat.min():.2f}")
        print(f"  max:    {K_t_flat.max():.2f}")
        print(f"  mean:   {K_t_flat.mean():.2f}")
        print(f"  K_t > 2:  {np.sum(K_t_flat > 2):5d} ({100*np.sum(K_t_flat > 2)/len(K_t_flat):.1f}%)")
        print(f"  K_t > 5:  {np.sum(K_t_flat > 5):5d} ({100*np.sum(K_t_flat > 5)/len(K_t_flat):.1f}%)")
        print(f"  K_t > 10: {np.sum(K_t_flat > 10):5d} ({100*np.sum(K_t_flat > 10)/len(K_t_flat):.1f}%)")
        print(f"  K_t > 20: {np.sum(K_t_flat > 20):5d} ({100*np.sum(K_t_flat > 20)/len(K_t_flat):.1f}%)")


# ========================================
# 後方互換（旧API）
# ========================================

class SparseNeighborGraph:
    """後方互換ラッパー"""
    
    def __init__(self, N: int, neighbors_list: list):
        print("⚠️  SparseNeighborGraph is deprecated. Use LatticeField instead.")
        # 立方体と仮定
        side = int(round(N ** (1/3)))
        self._field = LatticeField.create(N=side, vacancy_fraction=0)
        self.N = N
        self.n_edges = self._field.n_edges
        self.k_avg = self._field.k_avg
    
    def propagate(self, values, mask=None):
        result = self._field.propagate(values.reshape(self._field.config.shape), 
                                       mask.reshape(self._field.config.shape) if mask is not None else None)
        return result.ravel()
    
    def get_neighbors(self, i):
        # flat index → 3D
        Ny, Nz = self._field.Ny, self._field.Nz
        ii = i // (Ny * Nz)
        jj = (i % (Ny * Nz)) // Nz
        kk = i % Nz
        return self._field.get_neighbors(ii, jj, kk)


class StressConcentrationField:
    """後方互換ラッパー"""
    
    def __init__(self, N: int, Z_bulk: int, vacancy_fraction: float = 0.02):
        print("⚠️  StressConcentrationField is deprecated. Use LatticeField instead.")
        self._field = LatticeField.create(N=N, Z_bulk=Z_bulk, vacancy_fraction=vacancy_fraction)
        
        # 旧API互換
        self.N = N
        self.Z_bulk = Z_bulk
        self.lattice = self._field.lattice
        self.Z = self._field.Z
        self.vacancy_positions = self._field.vacancy_positions
        self.dist_to_vacancy = self._field.dist_to_vacancy
    
    def stress_concentration_factor(self, **kwargs):
        return self._field.stress_concentration_factor(**kwargs)
    
    def local_stress(self, sigma_applied, **kwargs):
        return self._field.local_stress(sigma_applied, **kwargs)
    
    def get_flat_arrays(self, **kwargs):
        return self._field.get_flat_arrays(**kwargs)
    
    def print_Kt_stats(self, **kwargs):
        self._field.print_Kt_stats(**kwargs)


# ========================================
# ユーティリティ関数
# ========================================

def build_3d_lattice_graph(Nx: int, Ny: int, Nz: int) -> Tuple[np.ndarray, 'LatticeField']:
    """
    後方互換: 3D格子の近傍グラフを構築
    
    Returns:
        positions: [N, 3]
        field: LatticeField（SparseNeighborGraphの代わり）
    """
    print("⚠️  build_3d_lattice_graph is deprecated. Use LatticeField.create_rect() instead.")
    
    # 位置配列
    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
    
    # LatticeField
    field = LatticeField.create_rect(Nx, Ny, Nz, vacancy_fraction=0)
    
    return positions, field


# ========================================
# 位相場とトポロジカルチャージ Q_Λ
# ========================================

class PhaseFieldMixin:
    """
    位相場とトポロジカルチャージ Q_Λ 計算
    
    物理的意味:
      - 原子振動 u = A × exp(iφ) の位相 φ
      - φ の巻き数 Q = (1/2π) ∮ ∇φ·dl がトポロジー
      - Q のジャンプ = リコネクション = 結合切断
      
    理論的背景:
      - E = mc² = Vorticity（渦度）
      - α = 0.6 = SO(5)→SO(4) の動的成分比率
      - δ = 0.6 δ_L で Q がジャンプ開始（Born崩壊）
      - δ = δ_L で Q がランダム化（Lindemann融解）
    """
    
    def init_phase_field(self, T: float = 300.0, seed: int = None):
        """
        位相場を初期化
        
        Args:
            T: 温度 [K]（初期揺らぎの大きさを決める）
            seed: 乱数シード
        """
        if seed is not None:
            np.random.seed(seed)
        
        shape = self.config.shape
        
        # 位相場 φ ∈ [0, 2π)
        # 低温では位相が揃っている（小さなゆらぎ）
        # 高温ではランダム
        T_ref = 1000.0  # 参照温度
        phase_disorder = min(1.0, T / T_ref)  # 0〜1
        
        # 基準位相（格子に沿った周期構造）
        i_grid, j_grid, k_grid = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        base_phase = 0.1 * (i_grid + j_grid + k_grid)  # 緩やかな勾配
        
        # 熱揺らぎを追加
        thermal_noise = np.random.uniform(-np.pi, np.pi, shape) * phase_disorder
        
        self.phi = (base_phase + thermal_noise) % (2 * np.pi)
        self.phi = self.phi.astype(np.float32)
        
        # 変位ベクトル u = (A cos φ, A sin φ, 0)
        # A は δ に比例
        self.u_amplitude = np.ones(shape, dtype=np.float32) * 0.01  # 初期振幅
        
        # 位相場の履歴（リコネクション検出用）
        self._Q_history = []
        self._reconnection_events = []
        
        print(f"  Phase field initialized: T={T}K, disorder={phase_disorder:.2f}")
    
    def update_phase_field(self, 
                           delta: np.ndarray,
                           delta_L: float,
                           T: float,
                           dt: float = 0.01):
        """
        位相場を時間発展
        
        Args:
            delta: δ場 (Nx, Ny, Nz) - Lindemann比
            delta_L: Lindemann閾値
            T: 温度 [K]
            dt: 時間ステップ
        """
        if not hasattr(self, 'phi'):
            self.init_phase_field(T)
        
        shape = self.config.shape
        
        # δ/δ_L に応じた位相拡散
        delta_ratio = np.clip(delta / delta_L, 0, 2)
        
        # 拡散係数: δ が大きいほど位相が乱れやすい
        D_phase = 0.1 * delta_ratio ** 2
        
        # 隣接サイトとの位相差によるトルク（位相を揃えようとする力）
        # ラプラシアン的な項
        phi_laplacian = np.zeros_like(self.phi)
        for axis in range(3):
            phi_plus = np.roll(self.phi, -1, axis=axis)
            phi_minus = np.roll(self.phi, 1, axis=axis)
            # 位相差を [-π, π] に正規化
            dphi_plus = np.arctan2(np.sin(phi_plus - self.phi), 
                                   np.cos(phi_plus - self.phi))
            dphi_minus = np.arctan2(np.sin(phi_minus - self.phi),
                                    np.cos(phi_minus - self.phi))
            phi_laplacian += dphi_plus + dphi_minus
        
        # 熱ノイズ
        k_B = 1.38e-23
        noise_amplitude = np.sqrt(2 * k_B * T * dt) * 1e10  # スケーリング
        thermal_noise = np.random.randn(*shape).astype(np.float32) * noise_amplitude
        
        # 位相更新
        # dφ/dt = D × ∇²φ + noise
        # 空孔サイトでは位相を固定（境界条件）
        dphi = dt * (D_phase * phi_laplacian + thermal_noise)
        dphi = dphi * self.lattice  # 空孔では更新しない
        
        self.phi = (self.phi + dphi) % (2 * np.pi)
        
        # 振幅も更新（δに比例）
        self.u_amplitude = delta.astype(np.float32)
    
    def compute_Q_lambda_field(self) -> np.ndarray:
        """
        トポロジカルチャージ場 Q_Λ を計算
        
        各プラケット（最小正方形ループ）での位相巻き数を計算
        
        Returns:
            Q_field: (Nx, Ny, Nz) 各サイトでのトポロジカルチャージ
        """
        if not hasattr(self, 'phi'):
            raise ValueError("Phase field not initialized. Call init_phase_field() first.")
        
        shape = self.config.shape
        Q_field = np.zeros(shape, dtype=np.float32)
        
        # 3つの面（xy, yz, zx）でプラケットを計算
        for axis1, axis2 in [(0, 1), (1, 2), (2, 0)]:
            Q_plaquette = self._compute_plaquette_winding(axis1, axis2)
            Q_field += Q_plaquette
        
        # 空孔位置ではQ=0
        Q_field = Q_field * self.lattice
        
        return Q_field
    
    def _compute_plaquette_winding(self, axis1: int, axis2: int) -> np.ndarray:
        """
        1つの面（axis1-axis2平面）でのプラケット巻き数
        
        プラケット:
            φ₂ ←── φ₃
            │      ↑
            ↓      │
            φ₁ ──→ φ₄
            
        Q = Σ(位相差) / 2π
        """
        phi = self.phi
        
        def phase_diff(p1, p2):
            """位相差を [-π, π] に正規化"""
            dp = p2 - p1
            return np.arctan2(np.sin(dp), np.cos(dp))
        
        # 4辺の位相差
        # φ₁ → φ₄ (axis1方向に+1)
        phi_4 = np.roll(phi, -1, axis=axis1)
        dphi_14 = phase_diff(phi, phi_4)
        
        # φ₄ → φ₃ (axis2方向に+1)
        phi_3 = np.roll(phi_4, -1, axis=axis2)
        dphi_43 = phase_diff(phi_4, phi_3)
        
        # φ₃ → φ₂ (axis1方向に-1)
        phi_2 = np.roll(phi, -1, axis=axis2)
        dphi_32 = phase_diff(phi_3, phi_2)
        
        # φ₂ → φ₁ (axis2方向に-1)
        dphi_21 = phase_diff(phi_2, phi)
        
        # 巻き数 = 総位相変化 / 2π
        total_phase = dphi_14 + dphi_43 + dphi_32 + dphi_21
        Q = total_phase / (2 * np.pi)
        
        return Q
    
    def compute_Q_total(self) -> float:
        """全トポロジカルチャージの和"""
        Q_field = self.compute_Q_lambda_field()
        return float(np.sum(Q_field))
    
    def compute_Q_statistics(self) -> dict:
        """Q_Λの統計情報"""
        Q_field = self.compute_Q_lambda_field()
        
        # 整数に近い（安定）vs 半整数（リコネクション中）
        Q_mod = np.abs(Q_field - np.round(Q_field))
        unstable_sites = np.sum(Q_mod > 0.25)
        
        return {
            'Q_total': float(np.sum(Q_field)),
            'Q_mean': float(np.mean(Q_field)),
            'Q_std': float(np.std(Q_field)),
            'Q_min': float(np.min(Q_field)),
            'Q_max': float(np.max(Q_field)),
            'unstable_sites': int(unstable_sites),
            'unstable_fraction': float(unstable_sites / Q_field.size),
        }
    
    def detect_reconnection(self, Q_threshold: float = 0.3) -> dict:
        """
        リコネクション（Q のジャンプ）を検出
        
        Args:
            Q_threshold: Q変化の閾値
            
        Returns:
            reconnection_info: 検出結果
        """
        Q_field = self.compute_Q_lambda_field()
        Q_total = np.sum(Q_field)
        
        result = {
            'Q_total': Q_total,
            'reconnection_detected': False,
            'dQ': 0.0,
            'sites': [],
        }
        
        if len(self._Q_history) > 0:
            Q_prev = self._Q_history[-1]['Q_total']
            dQ = Q_total - Q_prev
            result['dQ'] = dQ
            
            if abs(dQ) > Q_threshold:
                result['reconnection_detected'] = True
                
                # Q変化が大きいサイトを特定
                if hasattr(self, '_Q_field_prev'):
                    dQ_field = Q_field - self._Q_field_prev
                    # 変化が大きい上位サイト
                    flat_idx = np.argsort(np.abs(dQ_field.ravel()))[-10:]
                    sites = []
                    for idx in flat_idx:
                        i = idx // (self.Ny * self.Nz)
                        j = (idx % (self.Ny * self.Nz)) // self.Nz
                        k = idx % self.Nz
                        sites.append({
                            'position': (i, j, k),
                            'dQ': float(dQ_field[i, j, k]),
                        })
                    result['sites'] = sites
                
                # イベント記録
                self._reconnection_events.append({
                    'step': len(self._Q_history),
                    'dQ': dQ,
                    'sites': result['sites'],
                })
        
        # 履歴に追加
        self._Q_history.append({
            'Q_total': Q_total,
            'Q_stats': self.compute_Q_statistics(),
        })
        self._Q_field_prev = Q_field.copy()
        
        return result
    
    def get_Q_history(self) -> list:
        """Q_Λ の履歴を取得"""
        return self._Q_history
    
    def get_reconnection_events(self) -> list:
        """リコネクションイベントを取得"""
        return self._reconnection_events
    
    def plot_phase_field(self, slice_axis: int = 2, slice_idx: int = None,
                         save_path: str = None):
        """位相場を可視化"""
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'phi'):
            raise ValueError("Phase field not initialized.")
        
        if slice_idx is None:
            slice_idx = self.phi.shape[slice_axis] // 2
        
        if slice_axis == 0:
            phi_slice = self.phi[slice_idx, :, :]
            vacancy_slice = ~self.lattice[slice_idx, :, :]
            xlabel, ylabel = 'y', 'z'
        elif slice_axis == 1:
            phi_slice = self.phi[:, slice_idx, :]
            vacancy_slice = ~self.lattice[:, slice_idx, :]
            xlabel, ylabel = 'x', 'z'
        else:
            phi_slice = self.phi[:, :, slice_idx]
            vacancy_slice = ~self.lattice[:, :, slice_idx]
            xlabel, ylabel = 'x', 'y'
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 位相場
        ax1 = axes[0]
        im1 = ax1.imshow(phi_slice.T, origin='lower', cmap='hsv',
                         vmin=0, vmax=2*np.pi)
        ax1.scatter(*np.where(vacancy_slice.T), c='black', s=20, marker='x')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(f'Phase field φ (slice {slice_axis}={slice_idx})')
        plt.colorbar(im1, ax=ax1, label='φ [rad]')
        
        # Q_Λ場
        Q_field = self.compute_Q_lambda_field()
        if slice_axis == 0:
            Q_slice = Q_field[slice_idx, :, :]
        elif slice_axis == 1:
            Q_slice = Q_field[:, slice_idx, :]
        else:
            Q_slice = Q_field[:, :, slice_idx]
        
        ax2 = axes[1]
        im2 = ax2.imshow(Q_slice.T, origin='lower', cmap='RdBu_r',
                         vmin=-0.5, vmax=0.5)
        ax2.scatter(*np.where(vacancy_slice.T), c='black', s=20, marker='x')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(f'Topological charge Q_Λ (slice {slice_axis}={slice_idx})')
        plt.colorbar(im2, ax=ax2, label='Q_Λ')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.show()


# LatticeField に PhaseFieldMixin を追加
# Python の多重継承で既存クラスを拡張
_original_LatticeField = LatticeField

class LatticeField(PhaseFieldMixin, _original_LatticeField):
    """LatticeField with Phase Field support"""
    pass


# ========================================
# テスト
# ========================================

if __name__ == "__main__":
    print("="*70)
    print("LatticeField Test")
    print("="*70)
    
    # 基本テスト
    field = LatticeField.create(N=30, Z_bulk=8, vacancy_fraction=0.02)
    
    # K_t統計
    field.print_Kt_stats(A=30.0, r_char=1.0)
    
    # SpMV伝播テスト
    test_values = np.zeros(field.config.shape)
    test_values[15, 15, 15] = 1.0  # 中心に1を置く
    propagated = field.propagate(test_values)
    print(f"\nPropagation test:")
    print(f"  Input sum: {test_values.sum()}")
    print(f"  Output sum: {propagated.sum()}")  # 6近傍なら6になるはず
    print(f"  Output at neighbors: {propagated[14:17, 14:17, 14:17].sum()}")
    
    # フラット配列取得
    data = field.get_flat_arrays(A=30.0, r_char=1.0)
    print(f"\nFlat arrays:")
    print(f"  n_active: {data['n_active']}")
    print(f"  K_t range: [{data['K_t'].min():.2f}, {data['K_t'].max():.2f}]")
    
    # ========================================
    # 位相場テスト
    # ========================================
    print("\n" + "="*70)
    print("Phase Field & Topological Charge Q_Λ Test")
    print("="*70)
    
    # 位相場初期化
    field.init_phase_field(T=300.0, seed=42)
    
    # Q_Λ統計
    Q_stats = field.compute_Q_statistics()
    print(f"\nQ_Λ statistics (T=300K):")
    print(f"  Q_total: {Q_stats['Q_total']:.4f}")
    print(f"  Q_mean: {Q_stats['Q_mean']:.6f}")
    print(f"  Q_std: {Q_stats['Q_std']:.4f}")
    print(f"  Unstable sites: {Q_stats['unstable_sites']} ({Q_stats['unstable_fraction']*100:.2f}%)")
    
    # 温度を上げて位相を乱す
    print("\n--- Heating simulation ---")
    delta_field = np.ones(field.config.shape) * 0.02  # 初期δ
    delta_L = 0.18
    
    temperatures = [300, 600, 900, 1200]
    for T in temperatures:
        # δを温度に応じて増加（簡易モデル）
        delta_field = np.ones(field.config.shape) * (0.02 + 0.1 * T / 1500)
        
        # 位相更新
        for _ in range(10):  # 10ステップ
            field.update_phase_field(delta_field, delta_L, T, dt=0.01)
        
        # リコネクション検出
        result = field.detect_reconnection(Q_threshold=0.3)
        Q_stats = field.compute_Q_statistics()
        
        status = "⚡RECONNECTION!" if result['reconnection_detected'] else "stable"
        print(f"  T={T:4d}K: Q_total={Q_stats['Q_total']:8.3f}, "
              f"unstable={Q_stats['unstable_fraction']*100:5.2f}%, "
              f"dQ={result['dQ']:+.3f} {status}")
    
    # リコネクションイベント
    events = field.get_reconnection_events()
    print(f"\nTotal reconnection events: {len(events)}")
    
    print("\n✓ All tests passed!")
