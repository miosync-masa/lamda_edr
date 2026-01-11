#!/usr/bin/env python3
"""
Unified δ-Theory Package
========================

δ理論に基づく材料破壊予測の統合パッケージ

モジュール構成:
  - materials.py:         材料データベース
  - lattice_field.py:     3D格子場（空孔、K_t、SpMV伝播）
  - delta_engine.py:      δ計算エンジン（物理計算コア）
  - lifetime_predictor.py: 寿命予測API

使い方:
    from unified_delta import MaterialGPU, LatticeField, DeltaEngine, LifetimePredictor
    
    # 材料
    mat = MaterialGPU.Fe()
    
    # 格子場
    field = LatticeField.create(N=30, Z_bulk=mat.Z_bulk, vacancy_fraction=0.02)
    
    # δエンジン
    engine = DeltaEngine(mat)
    delta = engine.delta_total_vec(sigma, T)
    
    # 寿命予測
    predictor = LifetimePredictor(engine)
    tau = predictor.creep_lifetime(sigma, T)

Author: Tamaki & Masamichi
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Tamaki & Masamichi"

# 材料
from .materials import MaterialGPU

# 格子場
from .lattice_field import (
    LatticeField,
    LatticeConfig,
    # 後方互換
    SparseNeighborGraph,
    StressConcentrationField,
    build_3d_lattice_graph,
)

# δエンジン
from .delta_engine import (
    DeltaEngine,
    DeformationPhase,
    # 物理定数
    k_B,
    u_kg,
    eV_to_J,
    GPU_AVAILABLE,
)

# 寿命予測
from .lifetime_predictor import LifetimePredictor

# 引張試験
from .tensile_test import TensileTest, TensileResult, quick_tensile_test

# 公開API
__all__ = [
    # Core
    'MaterialGPU',
    'LatticeField',
    'LatticeConfig',
    'DeltaEngine',
    'LifetimePredictor',
    'DeformationPhase',
    
    # Tensile Test
    'TensileTest',
    'TensileResult',
    'quick_tensile_test',
    
    # Constants
    'k_B',
    'u_kg',
    'eV_to_J',
    'GPU_AVAILABLE',
    
    # Deprecated (backward compat)
    'SparseNeighborGraph',
    'StressConcentrationField',
    'build_3d_lattice_graph',
]


def info():
    """パッケージ情報を表示"""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Unified δ-Theory Package v{__version__}                      ║
╠══════════════════════════════════════════════════════════╣
║  Materials:       Fe, Cu, Al, Ni, W, Ti, Mg, Au, Ag, Zn  ║
║  GPU:             {'✓ CuPy available' if GPU_AVAILABLE else '✗ CPU mode (NumPy)'}             ║
║  Author:          {__author__}                         ║
╚══════════════════════════════════════════════════════════╝
""")


def quick_test():
    """クイックテスト"""
    print("Running quick test...")
    
    # 材料
    mat = MaterialGPU.Fe()
    print(f"✓ Material: {mat}")
    
    # 格子場
    field = LatticeField.create(N=10, Z_bulk=mat.Z_bulk, vacancy_fraction=0.01)
    print(f"✓ LatticeField: {field.N} sites")
    
    # δエンジン
    engine = DeltaEngine(mat)
    import numpy as np
    T = np.array([300.0, 600.0, 900.0])
    delta = engine.delta_thermal_vec(T)
    print(f"✓ DeltaEngine: δ_thermal = {delta}")
    
    # 寿命予測
    predictor = LifetimePredictor(engine)
    sigma = np.array([100e6])
    tau = predictor.creep_lifetime(sigma, np.array([600.0]))
    print(f"✓ LifetimePredictor: τ = {tau[0]:.2e} s")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    info()
    quick_test()
