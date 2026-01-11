#!/usr/bin/env python3
"""
Materials: 材料データベース
===========================

δ理論で使用する材料パラメータ

Author: Tamaki & Masamichi
"""

from dataclasses import dataclass


@dataclass
class MaterialGPU:
    """
    GPU用材料データ
    
    δ_L（Lindemann閾値）は材料固有の実験値
    fG（Born崩壊係数）は δ_L から逆算してフィッティング
    """
    name: str           # 材料名
    structure: str      # 結晶構造 (BCC, FCC, HCP)
    Z_bulk: int         # バルク配位数
    a_300K: float       # 300Kでの格子定数 [m]
    alpha: float        # 線膨張係数 [1/K]
    E0: float           # ヤング率 [Pa]
    nu: float           # ポアソン比
    T_melt: float       # 融点 [K]
    M_amu: float        # 原子質量 [amu]
    rho: float          # 密度 [kg/m³]
    delta_L: float      # Lindemann閾値
    lambda_base: float  # 熱軟化パラメータ
    kappa: float        # 非線形熱軟化係数
    E_bond_eV: float    # 結合エネルギー [eV]
    fG: float           # Born崩壊係数（融点でのG/G₀）
    
    # ========================================
    # BCC 金属
    # ========================================
    
    @classmethod
    def Fe(cls) -> 'MaterialGPU':
        """α鉄（BCC）- SECD相当"""
        return cls(
            name="Fe", structure="BCC", Z_bulk=8,
            a_300K=2.92e-10, alpha=1.50e-5,
            E0=211e9, nu=0.29, T_melt=1811,
            M_amu=55.845, rho=7870,
            delta_L=0.18,
            lambda_base=49.2, kappa=0.573,
            E_bond_eV=4.28,
            fG=0.027,
        )
    
    @classmethod
    def W(cls) -> 'MaterialGPU':
        """タングステン（BCC）"""
        return cls(
            name="W", structure="BCC", Z_bulk=8,
            a_300K=3.16e-10, alpha=4.51e-6,
            E0=411e9, nu=0.28, T_melt=3695,
            M_amu=183.84, rho=19300,
            delta_L=0.16,
            lambda_base=10.9, kappa=2.759,
            E_bond_eV=8.90,
            fG=0.021,
        )
    
    # ========================================
    # FCC 金属
    # ========================================
    
    @classmethod
    def Cu(cls) -> 'MaterialGPU':
        """銅（FCC）"""
        return cls(
            name="Cu", structure="FCC", Z_bulk=12,
            a_300K=3.61e-10, alpha=1.70e-5,
            E0=130e9, nu=0.34, T_melt=1357,
            M_amu=63.546, rho=8960,
            delta_L=0.10,
            lambda_base=26.3, kappa=1.713,
            E_bond_eV=3.49,
            fG=0.101,
        )
    
    @classmethod
    def Al(cls) -> 'MaterialGPU':
        """アルミニウム（FCC）"""
        return cls(
            name="Al", structure="FCC", Z_bulk=12,
            a_300K=4.05e-10, alpha=2.30e-5,
            E0=70e9, nu=0.35, T_melt=933,
            M_amu=26.982, rho=2700,
            delta_L=0.10,
            lambda_base=27.3, kappa=4.180,
            E_bond_eV=3.39,
            fG=0.101,
        )
    
    @classmethod
    def Ni(cls) -> 'MaterialGPU':
        """ニッケル（FCC）"""
        return cls(
            name="Ni", structure="FCC", Z_bulk=12,
            a_300K=3.52e-10, alpha=1.30e-5,
            E0=200e9, nu=0.31, T_melt=1728,
            M_amu=58.693, rho=8900,
            delta_L=0.11,
            lambda_base=22.6, kappa=0.279,
            E_bond_eV=4.44,
            fG=0.092,
        )
    
    @classmethod
    def Au(cls) -> 'MaterialGPU':
        """金（FCC）"""
        return cls(
            name="Au", structure="FCC", Z_bulk=12,
            a_300K=4.08e-10, alpha=1.42e-5,
            E0=79e9, nu=0.44, T_melt=1337,
            M_amu=196.967, rho=19300,
            delta_L=0.10,
            lambda_base=25.0, kappa=1.5,
            E_bond_eV=3.81,
            fG=0.101,
        )
    
    @classmethod
    def Ag(cls) -> 'MaterialGPU':
        """銀（FCC）"""
        return cls(
            name="Ag", structure="FCC", Z_bulk=12,
            a_300K=4.09e-10, alpha=1.89e-5,
            E0=83e9, nu=0.37, T_melt=1235,
            M_amu=107.868, rho=10490,
            delta_L=0.10,
            lambda_base=24.0, kappa=1.8,
            E_bond_eV=2.95,
            fG=0.101,
        )
    
    # ========================================
    # HCP 金属
    # ========================================
    
    @classmethod
    def Ti(cls) -> 'MaterialGPU':
        """チタン（HCP）"""
        return cls(
            name="Ti", structure="HCP", Z_bulk=12,
            a_300K=2.95e-10, alpha=8.60e-6,
            E0=116e9, nu=0.32, T_melt=1941,
            M_amu=47.867, rho=4500,
            delta_L=0.10,
            lambda_base=43.1, kappa=0.771,
            E_bond_eV=4.85,
            fG=0.101,
        )
    
    @classmethod
    def Mg(cls) -> 'MaterialGPU':
        """マグネシウム（HCP）"""
        return cls(
            name="Mg", structure="HCP", Z_bulk=12,
            a_300K=3.21e-10, alpha=2.70e-5,
            E0=45e9, nu=0.29, T_melt=923,
            M_amu=24.305, rho=1740,
            delta_L=0.117,
            lambda_base=7.5, kappa=37.568,
            E_bond_eV=1.51,
            fG=0.082,
        )
    
    @classmethod
    def Zn(cls) -> 'MaterialGPU':
        """亜鉛（HCP）"""
        return cls(
            name="Zn", structure="HCP", Z_bulk=12,
            a_300K=2.66e-10, alpha=3.02e-5,
            E0=108e9, nu=0.25, T_melt=693,
            M_amu=65.38, rho=7140,
            delta_L=0.12,
            lambda_base=15.0, kappa=5.0,
            E_bond_eV=1.35,
            fG=0.075,
        )
    
    # ========================================
    # エイリアス（後方互換）
    # ========================================
    
    @classmethod
    def SECD(cls) -> 'MaterialGPU':
        """SECD鋼 = Fe"""
        return cls.Fe()
    
    @classmethod
    def FCC_Cu(cls) -> 'MaterialGPU':
        """後方互換"""
        return cls.Cu()
    
    @classmethod
    def Iron(cls) -> 'MaterialGPU':
        """エイリアス"""
        return cls.Fe()
    
    @classmethod
    def Copper(cls) -> 'MaterialGPU':
        """エイリアス"""
        return cls.Cu()
    
    @classmethod
    def Aluminum(cls) -> 'MaterialGPU':
        """エイリアス"""
        return cls.Al()
    
    # ========================================
    # 一覧取得
    # ========================================
    
    @classmethod
    def list_materials(cls) -> list:
        """利用可能な材料一覧"""
        return ['Fe', 'W', 'Cu', 'Al', 'Ni', 'Au', 'Ag', 'Ti', 'Mg', 'Zn']
    
    @classmethod
    def get(cls, name: str) -> 'MaterialGPU':
        """名前から材料を取得"""
        materials = {
            'Fe': cls.Fe, 'Iron': cls.Fe, 'SECD': cls.Fe,
            'W': cls.W, 'Tungsten': cls.W,
            'Cu': cls.Cu, 'Copper': cls.Cu, 'FCC_Cu': cls.Cu,
            'Al': cls.Al, 'Aluminum': cls.Al,
            'Ni': cls.Ni, 'Nickel': cls.Ni,
            'Au': cls.Au, 'Gold': cls.Au,
            'Ag': cls.Ag, 'Silver': cls.Ag,
            'Ti': cls.Ti, 'Titanium': cls.Ti,
            'Mg': cls.Mg, 'Magnesium': cls.Mg,
            'Zn': cls.Zn, 'Zinc': cls.Zn,
        }
        if name not in materials:
            raise ValueError(f"Unknown material: {name}. Available: {cls.list_materials()}")
        return materials[name]()
    
    # ========================================
    # 表示
    # ========================================
    
    def __str__(self) -> str:
        return (f"MaterialGPU({self.name}, {self.structure}, "
                f"δ_L={self.delta_L}, T_m={self.T_melt}K)")
    
    def summary(self) -> str:
        """詳細サマリー"""
        return f"""
{'='*50}
Material: {self.name} ({self.structure})
{'='*50}
  Structure:
    Z_bulk     = {self.Z_bulk}
    a_300K     = {self.a_300K*1e10:.3f} Å
    α          = {self.alpha*1e6:.2f} ×10⁻⁶ /K
  
  Mechanical:
    E₀         = {self.E0/1e9:.0f} GPa
    ν          = {self.nu}
    ρ          = {self.rho} kg/m³
  
  Thermal:
    T_melt     = {self.T_melt} K
    λ_base     = {self.lambda_base}
    κ          = {self.kappa}
  
  δ-Theory:
    δ_L        = {self.delta_L}
    E_bond     = {self.E_bond_eV} eV
    fG         = {self.fG}
{'='*50}
"""


# ========================================
# テスト
# ========================================

if __name__ == "__main__":
    print("Available materials:", MaterialGPU.list_materials())
    print()
    
    for name in ['Fe', 'Cu', 'Al']:
        mat = MaterialGPU.get(name)
        print(mat.summary())
