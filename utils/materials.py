import openmc
from singleton_decorator import singleton
import neutronics_material_maker as nmm


class Material:
    def __init__(self, name: str, density: float, *args, **kwargs):
        self.mat = openmc.Material(name=name)
        self.mat.set_density('g/cm3', density)

    def set_volume(self, volume: int):
        self.mat.volume = volume


@singleton
class FuelMat(Material):
    def __init__(self, enr=5, density=8.3, name='Fuel'):
        super().__init__(name, density)
        self.mat.add_element('U', 1, enrichment=enr)
        self.mat.add_element('O', 2)


@singleton
class WaterMat(Material):
    def __init__(self, name='Water', density=1):
        super().__init__(name, density)
        self.mat.add_element('O', 1)
        self.mat.add_element('H', 2)

@singleton
class AbsorberMat(Material):
    def __init__(self, name='Absorber', density=1):
        super().__init__(name, density)
        self.mat.add_element('B', 1)
        self.mat.add_element('O', 2)

@singleton
class CladdingMat(Material):
    def __init__(self, name='Cladding', density=1):
        super().__init__(name, density)
        self.mat = nmm.Material.from_library(name='Zircaloy-2').openmc_material
