from math import sqrt, pi

import openmc
from matplotlib import pyplot as plt

from config import GeometryConfig
from utils.materials import FuelMat, WaterMat, CladdingMat


class TVEL:
    universe = openmc.Universe()
    bounding_box = ([0, 0, 0], [0, 0, 0])

    def __init__(self,
                 fuel: FuelMat,
                 coolant: WaterMat,
                 ceiling: CladdingMat,
                 g_config: GeometryConfig = None
                 ):
        if g_config is None:
            g_config = GeometryConfig()

        self.g_config = g_config

        self.mats = {
            'fuel': fuel,
            'water': coolant,
            'cladding': ceiling,
        }
        up = g_config.high / 2
        bot = -g_config.high / 2
        # self.calculate_volumes()

        self.bounding_box = ([-g_config.pitch, -g_config.pitch, bot], [g_config.pitch, g_config.pitch, up])
        fuel_clad_surf = openmc.ZCylinder(r=g_config.tvel_r - g_config.cladding_thick)
        clad_water_surf = openmc.ZCylinder(r=g_config.tvel_r)
        water_surf = openmc.model.HexagonalPrism(
            edge_length=g_config.pitch,
            orientation='x',
            boundary_type='reflective'
        )

        top_surf = openmc.ZPlane(z0=up)
        bottom_surf = openmc.ZPlane(z0=bot)

        top_surf.boundary_type = 'vacuum'
        bottom_surf.boundary_type = 'vacuum'

        fuel_cell = openmc.Cell(fill=self.mats['fuel'].mat, region=-fuel_clad_surf & +bottom_surf & -top_surf)
        cladding_cell = openmc.Cell(fill=self.mats['cladding'].mat,
                                    region=+fuel_clad_surf & -clad_water_surf & +bottom_surf & -top_surf)
        water_cell = openmc.Cell(fill=self.mats['water'].mat,
                                 region=+clad_water_surf & -water_surf & +bottom_surf & -top_surf)

        self.universe = openmc.Universe(cells=[fuel_cell, cladding_cell, water_cell])

    def calculate_volumes(self):
        fuel_volume = pi * (self.g_config.tvel_r - self.g_config.cladding_thick) ** 2
        clad_volume = pi * (self.g_config.tvel_r ** 2 - self.g_config.cladding_thick ** 2)
        water_volume = 3 * sqrt(3) / 2 * self.g_config.pitch ** 2 - pi * self.g_config.tvel_r ** 2

        self.mats['fuel'].set_volume(fuel_volume * self.g_config.high)
        self.mats['cladding'].set_volume(clad_volume * self.g_config.high)
        self.mats['water'].set_volume(water_volume * self.g_config.high)

    def plot(self, color_data: dict, width=(100, 100), savedir='results/plots/geometry.jpg'):
        fig, ax = plt.subplots(2, 2)
        self.universe.plot(
            width=width,
            pixels=(250, 250),
            basis='xz',
            **color_data,
            origin=(0, 0, self.g_config.high / 2),
            axes=ax[0][0]
        )
        self.universe.plot(
            width=width,
            pixels=(250, 250),
            basis='xz',
            **color_data,
            origin=(0, 0, 0),
            axes=ax[1][1]
        )
        self.universe.plot(
            width=width,
            pixels=(250, 250),
            basis='xz',
            **color_data,
            origin=(0, 0, -self.g_config.high / 2),
            axes=ax[0][1]
        )
        self.universe.plot(
            width=width,
            pixels=(250, 250),
            basis='xy',
            **color_data,
            origin=(0, 0, 0),
            axes=ax[1][0]
        )
        plt.savefig(savedir)

class SqrTVEL(TVEL):
    def __init__(self,
                fuel: FuelMat,
                coolant: WaterMat,
                ceiling: CladdingMat,
                g_config: GeometryConfig = None
                ):
        if g_config is None:
            g_config = GeometryConfig(lat_type='sqr')

        self.g_config = g_config

        self.mats = {
            'fuel': fuel,
            'water': coolant,
            'cladding': ceiling,
        }
        up = g_config.high / 2
        bot = -g_config.high / 2
        # self.calculate_volumes()

        self.bounding_box = ([-g_config.pitch, -g_config.pitch, bot], [g_config.pitch, g_config.pitch, up])

        fuel_clad_surf = openmc.ZCylinder(r=g_config.tvel_r - g_config.cladding_thick)
        clad_water_surf = openmc.ZCylinder(r=g_config.tvel_r)
        water_surf = openmc.model.RectangularPrism(
            width=g_config.pitch*2,
            height=g_config.pitch*2,
            axis='z',
            boundary_type='reflective'
        )

        top_surf = openmc.ZPlane(z0=up)
        bottom_surf = openmc.ZPlane(z0=bot)

        top_surf.boundary_type = 'vacuum'
        bottom_surf.boundary_type = 'vacuum'

        fuel_cell = openmc.Cell(fill=self.mats['fuel'].mat, region=-fuel_clad_surf & +bottom_surf & -top_surf)
        cladding_cell = openmc.Cell(fill=self.mats['cladding'].mat,
                                    region=+fuel_clad_surf & -clad_water_surf & +bottom_surf & -top_surf)
        water_cell = openmc.Cell(fill=self.mats['water'].mat,
                                region=+clad_water_surf & -water_surf & +bottom_surf & -top_surf)

        self.universe = openmc.Universe(cells=[fuel_cell, cladding_cell, water_cell])

    def calculate_volumes(self):
        fuel_volume = pi * (self.g_config.tvel_r - self.g_config.cladding_thick) ** 2
        clad_volume = pi * (self.g_config.tvel_r ** 2 - self.g_config.cladding_thick ** 2)
        water_volume = 4 * self.g_config.pitch ** 2 - pi * self.g_config.tvel_r ** 2

        self.mats['fuel'].set_volume(fuel_volume * self.g_config.high)
        self.mats['cladding'].set_volume(clad_volume * self.g_config.high)
        self.mats['water'].set_volume(water_volume * self.g_config.high)
