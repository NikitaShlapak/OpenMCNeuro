import os.path
import time

import numpy as np
import openmc
from openmc import deplete, stats

from config import GeometryConfig, MaterialConfig
from utils.geometry import TVEL
from utils.materials import FuelMat, WaterMat, CladdingMat

openmc.config['cross_sections'] = '/run/media/nikita/e40c1d03-27f0-4c5f-b778-1710c9a842d0/data/endfb-vii.1-hdf5/cross_sections.xml'
openmc.config['chain_file'] = '/run/media/nikita/e40c1d03-27f0-4c5f-b778-1710c9a842d0/data/chains/chain_endfb71_pwr.xml'

colors = {
    WaterMat().mat: (120, 120, 255),
    FuelMat().mat: 'green',
    CladdingMat().mat: (100, 100, 100)
}


def smooth_step(_from=1, _to=24, k=1.1):
    steps = [_from]
    while steps[-1] < _to:
        steps.append(steps[-1] * k)
    steps[-1] = _to
    return steps


def convert_dtypes(params: dict) -> dict:
    for key, value in params.items():
        if isinstance(value, dict):
            params[key] = convert_dtypes(value)
        elif isinstance(value, np.int64):
            params[key] = int(value)
        elif isinstance(value, np.float64):
            params[key] = float(value)
        elif isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            continue
        else:
            print(type(value))
            params[key] = str(value)
    return params


def prepare_geometry_and_mats(geom_config: GeometryConfig, mats_config: MaterialConfig, params_: dict = None) -> dict:
    if params_ is None:
        params_ = {
            'materials': {
                'fuel_enr': 5,
                'fuel_density': 8.3,
                'coolant_density': 1,
                'cladding_density': 2
            },
            'geometry': {
                'tvel_r': 1.0,
                'tvel_dist': 2.0,
                'cladding_thick': 0.1
            }
        }

    geom_config.update(**params_['geometry'])
    mats_config.update(**params_['materials'])

    # Materials
    fuel = FuelMat(enr=mats_config.fuel_enr, density=mats_config.fuel_density, name='Fuel')
    coolant = WaterMat(density=mats_config.coolant_density)
    ceiling = CladdingMat(density=mats_config.cladding_density)
    _materials = openmc.Materials([fuel.mat, coolant.mat, ceiling.mat])

    tvel = TVEL(
        g_config=geom_config,
        fuel=fuel,
        ceiling=ceiling,
        coolant=coolant
    )
    tvel.calculate_volumes()
    # Geometry
    _geometry = openmc.Geometry(tvel.universe)
    return {
        'params': params_,
        'materials': _materials,
        'geometry': _geometry,
        'tvel': tvel
    }


def run_experiment(
        geom_config: GeometryConfig,
        mats_config: MaterialConfig,
        params: dict = None,
        output_dir: str = 'results/depletion/'
) -> None:
    if params is None:
        params = {
            'materials': {
                'fuel_enr': 5,
                'fuel_density': 8.3,
                'coolant_density': 1,
                'cladding_density': 2
            },
            'geometry': {
                'tvel_r': 1.0,
                'tvel_dist': 2.0,
                'cladding_thick': 0.1
            },
            'power': 3000e6 / 270 / 324
        }
    start_time = time.strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(output_dir, start_time)

    prepared = prepare_geometry_and_mats(geom_config, mats_config, params)
    tvel = prepared['tvel']
    geometry = prepared['geometry']
    materials = prepared['materials']
    params = prepared['params']
    params['dir'] = output_dir

    plots = [openmc.Plot(), openmc.Plot(), openmc.Plot(), openmc.Plot()]
    width = np.array([geom_config.pitch * 2.5, geom_config.pitch * 2.5])
    for i in range(4):
        plots[i].width = width
        plots[i].pixels = (100, 100)
        plots[i].basis = 'xz'
        plots[i].color_by = 'material'
        plots[i].colors = colors
    plots[0].origin = (0, 0, geom_config.high / 2 - 1)
    plots[2].origin = (0, 0, -geom_config.high / 2 - 1)
    plots[-1].basis = 'xy'
    plots[-1].pixels = (500, 500)

    plots = openmc.Plots(plots)

    # Settings
    setting = openmc.Settings()
    setting.batches = 20
    setting.inactive = 5
    setting.particles = 100_000

    uniform_dist = openmc.stats.Box(tvel.bounding_box[0], tvel.bounding_box[1])
    setting.source = openmc.source.IndependentSource(space=uniform_dist, constraints={'fissionable': True})
    setting.run_mode = 'eigenvalue'
    # Tallies
    flux_tally = openmc.Tally(name='flux')
    flux_tally.scores = ['flux']

    U_tally = openmc.Tally(name='fuel')
    U_tally.scores = ['fission', 'total', 'absorption', 'elastic', 'scatter', 'decay-rate']
    U_tally.nuclides = ['U235', 'U238', 'O16', 'H1']

    tallies = openmc.Tallies([U_tally, flux_tally])

    # XML export
    materials.export_to_xml('config_files/materials.xml')
    geometry.export_to_xml('config_files/geometry.xml')
    setting.export_to_xml('config_files/settings.xml')
    tallies.export_to_xml('config_files/tallies.xml')
    plots.export_to_xml('config_files/plots.xml')
    # print("xml export finished")

    # RUN
    openmc.plot_geometry(path_input='config_files/', output=False)
    model = openmc.Model(geometry=geometry, materials=materials, tallies=tallies, settings=setting)

    # Deplition
    operator = deplete.CoupledOperator(model=model)
    operator.output_dir = output_dir
    power = params['power']
    time_steps = [6] * 4 * 300

    integrator = deplete.PredictorIntegrator(operator, time_steps, power, timestep_units='h')
    integrator.integrate()
    params = convert_dtypes(params)
    # with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    #     params['finish_time'] = time.strftime("%Y-%m-%d_%H:%M:%S")
    #     json.dump(params, f)
    print(params)


if __name__ == '__main__':
    run_experiment(GeometryConfig(), MaterialConfig())
