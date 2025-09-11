import os.path
import time
import json
import random

from tqdm import tqdm
import numpy as np
import openmc
from openmc import deplete, stats

from config import GeometryConfig, MaterialConfig
from utils.geometry import TVEL, SqrTVEL
from utils.materials import FuelMat, WaterMat, CladdingMat

from utils.experiments import ParamGrid


colors = {
    WaterMat().mat: (120, 120, 255),
    FuelMat().mat: 'green',
    CladdingMat().mat: (100, 100, 100)
}

DEFAULT_PARAMS= {
            'materials': {
                'fuel_enr': 5,
                'fuel_density': 8.3,
                'coolant_density': 1,
                'cladding_density': 2
            },
            'geometry': {
                'tvel_r': 1.0,
                'tvel_dist': 2.0,
                'cladding_thick': 0.1,
                'lat_type':'hex'
            },
            'power': 3000e6 / 270 / 324
        }


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


def update_params(old_params: dict[str, dict], new_params: dict[str]) -> dict[str, dict]:
    for key, value in old_params.items():
        if isinstance(value, dict):
            update_params(value, new_params)
        elif key in new_params.keys():
            old_params[key] = new_params[key]
    return old_params


def iter_param_grid(prob: float = 0.0):
    assert 0 <= prob < 1, "Probability must be between 0 and 1"
    params = DEFAULT_PARAMS
    params_grid = ParamGrid(
        {
            'tvel_r': list(np.arange(0.7, 1.5, 0.2)),
            'tvel_dist': list(np.arange(1.0, 3.0, 0.5)),
            'cladding_thick': [0.2],
            'fuel_density': list([10.]),
            'coolant_density': list(np.arange(0.5, 1.6, 0.5)),
            'cladding_density': [3.],
            'power': [1000, 10_000, 50_000, 100_000],
            'fuel_enr': list(np.arange(3, 5, 0.5)),
            'lat_type':['hex', 'sqr']
        }
    )

    for i, param_dict in enumerate(params_grid.grid):        
        if random.random() < prob:
            continue
        if param_dict["tvel_r"]*2.1>param_dict["tvel_dist"]:
            continue
        params['config_id']=str(i)
        yield update_params(params, param_dict)


def prepare_geometry_and_mats(geom_config: GeometryConfig, mats_config: MaterialConfig, params_: dict = None) -> dict:
    if params_ is None:
        params_ = DEFAULT_PARAMS

    if params_['geometry']['lat_type']=='sqr':
        params_['geometry']['tvel_dist'] *=  0.5 * 3**(3/4)

    geom_config.update(**params_['geometry'])
    mats_config.update(**params_['materials'])

    # Materials
    fuel = FuelMat(enr=mats_config.fuel_enr, density=mats_config.fuel_density, name='Fuel')
    coolant = WaterMat(density=mats_config.coolant_density, name='Coolant')
    ceiling = CladdingMat(density=mats_config.cladding_density, name='Cladding')
    _materials = openmc.Materials([fuel.mat, coolant.mat, ceiling.mat])


    if params_['geometry']['lat_type']=='hex':
        tvel = TVEL(
            g_config=geom_config,
            fuel=fuel,
            ceiling=ceiling,
            coolant=coolant
        )
    elif params_['geometry']['lat_type']=='sqr':
        tvel = SqrTVEL(
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
        output_dir: str = 'results/depletion/v7'
) -> None:
    if params is None:
        params = DEFAULT_PARAMS
    start_time = time.strftime("%Y-%m-%d_%H:%M:%S")
    folder = params.get('config_id',start_time)
    output_dir = os.path.join(output_dir, folder)
    os.makedirs(output_dir, exist_ok=True)

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
    setting.batches = 50
    setting.inactive = 5
    setting.particles = 1_000

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
    time_steps = [500] * 4 

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        print(convert_dtypes(params))
        json.dump(convert_dtypes(params), f)

    integrator = deplete.PredictorIntegrator(operator, time_steps, power, timestep_units='d')
    integrator.integrate()
    params = convert_dtypes(params)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        params['finish_time'] = time.strftime("%Y-%m-%d_%H:%M:%S")
        json.dump(convert_dtypes(params), f)
    # print(params)




def run_one_experiment(
        params: dict[str, dict[str]],
        output_dir: str = 'results/depletion/'
):
    params['start_time'] = time.strftime("%Y-%m-%d_%H:%M:%S")
    return run_experiment(GeometryConfig(), MaterialConfig(), params, output_dir)

def run_all_experiments(start_from=0, output_dir='results/depletion/', skip_prob =0.5):
    for i, p in tqdm(enumerate(iter_param_grid(skip_prob))):
        if i < start_from:
            continue
        else:
            p['experiment_number'] = i
            try:
                run_one_experiment(p, output_dir)
            except Exception as e:
                print(e)
                continue



if __name__ == '__main__':
    run_experiment(GeometryConfig(), MaterialConfig())
