import json
import os

import openmc
import pandas as pd
from lxml import etree
from tqdm import tqdm

from pprint import pprint

from utils.resulting import DepletionResultReader

print(openmc.__version__)

def get_materials_info(results_path):
    mats_path = os.path.join(results_path, "materials.xml")
    tree = etree.parse(mats_path)
    res = {
        'water': None,
        'fuel': None,
        'cladding': None
    }
    for child in tree.getroot():
        info = {
            "volume": float(child.get('volume', 0)),
            "density": float(child.find('density').get('value', 0)),
        }
        if child.get('name') == 'Water':
            res['water'] = info
        elif child.get('name') == 'Fuel':
            info['id'] = child.get('id', '2')
            res['fuel'] = info
        elif child.get('name') == 'Zircaloy-2':
            res['cladding'] = info
        else:
            print("Material not found: ", child.get('name'))
    return res

def merge_context(exp_params: dict, mats: dict) -> dict:
    res = {
        'enr': exp_params['materials'].get('fuel_enr', exp_params.get('enr', 3)),
        'power': exp_params.get('power', 1000)
    }
 
    for mat_name, mat_data in mats.items():
        for key, value in mat_data.items():
            res[f"{mat_name}_{key}"] = value
    return res

def prepare_sample(res_dir: str, save_dir: str = 'neuro/data/') -> None:
    mats_data = get_materials_info(res_dir)
    params = json.load(open(os.path.join(res_dir, "results.json"), 'r'))

    # try:
    #     with open(results_path + os.listdir(results_path)[0] + "/results.json", 'r') as f:
    #         enr = f.read().split('"fuel_enr": ')[1].split(',')[0]
    # except FileNotFoundError:
    #     enr = 3
    # params = {
    #     'enr': float(enr),
    #     "materials": {},
    #     'geometry': {}
    # }
    
    context = merge_context(params, mats_data)

    fuel_id = context.pop('fuel_id', '2')
    dep_res = DepletionResultReader(os.path.join(res_dir, "depletion_results.h5"), fuel_mat=fuel_id)
    df = pd.DataFrame(dep_res.prepare_data())
    os.makedirs(save_dir, exist_ok=True)
    json.dump(context, open(os.path.join(save_dir, "context.json"), 'w'))
    df.to_csv(os.path.join(save_dir, "data.csv"))





if __name__ == '__main__':
    results_path = "results/depletion/v5/"
    print(os.listdir(results_path))
    # res = DepletionResultReader(results_path + os.listdir(results_path)[0] + "/depletion_results.h5", fuel_mat='2')
    # data = res.prepare_data()
    # with open(results_path + os.listdir(results_path)[0] + "/results.json", 'r') as f:
    #     enr = f.read().split('"fuel_enr": ')[1].split(',')[0]
    # params = {
    #     'enr': float(enr),
    #     "materials": {},
    #     'geometry': {}
    # }
    # print(params)
    for result_dir in tqdm(os.listdir(results_path)):
        res_dir = os.path.join(results_path, result_dir)
        save_dir = os.path.join('results/neuro/data/v5', result_dir)
        try:
            prepare_sample(res_dir, save_dir)
        except Exception as e:
            print(f"Failed to prepare sample: {res_dir}", e)