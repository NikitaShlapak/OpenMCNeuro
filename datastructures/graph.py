from lxml import etree
import os
from dataclasses import dataclass
import pandas as pd
import json

BASE_DIR = "/run/media/nikita/e40c1d03-27f0-4c5f-b778-1710c9a842d0/data/server_sync/data/v6/"

@dataclass
class NeutronicGraphNode:
    node_type:str
    node_data:list
    node_name:str
    node_id:int


class MatSurfGraph():
    materials:dict=None
    planes:dict=None
    cylinders:dict=None
    edges:list=None
    isotopes:tuple = None
    
    def __init__(self, nodes:list[NeutronicGraphNode], isotopes:tuple=None):

        if self.isotopes is None:
            self.isotopes = isotopes
        assert self.isotopes is not None

        self.materials = {}
        self.cylinders = {}
        self.planes = {}
        pre_edges = []
        for node in nodes:
            if node.node_type == 'material':                
                mat_comp = {x:0 for x in self.isotopes}
                isos, vol_dens = node.node_data
                for nuc_name, nuc_count in isos.items():
                    mat_comp[nuc_name]=float(nuc_count)
                volume = float(vol_dens.get('volume'))
                density = float(vol_dens.get('density'))
                data = list(mat_comp.values())+[volume, density]
                self.materials[str(node.node_id)]=data
            elif node.node_type == 'surface':
                data = [float(x) for x in node.node_data]
                if 'plane' in node.node_name:
                    self.planes[str(node.node_id)]=data
                elif 'cylinder' in node.node_name:
                    self.cylinders[str(node.node_id)]=data
            elif node.node_type =='cell':
                node_data = node.node_data[0]
                mat_id = node_data['material']
                for surf_id in node_data['surfs']:
                    pre_edges.append((mat_id, surf_id))
        
        self.edges = self._convert_edges(pre_edges=pre_edges)
    
    def _convert_edges(self, pre_edges:list[tuple[int, int]])->list:
        edges = []
        num_mats = len(self.materials)
        mat_ids = self.all_node_ids[:num_mats]
        surf_ids = self.all_node_ids[num_mats:]
        for mat, surf in pre_edges:
            mat_ind = mat_ids.index(str(mat))
            surf_ind = num_mats + surf_ids.index(str(surf))
            edges.append((mat_ind, surf_ind))
            edges.append((surf_ind, mat_ind))

        return edges            


    
    @property
    def all_nodes(self) -> list:
        return list(self.materials.values()) + list(self.cylinders.values()) + list(self.planes.values())
    
    @property
    def all_node_ids(self) -> list:
        return list(self.materials.keys()) + list(self.cylinders.keys()) + list(self.planes.keys())

def load_graph(
    directory:str=os.path.join(BASE_DIR, '0'), 
    materials_filename:str='materials.xml',
    geometry_filename:str='geometry.xml',
    nucliedes:list|str=None
    ) -> MatSurfGraph:
    if nucliedes is None:
        nucliedes = 'graph_nucliedes.txt'
    if isinstance(nucliedes, str):
        with open(nucliedes, 'r') as f:
            nucliedes = f.read().split()
    # print(nucliedes)
    geom_xml = etree.parse(os.path.join(directory, 'geometry.xml')).getroot()
    mats_xml = etree.parse(os.path.join(directory, 'materials.xml')).getroot()

    mats, cells, surfs = {}, {}, {}
    for mat in mats_xml:
        # print(mat.tag, mat.keys(), mat.values())
        _el = {
                key:value for key,value in zip(mat.keys(), mat.values()) if key!='id' 
            } | {'nuclides':[], 'density':0}
        for item in mat:
            if item.tag == 'density':
                _el['density']=item.get('value')
            elif item.tag=='nuclide':
                _el['nuclides'].append(item.values())    
        mats[mat.get('id')] = _el
    for el in geom_xml:
        # print(el.tag, el.keys(), el.values())
        _el = {
                key:value for key,value in zip(el.keys(), el.values()) if key!='id' 
            }
        if el.tag == 'cell':
            cells[el.get('id')]=_el
        else:
            surfs[el.get('id')]=_el

    neutro_nodes = []
    for mat_id, mat_data in mats.items():
        _type = 'material'
        name = mat_data.pop('name')

        isos = mat_data.pop('nuclides')
        iso_dict = {key:0 for key in nucliedes}
        for iso_val, iso_name in isos:
            iso_dict[iso_name]=float(iso_val)
        
        node_data = [iso_dict, mat_data]

        # print(node_data)
        node = NeutronicGraphNode(
            node_type = _type,
            node_data = node_data,
            node_name = name,
            node_id = int(mat_id)
        )
        # print(node)
        neutro_nodes.append(node)
    for surface_id, surface_data in surfs.items():
        # print(surface_data)
        _type = "surface"
        if surface_data['type'] in ['z-cylinder', 'plane'] :
            surf_data = surface_data['coeffs'].split()
        elif surface_data['type'] == 'x-plane':
            surf_data = [1, 0, 0, float(surface_data['coeffs'])]
        elif surface_data['type'] == 'y-plane':
            surf_data = [0, 1, 0, float(surface_data['coeffs'])]
        elif surface_data['type'] == 'z-plane':
            surf_data = [0, 0, 1, float(surface_data['coeffs'])]

        node = NeutronicGraphNode(
            node_type = _type,
            node_data = surf_data,
            node_name = surface_data['type'],
            node_id = int(surface_id)
        )
        # print(node)
        neutro_nodes.append(node)
    for cell_id, cell_data in cells.items():
        # print(cell_data)
        _type = 'cell'
        _data = {
            'material': int(cell_data['material']),
            'surfs':[int(x) for x in cell_data['region'].replace('-', '').split(' ')]
        }
        node = NeutronicGraphNode(
            node_type = _type,
            node_data = [_data],
            node_name = f"Cell {cell_id}",
            node_id = int(cell_id)
        )
        # print(node)
        neutro_nodes.append(node)
    return MatSurfGraph(neutro_nodes, isotopes=nucliedes)

def load_power(basedir:str, filename='context.json'):
    with open(os.path.join(basedir, filename), 'r') as f:
        data = json.load(f)
        return data['power']

def load_x(
    directory:str=os.path.join(BASE_DIR, '0'), 
    materials_filename:str='materials.xml',
    geometry_filename:str='geometry.xml',
    nucliedes:list|str=None,
    context_filename:str='context.json'
    ):
    return load_graph(
        directory=directory,
        materials_filename=materials_filename,
        geometry_filename=geometry_filename,
        nucliedes=nucliedes
    ), load_power(
        basedir=directory,
        filename=context_filename
    )

def load_res(resultfilepath:str):
    results = pd.read_csv(resultfilepath, index_col=0)
    return results.k_inf.values[-1]-1

def load_pair(
    basedir:str = BASE_DIR,
    folder:str = '0',
    results_filename:str = 'data.csv',
    materials_filename:str='materials.xml',
    geometry_filename:str='geometry.xml',
    nucliedes:list|str=None,
    context_filename:str='context.json'
) -> tuple[MatSurfGraph, float]:
    return load_x(
            directory=os.path.join(basedir, folder),
            materials_filename = materials_filename,
            geometry_filename = geometry_filename,
            nucliedes = nucliedes,
            context_filename=context_filename
    ), load_res(
        os.path.join(basedir, folder, results_filename)
    )