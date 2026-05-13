from datastructures.graph import load_res, load_x, load_graph, MatSurfGraph
from neural_models.gcn import MatSurfGcn, GorynychConv, MatSurfGAT
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import json
from utils.experiments import ParamGrid
import numpy as np

SAVE_DIR = "/home/nikita/PycharmProjects/OpenMC/neuro/models/v7/first_SuperGATConv_2l"
BASE_DIR = "/run/media/nikita/e40c1d03-27f0-4c5f-b778-1710c9a842d0/data/server_sync/data/v7/"

K_MIN = 0.2526038758680176
K_MAX = 1.4825291427183294
# K_MIN = -0.6588020698598843
# K_MAX = 0.2819259768802344

SW_SCALE = 1.4583333333333335

WU_min = 1.0
WU_max = 2.1

with open("sample_weights_WU1.0-2.1_lognorm.json") as f:
    SAMPLE_WEIGHTS = json.load(f)

def calc_weight(k):
    for w, kk in SAMPLE_WEIGHTS.items():
        k1, k2 = kk        
        if k1<= float(k) <= k2:
            break
    return float(w)/SW_SCALE

def norm_k(k):
    # k = float(np.log(2-k))
    k = float(((k-K_MIN)/(K_MAX-K_MIN)-0.5)*2)
    return k

def denorm_k(k):
    k = float((k/2+0.5)*(K_MAX-K_MIN)+K_MIN)
    # k = float(2-np.exp(k))
    return k 

def norm_graph(graph:MatSurfGraph) -> MatSurfGraph:
    for key, mat in graph.materials.items():
        nucs, vol, dens = mat[:-2], mat[-2], mat[-1]
        vol /= graph.total_volume
        
        exps = np.exp(nucs)
        nucs = exps/exps.sum()

        graph.materials[key] = nucs.tolist() + [vol, dens]

    return graph

def norm_power(P:float) -> float:
    return P/10_000

def norm_x(graph: MatSurfGraph, power: float) -> (MatSurfGraph, float): 
    return norm_graph(graph=graph), norm_power(P=power)

def read_WU(filepath: str):
    graph = load_graph(directory=filepath)
    return graph.WURelation

def run_tests(model: MatSurfGcn) :
    test_res = {
        'true':[],
        'pred':[],
        # 'cls':[]
    }
    test_loss = []
    for x, y in zip(x_test, y_test):
        outputs = gcn(graph=x[0], power=x[1])
        k = outputs.cpu() 
        test_res['true'].append(denorm_k(y))
        test_res['pred'].append(denorm_k(k))
        # test_res['cls'].append(_cls.tolist())
        test_loss.append( abs(denorm_k(y) - denorm_k(k)) / abs(denorm_k(k)))
    test_loss = sum(test_loss) / len(test_loss)
    with open(os.path.join(savedir, 'test.json'), 'w') as f:
        json.dump(test_res, f)
    return test_loss, test_res


if __name__=='__main__':
    with open('graph_nucliedes.txt', 'r') as f:
        isotopes = f.read().split()

    data_dirs = list(filter(lambda x: os.path.isdir(os.path.join(BASE_DIR, x)),  os.listdir(BASE_DIR)[:-1]))
    data_dirs = list(filter(lambda x: WU_min < read_WU(os.path.join(BASE_DIR, x)) <= WU_max, data_dirs))

    train_dirs, not_train_dirs = train_test_split(data_dirs, test_size=0.2, random_state=42)
    test_dirs, val_dirs = train_test_split(not_train_dirs, test_size=0.5, random_state=42)

    x_train = list(norm_x(*load_x(os.path.join(BASE_DIR, x))) for x in tqdm(train_dirs, desc='Loading x_train'))
    k_train = list(load_res(os.path.join(BASE_DIR, x, 'data.csv')) for x in tqdm(train_dirs, desc='Loading k_train'))
    y_train = list(map(norm_k, k_train))
    sw_train = list(map(calc_weight, y_train))

    x_val = list(norm_x(*load_x(os.path.join(BASE_DIR, x))) for x in tqdm(val_dirs, desc='Loading x_val'))
    k_val = list(load_res(os.path.join(BASE_DIR, x, 'data.csv')) for x in tqdm(val_dirs, desc='Loading k_val'))
    y_val = list(map(norm_k, k_val))
    sw_val = list(map(calc_weight, y_val))

    x_test = list(norm_x(*load_x(os.path.join(BASE_DIR, x))) for x in tqdm(test_dirs, desc='Loading x_test'))
    k_test = list(load_res(os.path.join(BASE_DIR, x, 'data.csv')) for x in tqdm(test_dirs, desc='Loading k_test'))
    y_test = list(map(norm_k, k_test))
    sw_test = list(map(calc_weight, y_test))

    # inter_dims =  list(range(16, 32, 4)) + list(range(32, 65, 8)) 
    # inter_dims2 = list(range(8,16,2)) + list(range(16, 32, 4))     
    inter_dims = list(range(48, 65, 8)) 
    inter_dims2 = list(range(24, 32, 4)) 
    print(inter_dims, len(inter_dims), inter_dims2, len(inter_dims2), sep='\n')
    pg = ParamGrid({
        'inter_dim1':inter_dims,
        'inter_dim2':inter_dims2,
        'K':[1,4,8]
    })
    
    
    for var in pg.grid:
        savedir = os.path.join(SAVE_DIR, f'first_SuperGATConv_2l_k{var["K"]}_id{var["inter_dim1"]}_{var["inter_dim2"]}')
        gcn = MatSurfGAT(
            isotopes=isotopes,
            out_dim=1,
            **var
        )
        hist = gcn.fit(
            savedir=savedir,
            x_train=x_train,
            y_train=y_train,
            # sw_train=sw_train,
            x_val=x_val,
            y_val=y_val,
            # sw_val=sw_val,
            epochs=200,
            lr=0.05,
            # weight_decay=0           
        )
        with open(os.path.join(savedir, 'train.json'), 'w') as f:
            json.dump(hist, f)



        with torch.no_grad():
            gcn.load_state_dict(torch.load(os.path.join(savedir, 'best.pt'), weights_only=False))
            test_loss, test_res = run_tests(gcn)
            
        print(f'Test_loss: {test_loss*100:.4f}%')  
        print(savedir)              
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,10))
        ax[0].plot(hist['train_loss'], label='train')
        ax[0].plot(hist['val_loss'], label='validation')
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].set_title(f"Loss Curve\nTest_loss:{test_loss}")
        ax[0].grid()
        ax[0].legend()

        ax[1].scatter(y=test_res['true'], x=range(len(test_res['true'])), label='true')
        ax[1].scatter(y=test_res['pred'], x=range(len(test_res['pred'])), label='pred')
        ax[1].set_xticks(range(len(test_res['true'])), labels=range(len(test_res['true'])))
        ax[1].set_xlabel("i")
        ax[1].set_ylabel("$K_{\inf}$")
        ax[1].grid()
        ax[1].legend()
        plt.savefig(os.path.join(savedir, "results.jpeg"))
        plt.close()

