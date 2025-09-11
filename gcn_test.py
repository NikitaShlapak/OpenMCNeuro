from datastructures.graph import load_pair, load_res,load_x,  BASE_DIR
from neural_models.gcn import MatSurfGcn
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import json
from utils.experiments import ParamGrid

SAVE_DIR = "/home/nikita/PycharmProjects/OpenMC/neuro/models/v6/gcn3"


def pair_loader(dirs:list[str]):
    for _dir in dirs:
        yield load_pair(folder=_dir)

if __name__=='__main__':
    with open('graph_nucliedes.txt', 'r') as f:
        isotopes = f.read().split()

    data_dirs = list(filter(lambda x: os.path.isdir(os.path.join(BASE_DIR, x)),  os.listdir(BASE_DIR)[:-1]))
    train_dirs, not_train_dirs = train_test_split(data_dirs, test_size=0.4, random_state=42)
    test_dirs, val_dirs = train_test_split(not_train_dirs, test_size=0.5, random_state=42)

    x_train = list(load_x(os.path.join(BASE_DIR, x)) for x in tqdm(train_dirs, desc='Loading x_train'))
    y_train = list(load_res(os.path.join(BASE_DIR, x, 'data.csv')) for x in tqdm(train_dirs, desc='Loading y_train'))
    x_val = list(load_x(os.path.join(BASE_DIR, x)) for x in tqdm(val_dirs, desc='Loading x_val'))
    y_val = list(load_res(os.path.join(BASE_DIR, x, 'data.csv')) for x in tqdm(val_dirs, desc='Loading y_val'))
    x_test = list(load_x(os.path.join(BASE_DIR, x)) for x in tqdm(test_dirs, desc='Loading x_test'))
    y_test = list(load_res(os.path.join(BASE_DIR, x, 'data.csv')) for x in tqdm(test_dirs, desc='Loading y_test'))

    inter_dims = list(range(2,16,2)) + list(range(16, 32, 4)) + list(range(32, 65, 8)) 
    print(inter_dims, len(inter_dims))
    pg = ParamGrid({
        'inter_dim1':inter_dims,
        'inter_dim2':inter_dims,
        # 'K':[1,3,5]
    })
    
    
    for var in pg.grid:
        savedir = os.path.join(SAVE_DIR, f'power_gcn_id{var["inter_dim1"]}_{var["inter_dim2"]}')
        gcn = MatSurfGcn(
            isotopes=isotopes,
            out_dim=1,
            **var
        )
        hist = gcn.fit(
            savedir=savedir,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=200,
            lr=0.01,
            weight_decay=5e-4            
        )
        with open(os.path.join(savedir, 'train.json'), 'w') as f:
            json.dump(hist, f)

        with torch.no_grad():
            gcn.load_state_dict(torch.load(os.path.join(savedir, 'best.pt'), weights_only=False))
            test_res = {
                'true':[],
                'pred':[]
            }
            test_loss = []
            for x, y in zip(x_test, y_test):
                outputs = gcn(graph=x[0], power=x[1])
                test_res['true'].append(float(y)+1)
                test_res['pred'].append(float(outputs)+1)
                test_loss.append( float(y-outputs)**2)
            test_loss = sum(test_loss) / len(test_loss)
            with open(os.path.join(savedir, 'test.json'), 'w') as f:
                json.dump(test_res, f)
        print('Test_loss: ', test_loss)  
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

