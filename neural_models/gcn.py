import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, TAGConv, GATv2Conv, SuperGATConv
from datastructures.graph import MatSurfGraph
from tqdm import trange
import os
import numpy as np

class GCNExample(torch.nn.Module):
    def __init__(self, in_features:int, inter_dim:int=16, out_features:int=1):
        super().__init__()
        self.conv1 = GCNConv(in_features, inter_dim)
        self.conv2 = GCNConv(inter_dim, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
class MatSurfGcn(torch.nn.Module):
    
    def __init__(self, isotopes:list[str], inter_dim1:int, inter_dim2:int, out_dim:int = 1, device = None, K=3):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.isotopes = isotopes

        self.material_encoder = torch.nn.Linear(len(isotopes)+2, inter_dim1)
        self.cylinder_encoder = torch.nn.Linear(3, inter_dim1)
        self.plane_encoder = torch.nn.Linear(4, inter_dim1)
        self.power_encoder = torch.nn.Linear(1, inter_dim2)

        self.graph_conv = torch.nn.ModuleList([
            GCNConv(inter_dim1, inter_dim1),
            GCNConv(inter_dim1, inter_dim2),
            GCNConv(inter_dim2, inter_dim2),
            ])

        self.reg_head_hex = torch.nn.Sequential(
            torch.nn.RMSNorm((14*inter_dim2)),
            torch.nn.Linear(14*inter_dim2, inter_dim2),            
            torch.nn.RMSNorm((inter_dim2)),
            torch.nn.CELU(),            
            torch.nn.Linear(inter_dim2, out_dim),
            )
        self.reg_head_sqr = torch.nn.Sequential(
            torch.nn.RMSNorm((12*inter_dim2)),
            torch.nn.Linear(12*inter_dim2, inter_dim2),
            torch.nn.RMSNorm((inter_dim2)),
            torch.nn.CELU(),            
            torch.nn.Linear(inter_dim2, out_dim),
            )
        self.to(device)
    
    def forward(self, graph:MatSurfGraph, power:float):     
        mats = torch.as_tensor(list(graph.materials.values())).to(self.device)
        cyls = torch.as_tensor(list(graph.cylinders.values())).to(self.device)
        planes = torch.as_tensor(list(graph.planes.values())).to(self.device)
        power = torch.as_tensor([power]).to(self.device)[None,:]
        edges = torch.as_tensor(graph.edges, dtype=torch.long).t().contiguous().to(self.device)

        if graph.lattype =='hex':
            reg_head = self.reg_head_hex
        elif graph.lattype =='sqr':
            reg_head = self.reg_head_sqr

        mats = self.material_encoder(mats).relu()
        cyls = self.cylinder_encoder(cyls).relu()
        planes = self.plane_encoder(planes).relu()

        power = self.power_encoder(power)

        x = torch.concat((mats, cyls, planes)) 

        for gc in self.graph_conv:
            x = gc(x, edges)
            x = torch.nn.CELU()(x)
            x=F.dropout(x, training=self.training)

        x = torch.concat((x, power))

        x = torch.flatten(x).T
        # print(x.shape)

        x = reg_head(x)
        # x = torch.nn.Tanh()(x)
        x = torch.nn.Softsign()(x)
        return x
    
    def calculate_loss(self, true, pred, sample_weights):
        loss_fn = torch.nn.MSELoss(reduction='none')
        loss = loss_fn(true, pred)*sample_weights
        return loss.mean()
    
    def fit(
        self, 
        savedir:str,
        x_train:list[MatSurfGraph, float], 
        y_train:list[float],         
        x_val:list[MatSurfGraph, float], 
        y_val:list[float],
        sw_train:list[float]=None,
        sw_val:list[float]=None,
        epochs = 100,
        lr = 0.01,
        weight_decay=5e-4
        )->dict:
        os.makedirs(savedir, exist_ok=True)
        loss_fn = self.calculate_loss
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, cooldown=3, patience=5)
        best_loss = None

        y_train = torch.as_tensor(y_train, dtype=torch.float32).to(self.device)[:,None]
        y_val = torch.as_tensor(y_val, dtype=torch.float32).to(self.device)[:,None]
        if sw_train is not None:
            sw_train = torch.as_tensor(sw_train, dtype=torch.float32).to(self.device)
        else:
            sw_train = torch.ones_like(y_train, dtype=torch.float32).to(self.device)
        if sw_val is not None:
            sw_val = torch.as_tensor(sw_val, dtype=torch.float32).to(self.device)
        else:
            sw_val = torch.ones_like(y_val, dtype=torch.float32).to(self.device)

        history = {
            'train_loss':[],
            'val_loss':[]
        }

        pbar = trange(epochs, desc="Training started...", leave=False)
        for epoch in pbar:
            epoch_loss = []

            pbar.set_description(f"Epoch {epoch}/{epochs}:")
            pbar.refresh()
            optimizer.zero_grad()

            for x, y, w in zip(x_train, y_train, sw_train):
                outputs = self(graph=x[0], power=x[1])
                # Compute the loss and its gradients
                loss = loss_fn(y, outputs, w)
                loss.backward()
                # Adjust learning weights
                optimizer.step()
                epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            history['train_loss'].append(float(epoch_loss))

            with torch.no_grad():
                epoch_val_loss = []
                for x, y, w in zip(x_val, y_val, sw_val):
                    outputs = self(graph=x[0], power=x[1])
                    # Compute the loss and its gradients
                    
                    val_loss = loss_fn(y, outputs, w)
                    # Adjust learning weights
                    epoch_val_loss.append(val_loss.item())
                epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)

                if not best_loss or epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    torch.save(self.state_dict(), os.path.join(savedir, "best.pt"))

            pbar.set_postfix_str(
                f"Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
            )

            scheduler.step(epoch_val_loss)
            history['val_loss'].append(float(epoch_val_loss))
            torch.save(self.state_dict(), os.path.join(savedir, "last.pt"))

        print(f"Training finished. Best Loss: {best_loss}")
        return history

class GorynychConv(torch.nn.Module):
    Q1 = 0.16060019211900226 
    Q2 = 0.6698719147294607

    @classmethod
    def get_mode(cls, value:float):
        if value < cls.Q1:
            mode = 'under'
        elif value <= cls.Q2:
            mode = 'equal'
        else:
            mode = 'over'
        return mode


    def __init__(self, isotopes:list[str], inter_dim1:int, inter_dim2:int, out_dim:int = 1, device = None):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.isotopes = isotopes

        self.material_encoder = torch.nn.Linear(len(isotopes)+2, inter_dim1)
        self.cylinder_encoder = torch.nn.Linear(3, inter_dim1)
        self.plane_encoder = torch.nn.Linear(4, inter_dim1)
        self.power_encoder = torch.nn.Linear(1, inter_dim2)

        self.graph_conv = torch.nn.ModuleList([ 
            GCNConv(inter_dim1, inter_dim1),
            # GCNConv(inter_dim1, inter_dim1),
            # GCNConv(inter_dim1, inter_dim1),
            GCNConv(inter_dim1, inter_dim1),
            GCNConv(inter_dim1, inter_dim2),
            ])

        # self.cls_head_hex = torch.nn.Sequential(
        #     torch.nn.Linear(14*inter_dim2, 3),
        #     #torch.nn.SELU(),
        #     #torch.nn.Linear(inter_dim2, 3),
        #     torch.nn.Softmax(dim=0)
        #     )
        # self.cls_head_sqr = torch.nn.Sequential(
        #     torch.nn.Linear(12*inter_dim2, 3),
        #     #torch.nn.SELU(),
        #     #torch.nn.Linear(inter_dim2, 3),
        #     torch.nn.Softmax(dim=0)
        #     )
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(inter_dim2, 3),
            #torch.nn.SELU(),
            #torch.nn.Linear(inter_dim2, 3),
            torch.nn.Softmax(dim=0)
            )

        self.cls_encoder = torch.nn.Linear(3, inter_dim2)

        # self.reg_neck_sqr = torch.nn.Sequential(
        #     torch.nn.Linear(13*inter_dim2, inter_dim2),
        #     torch.nn.LayerNorm(inter_dim2),
        #     torch.nn.SELU()
        #     )
        # self.reg_neck_hex = torch.nn.Sequential(
        #     torch.nn.Linear(15*inter_dim2, inter_dim2),
        #     torch.nn.LayerNorm(inter_dim2),
        #     torch.nn.SELU()
        #     )
        self.reg_neck = torch.nn.Sequential(
            torch.nn.Linear(inter_dim2+3, inter_dim1),
            torch.nn.LayerNorm(inter_dim1),
            torch.nn.SELU()
            )

        self.reg_head_over = torch.nn.Sequential(
            torch.nn.Linear(inter_dim1, out_dim),
            # torch.nn.Tanh()
            # torch.nn.SELU()
            )
        self.reg_head_under = torch.nn.Sequential(
            torch.nn.Linear(inter_dim1, out_dim),
            # torch.nn.Tanh()
            # torch.nn.SELU()
            )
        self.reg_head_equal = torch.nn.Sequential(
            torch.nn.Linear(inter_dim1, out_dim),
            # torch.nn.Tanh()
            # torch.nn.SELU()
            )
        self.to(device)
    
    def forward(self, graph:MatSurfGraph, power:float, mode=None):     
        mats = torch.as_tensor(list(graph.materials.values())).to(self.device)
        cyls = torch.as_tensor(list(graph.cylinders.values())).to(self.device)
        planes = torch.as_tensor(list(graph.planes.values())).to(self.device)
        power = torch.as_tensor([power]).to(self.device)[None,:]
        edges = torch.as_tensor(graph.edges, dtype=torch.long).t().contiguous().to(self.device)

        # print(graph.lattype)

        # if graph.lattype =='hex':
        #     # cls_head = self.cls_head_hex
        #     reg_neck = self.reg_neck_hex
        # elif graph.lattype =='sqr':
        #     # cls_head = self.cls_head_sqr
        #     reg_neck = self.reg_neck_sqr

        mats = self.material_encoder(mats).relu()
        cyls = self.cylinder_encoder(cyls).relu()
        planes = self.plane_encoder(planes).relu()

        power = self.power_encoder(power).relu()

        x = torch.concat((mats, cyls, planes)) 

        for l in self.graph_conv:
            x = l(x, edges)
            x = torch.nn.SELU()(x)
            x = F.dropout(x, training=self.training)
        # x = self.graph_conv2(x, edges)
        # x = torch.nn.SELU()(x)
        # x = F.dropout(x, training=self.training)

        x = torch.concat((x, power))
        x = x.mean(dim=0)
        # x = torch.flatten(x)

        # print(x.shape)

        x_cls = self.cls_head(x)
        # print(x_cls.shape, x_cls)

        #x_cls_reg = self.cls_encoder(x_cls).relu()
        # print(x_cls_reg.shape)

        x = torch.concat((x, x_cls))
        
        # print(x.shape)
        # x = torch.flatten(x)

        x = self.reg_neck(x)
        # print(x.shape)

        if mode is None:
            over = self.reg_head_over(x)
            under = self.reg_head_under(x)
            equal = self.reg_head_equal(x)
            x = torch.concat((over, equal, under)) * x_cls
            x = x.sum(dim=0)
        elif mode == 'over':
            x = self.reg_head_over(x)
        elif mode == 'under':
            x = self.reg_head_under(x)
        elif mode == 'equal':
            x = self.reg_head_equal(x)           

        return x, x_cls
    
    def calculate_loss(self, true, pred):
        
        reg_loss_fn = torch.nn.MSELoss()
        cls_loss_fn = torch.nn.CrossEntropyLoss()

        out, _cls = pred
        # print(pred)

        target_cls = [int(true<self.Q1), int(self.Q1<=true<self.Q2), int(true>=self.Q2)]
        target_cls = torch.tensor(target_cls, dtype=_cls.dtype, device=self.device)
        #print(true, target_cls)

        reg_loss = reg_loss_fn(true, out) 
        cls_loss = cls_loss_fn(_cls, target_cls)
        # print(reg_loss.item(), cls_loss.item())

        loss = reg_loss + cls_loss
        #print(loss)
        return loss

    def fit(
        self, 
        savedir:str,
        x_train:list[MatSurfGraph, float], 
        y_train:list[float], 
        x_val:list[MatSurfGraph, float], 
        y_val:list[float],
        epochs = 100,
        lr = 0.01,
        weight_decay=5e-4
        )->dict:
        os.makedirs(savedir, exist_ok=True)
        loss_fn = self.calculate_loss
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, cooldown=3, patience=50)
        best_loss = None

        y_train = torch.as_tensor(y_train, dtype=torch.float32).to(self.device)[:,None]
        y_val = torch.as_tensor(y_val, dtype=torch.float32).to(self.device)[:,None]

        history = {
            'train_loss':[],
            'val_loss':[]
        }

        pbar = trange(epochs, desc="Training started...", leave=False)
        for epoch in pbar:
            epoch_loss = []

            pbar.set_description(f"Epoch {epoch}/{epochs}:")
            pbar.refresh()
            optimizer.zero_grad()

            for x, y in zip(x_train, y_train):

                mode =self.get_mode(float(y)) if epoch <= epochs/10*9 else None
                
                outputs = self(graph=x[0], power=x[1], mode=mode)
                # Compute the loss and its gradients
                loss = loss_fn(y, outputs)
                loss.backward()
                # Adjust learning weights
                optimizer.step()
                epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            history['train_loss'].append(float(epoch_loss))

            with torch.no_grad():
                epoch_val_loss = []
                epoch_accuracy = []
                epoch_agreement = []

                for x, y in zip(x_val, y_val):

                    mode =self.get_mode(float(y))

                    outputs = self(graph=x[0], power=x[1], mode=None)
                    # Compute the loss and its gradients
                    
                    val_loss = loss_fn(y, outputs)

                    out, _cls = outputs

                    target_cls = np.array([int(y<self.Q1), int(self.Q1<=y<self.Q2), int(y>=self.Q2)])
                    agreement_cls = np.array([int(out<self.Q1), int(self.Q1<=out<self.Q2), int(out>=self.Q2)])

                    accuracy = np.argmax(_cls.cpu().numpy()) == np.argmax(target_cls)
                    agreement = np.argmax(_cls.cpu().numpy()) == np.argmax(agreement_cls)
                    # Adjust learning weights
                    epoch_val_loss.append(val_loss.item())
                    epoch_accuracy.append(int(accuracy))
                    epoch_agreement.append(int(agreement))

                epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
                epoch_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
                epoch_agreement = sum(epoch_agreement) / len(epoch_agreement)

                if not best_loss or epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    torch.save(self.state_dict(), os.path.join(savedir, "best.pt"))

            pbar.set_postfix_str(
                f"Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_accuracy*100:.1f}%, Agreement:{epoch_agreement*100:.1f}%"
            )

            scheduler.step(epoch_val_loss)
            history['val_loss'].append(float(epoch_val_loss))
            torch.save(self.state_dict(), os.path.join(savedir, "last.pt"))

        print(f"Training finished. Best Loss: {best_loss}")
        return history
    

class MatSurfGAT(MatSurfGcn):

    def __init__(self, isotopes:list[str], inter_dim1:int, inter_dim2:int, out_dim:int = 1, device = None, K=3):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        super().__init__(isotopes, inter_dim1, inter_dim2, out_dim, device, K)

        self.power_encoder = torch.nn.Linear(1, inter_dim2*K)

        self.graph_conv = torch.nn.ModuleList([
            #TAGConv(inter_dim1, inter_dim1, K),
            SuperGATConv(inter_dim1, inter_dim2, K),
            SuperGATConv(inter_dim2*K, inter_dim2, K),
            ])
        
        self.reg_head_hex = torch.nn.Sequential(
            torch.nn.RMSNorm((14*inter_dim2*K)),
            torch.nn.Linear(14*inter_dim2*K, inter_dim2*K),            
            torch.nn.RMSNorm((inter_dim2*K)),
            torch.nn.CELU(),            
            torch.nn.Linear(inter_dim2*K, out_dim),
            )
        self.reg_head_sqr = torch.nn.Sequential(
            torch.nn.RMSNorm((12*inter_dim2*K)),
            torch.nn.Linear(12*inter_dim2*K, inter_dim2*K),
            torch.nn.RMSNorm((inter_dim2*K)),
            torch.nn.CELU(),            
            torch.nn.Linear(inter_dim2*K, out_dim),
            )

        self.to(device)

    def forward(self, graph:MatSurfGraph, power:float):     
        mats = torch.as_tensor(list(graph.materials.values())).to(self.device)
        cyls = torch.as_tensor(list(graph.cylinders.values())).to(self.device)
        planes = torch.as_tensor(list(graph.planes.values())).to(self.device)
        power = torch.as_tensor([power]).to(self.device)[None,:]
        edges = torch.as_tensor(graph.edges, dtype=torch.long).t().contiguous().to(self.device)

        if graph.lattype =='hex':
            reg_head = self.reg_head_hex
        elif graph.lattype =='sqr':
            reg_head = self.reg_head_sqr

        mats = self.material_encoder(mats).relu()
        cyls = self.cylinder_encoder(cyls).relu()
        planes = self.plane_encoder(planes).relu()

        power = self.power_encoder(power)

        # print(mats.shape, cyls.shape, planes.shape)

        x = torch.concat((mats, cyls, planes)) 

        # print(x.shape)

        for gc in self.graph_conv:
            x = gc(x, edges)
            x = torch.nn.CELU()(x)
            x=F.dropout(x, training=self.training)
            # print(x.shape)

        # print(x.shape)
        x = torch.concat((x, power))

        x = torch.flatten(x).T
        # print(x.shape)

        x = reg_head(x)
        # x = torch.nn.Tanh()(x)
        x = torch.nn.Softsign()(x)
        return x