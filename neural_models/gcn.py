import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, TAGConv
from datastructures.graph import MatSurfGraph
from tqdm import trange
import os

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
        self.power_encoder = torch.nn.Linear(1, inter_dim1)

        self.graph_conv1 = GCNConv(inter_dim1, inter_dim2)
        self.graph_conv2 = GCNConv(inter_dim2, out_dim)

        self.reg_head_hex = torch.nn.Linear(14, 1)
        self.reg_head_sqr = torch.nn.Linear(12, 1)

        self.to(device)
    
    def forward(self, graph:MatSurfGraph, power:float):     
        mats = torch.as_tensor(list(graph.materials.values())).to(self.device)
        cyls = torch.as_tensor(list(graph.cylinders.values())).to(self.device)
        planes = torch.as_tensor(list(graph.planes.values())).to(self.device)
        power = torch.as_tensor([power/10_000]).to(self.device)[None,:]
        edges = torch.as_tensor(graph.edges, dtype=torch.long).t().contiguous().to(self.device)

        if graph.lattype =='hex':
            reg_head = self.reg_head_hex
        elif graph.lattype =='sqr':
            reg_head = self.reg_head_sqr

        mats = self.material_encoder(mats).relu()
        cyls = self.cylinder_encoder(cyls).relu()
        planes = self.plane_encoder(planes).relu()
        power = self.power_encoder(power).relu()

        x = torch.concat((mats, cyls, planes, power))   

        x = self.graph_conv1(x, edges)
        # x = torch.nn.SELU()(x)
        x = F.dropout(x, training=self.training)
        x = self.graph_conv2(x, edges)
        x = reg_head(x.T)
        # x = torch.nn.SELU()(x)
        return x.mean(dim=0)
    
    def calculate_loss(self, true, pred):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(true, pred)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, cooldown=3, patience=5)
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
                outputs = self(graph=x[0], power=x[1])
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
                for x, y in zip(x_val, y_val):
                    outputs = self(graph=x[0], power=x[1])
                    # Compute the loss and its gradients
                    
                    val_loss = loss_fn(y, outputs)
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
