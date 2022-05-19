import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_add_pool, ChebConv, global_max_pool
from torch.nn import BatchNorm1d
from math import floor


class EEGGraphConvNet(nn.Module):
    
    def __init__(self, reduced_sensors=True, sfreq=None, batch_size=256):
        super(EEGGraphConvNet, self).__init__()
        # Define and initialize hyperparameters
        self.sfreq = sfreq
        self.batch_size = batch_size
        self.input_size = 8 if reduced_sensors else 62
        
        # Layers definition
        # Graph convolutional layers
        self.conv1 = GCNConv(6, 16, cached=True, normalize=False)
        self.conv2 = GCNConv(16, 32, cached=True, normalize=False)
        self.conv3 = GCNConv(32, 64, cached=True, normalize=False)
        self.conv4 = GCNConv(64, 50, cached=True, normalize=False)
        
        # Batch normalization
        self.batch_norm = BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(50, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 2)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        # Perform all graph convolutions
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weigth), negative_slope=0.01)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.leaky_relu(self.conv2(x, edge_index, edge_weigth), negative_slope=0.01)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.leaky_relu(self.conv3(x, edge_index, edge_weigth), negative_slope=0.01)
        x = F.dropout(x, p=0.2, training=self.training)
        
        conv_out = self.conv4(x, edge_index, edge_weigth)
        # Perform batch normalization
        batch_norm_out = F.leaky_relu(self.batch_norm(conv_out), negative_slope=0.01)
        x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        mean_pool = global_add_pool(batch_norm_out, batch=batch)
        
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(mean_pool), negative_slope=0.01)
        out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc3(out))
        return out
    
    
    

class GCNsNet(nn.Module):
    def __init__(self, input_size, filter_size, num_classes=3) -> None:
        super().__init__()
        
        self.conv1 = ChebConv(input_size, 16, filter_size, normalization="rw")
        self.conv2 = ChebConv(16, 32, filter_size, normalization="rw")
        self.conv3 = ChebConv(32, 64, filter_size, normalization="rw")
        self.conv4 = ChebConv(64, 128, filter_size, normalization="rw")
        self.conv5 = ChebConv(128, 256, filter_size, normalization="rw")
        self.conv6 = ChebConv(256, 512, filter_size, normalization="rw")
        
        self.fc1 = torch.nn.Linear(512, num_classes)
        
    
    def forward(self, x, edge_index, edge_weigth, batch):
        x = self.conv1(x, edge_index, edge_weigth)
        x = global_max_pool(x, batch)
        x = F.softplus(x)
        
        x = self.conv2(x, edge_index, edge_weigth)
        x = global_max_pool(x, batch)
        x = F.softplus(x)
        
        x = self.conv3(x, edge_index, edge_weigth)
        x = global_max_pool(x, batch)
        x = F.softplus(x)
        
        x = self.conv3(x, edge_index, edge_weigth)
        x = global_max_pool(x, batch)
        x = F.softplus(x)
        
        x = self.conv4(x, edge_index, edge_weigth)
        x = global_max_pool(x, batch)
        x = F.softplus(x)
        
        x = self.conv5(x, edge_index, edge_weigth)
        x = global_max_pool(x, batch)
        x = F.softplus(x)
        
        x = self.conv6(x, edge_index, edge_weigth)
        x = global_max_pool(x, batch)
        x = F.softplus(x)
        
        out = self.fc1(x)
        return out
        
        
        
        
        
    