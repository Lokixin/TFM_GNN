import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_add_pool, ChebConv, global_max_pool, SAGPooling
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
        self.conv1 = GCNConv(-1, 16, cached=True, normalize=False)
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
        #x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.leaky_relu(self.conv2(x, edge_index, edge_weigth), negative_slope=0.01)
        #x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.leaky_relu(self.conv3(x, edge_index, edge_weigth), negative_slope=0.01)
        #x = F.dropout(x, p=0.2, training=self.training)
        
        conv_out = self.conv4(x, edge_index, edge_weigth)
        # Perform batch normalization
        batch_norm_out = F.leaky_relu(self.batch_norm(conv_out), negative_slope=0.01)
        #x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        mean_pool = global_add_pool(batch_norm_out, batch=batch)
        
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(mean_pool), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
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
        
        


class EEGGraphConvNetLSTM(nn.Module):
    
    def __init__(self, reduced_sensors=True, sfreq=None, batch_size=256, **kwargs):
        super(EEGGraphConvNetLSTM, self).__init__()
        # Define and initialize hyperparameters
        self.sfreq = sfreq
        self.batch_size = batch_size
        #self.input_size = 8 if reduced_sensors else 62
        
        # Layers definition
        input_size = kwargs.pop("input_size", 1280)
        hidden_dim = 320
        
        self.lstm = nn.LSTM(input_size, hidden_dim, 2)
        
        # Graph convolutional layers
        self.conv1 = GCNConv(hidden_dim, 320, cached=True, normalize=False)
        self.conv2 = GCNConv(320, 180, cached=True, normalize=False)
        self.conv3 = GCNConv(180, 90, cached=True, normalize=False)
        self.conv4 = GCNConv(90, 50, cached=True, normalize=False)
        
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
        x, _ = self.lstm(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        # Perform all graph convolutions
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weigth), negative_slope=0.01)
        #x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.leaky_relu(self.conv2(x, edge_index, edge_weigth), negative_slope=0.01)
        #x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.leaky_relu(self.conv3(x, edge_index, edge_weigth), negative_slope=0.01)
        #x = F.dropout(x, p=0.2, training=self.training)
        
        conv_out = self.conv4(x, edge_index, edge_weigth)
        # Perform batch normalization
        batch_norm_out = F.leaky_relu(self.batch_norm(conv_out), negative_slope=0.01)
        #x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        mean_pool = global_add_pool(batch_norm_out, batch=batch)
        
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(mean_pool), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc3(out))
        return out
        
        
    
class EEGSmall(nn.Module):

    def __init__(self, reduced_sensors=True, sfreq=None, batch_size=256, **kwargs):
        super(EEGSmall, self).__init__()
       
        #self.batch_norm = BatchNorm1d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1280, 640)
        self.fc2 = nn.Linear(640, 320)
        self.fc3 = nn.Linear(320, 2)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
       
        # Perform batch normalization
        #x = F.leaky_relu(self.batch_norm(x), negative_slope=0.01)
        #x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        #mean_pool = global_add_pool(x, batch=batch)
        print(x.shape)
        print(batch.shape)
        x = x.view(batch.shape[0], -1)
        print(x.shape)
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = self.fc3(out)
        return out
    
    

class EEGConvNetMini(nn.Module):
    """Same as EEGGraphConvNet but with fewer 
    convolutional layers
    """
    def __init__(self, reduced_sensors=True, sfreq=None, batch_size=256):
        super(EEGConvNetMini, self).__init__()
        # Define and initialize hyperparameters
        self.sfreq = sfreq
        self.batch_size = batch_size
        self.input_size = 8 if reduced_sensors else 62
        
        # Layers definition
        # Graph convolutional layers
        self.conv1 = GCNConv(-1, 16, cached=True, normalize=False)
        
        # Batch normalization
        self.batch_norm = BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        # Perform all graph convolutions
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weigth), negative_slope=0.01)
        #x = F.dropout(x, p=0.2, training=self.training)
        
        # Perform batch normalization
        x = F.leaky_relu(self.batch_norm(x), negative_slope=0.01)
        #x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        mean_pool = global_add_pool(x, batch=batch)
        
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(mean_pool), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc3(out))
        return out
    
    
class EEGConvNetMiniV2(nn.Module):
    """Same as EEGGraphConvNet but with fewer 
    convolutional layers
    """
    def __init__(self, **kwargs):
        super(EEGConvNetMiniV2, self).__init__()
        # Layers definition
        # Graph convolutional layers
        self.conv1 = GCNConv(-1, 32, cached=True, normalize=False) #16
        self.conv2 = GCNConv(32, 64, cached=True, normalize=False) #32
        
        # Batch normalization
        self.batch_norm1 = BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        # Perform all graph convolutions
        x = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index, edge_weigth)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x, edge_index, edge_weigth)), negative_slope=0.01)

        #x = F.dropout(x, p=0.2, training=self.training)
        #x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        mean_pool = global_add_pool(x, batch=batch)
        
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(mean_pool), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc3(out))
        return out
    

from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class EEGConvNetMiniV3(nn.Module):
    """Same as EEGGraphConvNet but with fewer 
    convolutional layers
    """
    def __init__(self, **kwargs):
        super(EEGConvNetMiniV3, self).__init__()
        # Layers definition
        # Graph convolutional layers
        self.conv1 = GCNConv(-1, 16, cached=True, normalize=False)
        self.conv2 = GCNConv(16, 32, cached=True, normalize=False)
        
        # Batch normalization
        self.batch_norm1 = BatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.pool1 = SAGPooling(16, ratio=0.5)
        self.pool2 = SAGPooling(32, ratio=0.5)
        
        
        # Fully connected layers
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        # Perform all graph convolutions
        print("Original Size ")
        print(x.size())
        x = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index)), negative_slope=0.01)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        print("Size after conv1")
        print(x.size())
        x, edge_index_downsampled, _, batch, perm, _  = self.pool1(x, edge_index)
        print("Size after pool")
        print(x.size())
        
        x = F.leaky_relu(self.batch_norm2(self.conv2(x, edge_index_downsampled)), negative_slope=0.01)
        print("Size after conv2")
        print(x.size())
        x, edge_index_downsampled, _, batch, perm, _  = self.pool2(x, edge_index_downsampled)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        #x = x1 + x2
        print("Out of the convolutions")
        #x_downsampled, edge_index_downsampled, _, _, perm, _ 

        #x = F.dropout(x, p=0.2, training=self.training)
        #x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        mean_pool = global_add_pool(x, batch=batch)
        print("Out of the global add pool")
        
        
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(mean_pool), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc3(out))
        return out
    



class EEGConvNetMiniLSTM(nn.Module):
    """Same as EEGGraphConvNet but with fewer 
    convolutional layers
    """
    def __init__(self, **kwargs):
        super(EEGConvNetMiniLSTM, self).__init__()
        
        # Layers definition
        # Graph convolutional layers
        self.lstm = nn.LSTM(1280, 512)
        self.lstm2 = nn.LSTM(512, 256)
        self.fc1 = nn.Linear(256, 128)
        
        self.conv1 = GCNConv(128, 64, cached=True, normalize=False)
        self.conv2 = GCNConv(64, 32, cached=True, normalize=False)
        
        # Batch normalization
        self.batch_norm1 = BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        
        # Fully connected layers
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc4.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        # Perform all graph convolutions
        x = F.leaky_relu(self.lstm(x)[0])
        #print("LSTM-1 output: ")
        #print(x.size())
        x = F.leaky_relu(self.lstm2(x)[0])
        #print("LSTM-2 output: ")
        #print(x.size())
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        #print("LSTM-2 output: ")
        #print(x.size())
        x = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index, edge_weigth)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x, edge_index, edge_weigth)), negative_slope=0.01)
        
        #x = F.dropout(x, p=0.2, training=self.training)
        
        # Perform batch normalization
        #x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        mean_pool = global_add_pool(x, batch=batch)
        
        # Apply fully connected layters
        out = F.leaky_relu(self.fc2(mean_pool), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc3(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc4(out))
        return out