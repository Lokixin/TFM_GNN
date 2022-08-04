import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_add_pool, ChebConv, global_max_pool, SAGPooling, GATConv, GATv2Conv, TransformerConv, SuperGATConv, global_mean_pool
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

class EEGGraphConvNetTemporal(nn.Module):
    
    def __init__(self, **kwargs):
        super(EEGGraphConvNetTemporal, self).__init__()
        # Layers definition
        input_size = kwargs.pop("input_size", 1280)        
        # Graph convolutional layers
        self.conv1 = GCNConv(input_size, 640, cached=True, normalize=False)
        self.conv2 = GCNConv(640, 512, cached=True, normalize=False)
        self.conv3 = GCNConv(512, 256, cached=True, normalize=False)
        self.conv4 = GCNConv(256, 256, cached=True, normalize=False)
        
        # Batch normalization
        self.batch_norm1 = BatchNorm(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm3 = BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm4 = BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        x = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index, edge_weigth)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x, edge_index, edge_weigth)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm3(self.conv3(x, edge_index, edge_weigth)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm4(self.conv4(x, edge_index, edge_weigth)), negative_slope=0.01)
        #out = F.dropout(x, p = 0.2, training=self.training)

        add_pool = global_add_pool(x, batch=batch)
        out = F.leaky_relu(self.fc1(add_pool), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        out = F.leaky_relu(self.fc3(out))
        #out = F.dropout(out, p = 0.2, training=self.training)
        
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
        
        # Graph convolutional layers
        self.conv1 = GCNConv(1280, 640, cached=True, normalize=False)
        self.conv2 = GCNConv(640, 512, cached=True, normalize=False)
        self.conv3 = GCNConv(512, 256, cached=True, normalize=False)
        
        # Batch normalization
        self.batch_norm1 = BatchNorm(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = BatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm3 = BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm4 = BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.lstm = nn.LSTM(256, 256, 1, dropout=0.2)
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        
        x = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index, edge_weigth)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x, edge_index, edge_weigth)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm3(self.conv3(x, edge_index, edge_weigth)), negative_slope=0.01)
        x, _ = self.lstm(x)
        #x = F.leaky_relu(self.batch_norm4(self.conv4(x, edge_index, edge_weigth)), negative_slope=0.01)
        #x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        add_pool = global_add_pool(x, batch=batch)
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(add_pool), negative_slope=0.01)
        out = F.dropout(out, p = 0.2, training=self.training)
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        out = F.leaky_relu(self.fc3(out))
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
    

class EEGConvNetMiniV3(nn.Module):
    """Same as EEGGraphConvNet but with fewer 
    convolutional layers
    """
    def __init__(self, **kwargs):
        super(EEGConvNetMiniV3, self).__init__()
        # Layers definition
        # Graph convolutional layers
        self.conv1 = GCNConv(-1, 32, cached=True, normalize=False)
        self.conv2 = GCNConv(32, 64, cached=True, normalize=False)
        
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
        x = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index, edge_weigth)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x, edge_index, edge_weigth)), negative_slope=0.01)
        mean_pool = global_add_pool(x, batch=batch)
        out = F.leaky_relu(self.fc1(mean_pool), negative_slope=0.01)        
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        out = F.leaky_relu(self.fc3(out))
        return out
    

class EEGConvNetMiniV2Attention(nn.Module):
    """Same as EEGGraphConvNet but with fewer 
    convolutional layers
    """
    def __init__(self, **kwargs):
        super(EEGConvNetMiniV2Attention, self).__init__()
        # Layers definition
        # Graph convolutional layers
        conv_layers = {
            "gatconv": GATConv,
            "gatconv2": GATv2Conv,
            "transformer": TransformerConv,
            "super": SuperGATConv
        }
        
        AttentionConv = conv_layers[kwargs.get("attention_conv", "gatconv")]
        nheads = kwargs.get("nheads", 1)
        hidden_dim = kwargs.get("hidden_dim", 32)
        output_dim = kwargs.get("output_dim", 64)
        concat = kwargs.get("concat", True)
        nclasses = kwargs.get("nclasses", 2)
        
        self.conv1 = AttentionConv(-1, hidden_dim, heads=nheads, concat=concat) #16
        self.conv2 = AttentionConv(hidden_dim * nheads, output_dim, heads=nheads, concat=concat)
        self.conv3 = AttentionConv(output_dim * nheads, output_dim * nheads * 2, heads=nheads, concat=concat)
        
        # Batch normalization
        self.batch_norm1 = BatchNorm(hidden_dim * nheads, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = BatchNorm(output_dim * nheads, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm3 = BatchNorm(output_dim * nheads * 2 * nheads, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) #output_dim * nheads * 2 * 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(output_dim * nheads * 2 * nheads, output_dim * nheads * 2)
        self.fc2 = nn.Linear(output_dim * nheads * 2, output_dim * nheads)
        self.fc3 = nn.Linear((output_dim * nheads), nclasses)
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        # Perform all graph convolutions
        x = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x, edge_index)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm3(self.conv3(x, edge_index)), negative_slope=0.01)

        #x = F.dropout(x, p=0.2, training=self.training)
        #x = F.dropout(batch_norm_out, p=0.2, training=self.training)
        
        # Global add pooling
        mean_pool = global_add_pool(x, batch=batch)
        #print(mean_pool.size())
        
        # Apply fully connected layters
        out = F.leaky_relu(self.fc1(mean_pool), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        #out = F.dropout(out, p = 0.2, training=self.training)
        
        out = F.leaky_relu(self.fc3(out))
        return out
    
    
        
    
class MultiLevelConvNet(nn.Module):
    """Same as EEGGraphConvNet but with fewer 
    convolutional layers
    """
    def __init__(self, **kwargs):
        super(MultiLevelConvNet, self).__init__()
        # Layers definition
        # Graph convolutional layers
        self.conv1 = GCNConv(-1, 32, cached=True, normalize=False)
        self.conv2 = GCNConv(32, 32, cached=True, normalize=False)
        self.conv3 = GCNConv(32, 64, cached=True, normalize=False)
        
        
        # Batch normalization
        self.batch_norm1 = BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = BatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm3 = BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        x1 = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index, edge_weigth)), negative_slope=0.01)
        x2 = F.leaky_relu(self.batch_norm2(self.conv2(x1, edge_index, edge_weigth)), negative_slope=0.01)
        x3 = F.leaky_relu(self.batch_norm3(self.conv3(x2, edge_index, edge_weigth)), negative_slope=0.01)
        
        add_pool1 = global_add_pool(x1, batch=batch)
        add_pool2 = global_add_pool(x2, batch=batch)
        add_pool3 = global_add_pool(x3, batch=batch)
        
        out1 = F.leaky_relu(self.fc1(add_pool1), negative_slope=0.01)        
        out2 = F.leaky_relu(self.fc2(add_pool2), negative_slope=0.01)        
        out3 = F.leaky_relu(self.fc3(add_pool3), negative_slope=0.01)
        
        out = torch.cat((out1, out2, out3), dim=1)        
        out = self.classifier(out)
        return out
        
        
class MAGE(nn.Module):
    def __init__(self, **kwargs):
        super(MAGE, self).__init__()
        # Layers definition
        # Graph convolutional layers
        conv_layers = {
            "gatconv": GATConv,
            "gatconv2": GATv2Conv,
            "transformer": TransformerConv,
            "super": SuperGATConv
        }
        
        AttentionConv = conv_layers[kwargs.get("attention_conv", "gatconv")]
        nheads = kwargs.get("nheads", 1)
        hidden_dim = kwargs.get("hidden_dim", 32)
        output_dim = kwargs.get("output_dim", 64)
        concat = kwargs.get("concat", True)
        
        nclasses = kwargs.get("nclasses", 2)
        
        self.conv1 = AttentionConv(-1, hidden_dim, heads=nheads, concat=concat) #16
        self.conv2 = AttentionConv(hidden_dim * nheads, hidden_dim * nheads, heads=nheads, concat=concat)
        self.conv3 = AttentionConv(hidden_dim * nheads * nheads, hidden_dim * nheads * nheads, heads=nheads, concat=concat)
        
        # Batch normalization
        self.batch_norm1 = BatchNorm(hidden_dim * nheads, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = BatchNorm(hidden_dim * nheads * nheads, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm3 = BatchNorm(hidden_dim * nheads * nheads * nheads, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * nheads, hidden_dim * nheads)
        self.fc2 = nn.Linear(output_dim * nheads, output_dim * nheads)
        self.fc3 = nn.Linear(output_dim * nheads , output_dim * nheads * nheads)
        
        self.classifier = nn.Sequential(
            nn.Linear((hidden_dim * nheads) + (hidden_dim * nheads * nheads) + (hidden_dim * nheads * nheads * nheads), 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )
        
        # Xavier initializacion for fully connected layers
        self.fc1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        self.fc3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if isinstance(x, nn.Linear) else None)
        
        
    def forward(self, x, edge_index, edge_weigth, batch):
        # Perform all graph convolutions
        x1 = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index)), negative_slope=0.01)
        x2 = F.leaky_relu(self.batch_norm2(self.conv2(x1, edge_index)), negative_slope=0.01)
        x3 = F.leaky_relu(self.batch_norm3(self.conv3(x2, edge_index)), negative_slope=0.01)
        
        # Global add pooling
        add_pool1 = global_add_pool(x1, batch=batch)
        add_pool2 = global_add_pool(x2, batch=batch)
        add_pool3 = global_add_pool(x3, batch=batch)
        
        #out1 = F.leaky_relu(self.fc1(add_pool1), negative_slope=0.01)        
        #out2 = F.leaky_relu(self.fc2(add_pool2), negative_slope=0.01)        
        #out3 = F.leaky_relu(self.fc3(add_pool3), negative_slope=0.01)

        out = torch.cat((add_pool1, add_pool2, add_pool3), dim=1)        
        out = self.classifier(out)
        return out
    