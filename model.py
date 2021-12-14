import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
from mlp import MLP
import time
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RW_layer(nn.Module):  
    def __init__(self, input_dim, out_dim, hidden_dim = None, max_step = 1, size_graph_filter = 10, dropout = 0.5):
        super(RW_layer, self).__init__()
        self.max_step = max_step
        self.size_graph_filter = size_graph_filter
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.fc_in = torch.nn.Linear(input_dim, hidden_dim)
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, hidden_dim, out_dim))
        else:
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, input_dim, out_dim))
        self.adj_hidden = Parameter(torch.FloatTensor( (size_graph_filter*(size_graph_filter-1))//2 , out_dim))
        self.bn = nn.BatchNorm1d(out_dim)
  
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        
    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)

    def forward(self, adj, features, idxs):
        adj_hidden_norm = torch.zeros( self.size_graph_filter, self.size_graph_filter, self.out_dim).to(device)
        idx = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1)
        adj_hidden_norm[idx[0], idx[1], :] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)
        
        #construct feature array for each subgraph
        x = features
        if self.hidden_dim:
            x = nn.ReLU()(self.fc_in(x)) # (#G, D_hid)
        x = torch.cat([x,torch.zeros(1, x.shape[1]).to(device)])
        x = x[idxs] # (#G, #Nodes_sub, D_hid)
        
        #construct feature array for each graph filter
        z = self.features_hidden # (Nhid,Dhid,Dout)

        zx = torch.einsum("mcn,abc->ambn", (z, x)) # (#G, #Nodes_filter, #Nodes_sub, D_out)
        out = []
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.size_graph_filter, device=device)             
                o = torch.einsum("ab,bcd->acd", (eye, z))
                t = torch.einsum("mcn,abc->ambn", (o, x))
            else:
                x = torch.einsum("abc,acd->abd",(adj, x))
                z = torch.einsum("abd,bcd->acd", (adj_hidden_norm, z)) # adj_hidden_norm: (Nhid,Nhid,Dout)
                t = torch.einsum("mcn,abc->ambn", (z, x))
            t = self.dropout(t) 
            t = torch.mul(zx, t) # (#G, #Nodes_filter, #Nodes_sub, D_out)
            t = torch.mean(t, dim=[1,2])
            out.append(t)
        out = sum(out)/len(out)
        return out

class DRW_layer(nn.Module):  
    def __init__(self, input_dim, out_dim, hidden_dim = None, max_step = 1, size_graph_filter = 10, dropout = 0.5, size_subgraph = None):
        super(DRW_layer, self).__init__()
        self.max_step = max_step
        self.size_graph_filter = size_graph_filter
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.fc_in = torch.nn.Linear(input_dim, hidden_dim)
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, hidden_dim, out_dim))
        else:
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, input_dim, out_dim))

        self.adj_hidden = Parameter(torch.FloatTensor( (size_graph_filter*(size_graph_filter-1))//2 , out_dim))

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.kerlin_weights = torch.nn.Linear(size_graph_filter*size_subgraph,1)

    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
        
    def forward(self, adj, features, idxs):
    
        adj_hidden_norm = torch.zeros( self.size_graph_filter, self.size_graph_filter,self.out_dim).to(device)
        idx = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1)
        adj_hidden_norm[idx[0],idx[1],:] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)

        x = features
        if self.hidden_dim:
            x = self.sigmoid(self.fc_in(x))
  
        x = torch.cat([x,torch.zeros(1,x.shape[1]).to(device)])
        x = x[idxs]
        z = self.features_hidden 

        zx = torch.einsum("mcn,abc->ambn", (z, x))
        out = []
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.size_graph_filter, device=device)             
                o = torch.einsum("ab,bcd->acd", (eye, z))
                t = torch.einsum("mcn,abc->ambn", (o, x))
            else:
                x_sub = torch.einsum("abc,acd->abd",(adj, x))
                z = torch.einsum("abd,bcd->acd", (adj_hidden_norm, z)) 
                t = torch.einsum("mcn,abc->ambn", (z, x))
            t = self.dropout(t) 
            t = torch.mul(zx, t) # (#G, #Nodes_filter, #Nodes_sub, D_out)
            t = t.permute(0,3,1,2).reshape(t.shape[0],t.shape[3],-1)
            t = self.kerlin_weights(t).squeeze()
            out.append(t)
        out = sum(out)/len(out)
        return out

class kergnn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = None, kernel = 'rw', size_graph_filter = None, num_mlp_layers = 1, mlp_hidden_dim = None, max_step = 1, dropout_rate=0.5, size_subgraph = None, no_norm = False):
        
        super(kergnn, self).__init__()
        self.no_norm = no_norm
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_layers = len(hidden_dims)-1

        self.ker_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                if kernel == 'rw':
                    self.ker_layers.append(RW_layer(input_dim, hidden_dims[1], hidden_dim = hidden_dims[0], max_step = max_step, size_graph_filter = size_graph_filter[0], dropout = dropout_rate))
                elif kernel == 'drw':
                    self.ker_layers.append(DRW_layer(input_dim, hidden_dims[1], hidden_dim = hidden_dims[0], max_step = max_step, size_graph_filter = size_graph_filter[0], dropout = dropout_rate, size_subgraph = size_subgraph))
                else:
                    exit('Error: unrecognized model')
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[1]))
            else:
                if kernel == 'rw':
                    self.ker_layers.append(RW_layer(hidden_dims[layer], hidden_dims[layer+1], hidden_dim = None, max_step = max_step, size_graph_filter = size_graph_filter[layer], dropout = dropout_rate))
                elif kernel == 'drw':
                    self.ker_layers.append(DRW_layer(hidden_dims[layer], hidden_dims[layer+1], hidden_dim = None,max_step = max_step, size_graph_filter = size_graph_filter[layer], dropout = dropout_rate, size_subgraph = size_subgraph ))
                else:
                    exit('Error: unrecognized model')            
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[layer+1]))

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.num_layers + 1):
            if layer == 0:
                self.linears_prediction.append(MLP(num_mlp_layers, input_dim, mlp_hidden_dim, output_dim))
            else:
                self.linears_prediction.append(MLP(num_mlp_layers, hidden_dims[layer], mlp_hidden_dim, output_dim))

    def forward(self, adj, features, idxs, graph_indicator):
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.size(0)

        hidden_rep = [features]
        h = features

        for layer in range(self.num_layers):
            h = self.ker_layers[layer](adj, h, idxs)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.zeros(n_graphs, h.shape[1], device=device).index_add_(0, graph_indicator, h)               
            if not self.no_norm:
                norm = counts.unsqueeze(1).repeat(1, pooled_h.shape[1])
                pooled_h = pooled_h/norm
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.dropout_rate, training = self.training)

        return score_over_layer