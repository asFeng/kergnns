import os
import sys
import torch
import torch.utils.data as utils
import numpy as np
import logging
from math import ceil
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import OneHotEncoder,normalize
import networkx as nx

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

def neighborhood(G, node, n):
    paths = nx.single_source_shortest_path(G, node)
    return [node for node, traversed_nodes in paths.items()
            if len(traversed_nodes) == n+1]

def load_data(ds_name, use_node_labels=False,use_node_attri=False):    
    graph_indicator = np.loadtxt("datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name), dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)
    
    edges = np.loadtxt("datasets/%s/%s_A.txt"%(ds_name,ds_name), dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
    
    if use_node_labels:
        x = np.loadtxt("datasets/%s/%s_node_labels.txt"%(ds_name,ds_name), dtype=np.int64).reshape(-1,1)
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)
    elif use_node_attri:
        x = np.loadtxt("datasets/%s/%s_node_attributes.txt"%(ds_name,ds_name), delimiter=',',dtype=np.float64)#.reshape(-1,1)

    else:
        x = A.sum(axis=1)
        
    adj = []
    features = []
    idx = 0
    for i in range(graph_size.size):
        adj.append(A[idx:idx+graph_size[i],idx:idx+graph_size[i]])
        features.append(x[idx:idx+graph_size[i],:])
        idx += graph_size[i]

    class_labels = np.loadtxt("datasets/%s/%s_graph_labels.txt"%(ds_name,ds_name), dtype=np.int64)
    return adj, features, class_labels

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def generate_batches(adj, features, y, batch_size, device, shuffle=False):
    N = len(y)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    n_batches = ceil(N/batch_size)

    adj_lst = list()
    features_lst = list()
    graph_indicator_lst = list()
    y_lst = list() 
    for i in range(0, N, batch_size):
        n_graphs = min(i+batch_size, N) - i
        n_nodes = sum([adj[index[j]].shape[0] for j in range(i, min(i+batch_size, N))])

        adj_batch = lil_matrix((n_nodes, n_nodes))
        features_batch = np.zeros((n_nodes, features[0].shape[1]))
        graph_indicator_batch = np.zeros(n_nodes)
        y_batch = np.zeros(n_graphs)

        idx = 0
        for j in range(i, min(i+batch_size, N)):
            n = adj[index[j]].shape[0]
            adj_batch[idx:idx+n, idx:idx+n] = adj[index[j]]    
            features_batch[idx:idx+n,:] = features[index[j]]
            graph_indicator_batch[idx:idx+n] = j-i
            y_batch[j-i] = y[index[j]]

            idx += n
                  
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
        features_lst.append(torch.FloatTensor(features_batch).to(device))
        graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
        y_lst.append(torch.LongTensor(y_batch).to(device))

    return adj_lst, features_lst, graph_indicator_lst, y_lst

def generate_sub_features_idx(adj_batch, features_batch, size_subgraph = 10, k_neighbor=1):
    sub_features_idx_list, sub_adj_list = [],[]
    for i in range(len(adj_batch)):
        adj = adj_batch[i]
        x = features_batch[i]
        num_B_nodes = x.shape[0]
        G = nx.from_numpy_array(adj.to_dense().cpu().numpy())
        subgraph_idx = []
        x_sub_adj = torch.zeros(x.shape[0], size_subgraph, size_subgraph).to( device)
        x_sub_idx = torch.zeros(x.shape[0], size_subgraph).to(device) 

        for node in range(x.shape[0]):

            # determine neighbors' idx
            tmp = []
            for k in range(k_neighbor+1):
                tmp = tmp + neighborhood(G, node, k)
            if len(tmp) > size_subgraph:
                tmp = tmp[:size_subgraph]
            sub_idxs = tmp
            
            if len(tmp) < size_subgraph:
                padded_sub_idxs = tmp + [num_B_nodes for i in range(size_subgraph-len(tmp))]
            else:
                padded_sub_idxs = tmp
     
            x_sub_idx[node] = torch.tensor(padded_sub_idxs)

            # corresponding neighbor and neighbor features      
            G_sub = G.subgraph(sub_idxs)
            tmp = nx.to_numpy_array(G_sub)
            if tmp.shape[0] < size_subgraph:
                tmp_adj = np.zeros([size_subgraph, size_subgraph])
                tmp_adj[:tmp.shape[0],:tmp.shape[1]] = tmp
                tmp = tmp_adj
            x_sub_adj_ = torch.from_numpy(tmp).float().to(device) #(Nsub,Nsub)
            if 2 in x_sub_adj_:
                x_sub_adj_ = x_sub_adj_/2
            x_sub_adj[node] = x_sub_adj_

        sub_features_idx_list.append(x_sub_idx.long()) 
        sub_adj_list.append(x_sub_adj)

    return sub_adj_list, sub_features_idx_list




