import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import networkx as nx
import time

from embed_methods.dgi.models import DGI, LogReg
from embed_methods.dgi.utils import process

def dgi(G, features):

  batch_size = 1
  nb_epochs = 10000
  patience = 20
  lr = 0.001
  l2_coef = 0.0
  drop_prob = 0.0
  hid_units = 512
  sparse = True
  nonlinearity = 'prelu' # special name to separate parameters

  adj = nx.to_scipy_sparse_matrix(G, weight='wgt')
  features = sp.lil_matrix(np.matrix(features))
  features, _ = process.preprocess_features(features)

  nb_nodes = features.shape[0]
  ft_size = features.shape[1]

  adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

  if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
  else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

  features = torch.FloatTensor(features[np.newaxis])
  if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])

  model = DGI(ft_size, hid_units, nonlinearity)
  optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

  if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()

  b_xent = nn.BCEWithLogitsLoss()
  xent = nn.CrossEntropyLoss()
  cnt_wait = 0
  best = 1e9
  best_t = 0

  for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()
    
    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    
    logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None) 
    loss = b_xent(logits, lbl)

    print('Loss:', loss)
    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print("epochs: ", epoch)
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()
  return (((model.embed(features, sp_adj if sparse else adj, sparse, None)[0]).squeeze()).data).cpu().numpy()
