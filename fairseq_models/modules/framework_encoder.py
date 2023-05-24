import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import *
from .nnutils import * 
from .protein_features import ProteinFeatures



class HierarchicalDecoder(nn.Module):

    def __init__(self, args):
        super(HierarchicalDecoder, self).__init__()
        self.cdr_type = args.cdr_type
        self.k_neighbors = args.k_neighbors
        self.block_size = args.block_size
        self.hidden_size = args.hidden_size
        self.args = args

        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='full',
                direction='bidirectional'
        )
        # self.node_in, self.edge_in = self.features.feature_dimensions['full']
        self.embedding = AAEmbedding()
        self.W_s = nn.Linear(self.embedding.dim(), args.hidden_size)
        
        self.rnn = nn.GRU(
                args.hidden_size, args.hidden_size, batch_first=True, 
                num_layers=1, bidirectional=True
        )


        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
    def mask_mean(self, X, mask, i):
        # [B, N, 4, 3] -> [B, 1, 4, 3] / [B, 1, 1, 1]
        X = X[:, i:i+self.block_size]
        if X.dim() == 4:
            mask = mask[:, i:i+self.block_size].unsqueeze(-1).unsqueeze(-1)
        else:
            mask = mask[:, i:i+self.block_size].unsqueeze(-1)
        return torch.sum(X * mask, dim=1, keepdims=True) / (mask.sum(dim=1, keepdims=True) + 1e-8)

    def make_X_blocks(self, X, l, r, mask):
        N = X.size(1)
        lblocks = [self.mask_mean(X, mask, i) for i in range(0, l, self.block_size)]
        rblocks = [self.mask_mean(X, mask, i) for i in range(r + 1, N, self.block_size)]
        bX = torch.cat(lblocks + [X[:, l:r+1]] + rblocks, dim=1)
        return bX.detach()

    def make_S_blocks(self, hS, l, r, mask):
        N = hS.size(1) # 130
        # l=T_min=96, r=T_max=109, range of cdr-3
        # block_size=8
        # ==> len(LS)=12, LS[i].shape=[7, 1, 256]
        # ==> len(RS)=3, RS[i].shape=[7, 1, 256]
        lseqs = [self.mask_mean(hS, mask, i) for i in range(0, l, self.block_size)] # mask_mean: calculate mean without mask
        rseqs = [self.mask_mean(hS, mask, i) for i in range(r + 1, N, self.block_size)]
        bS = torch.cat(lseqs + [hS[:, l:r+1]] + rseqs, dim=1) # 12 + 14 + 3 = 29 ==> [7, 29, 256]
        return bS, len(lseqs), len(rseqs)
    
    def make_mask_blocks(self, mask, l, r):
        N = mask.size(1)
        lmask = [mask[:, i:i+self.block_size].amax(dim=1, keepdims=True) for i in range(0, l, self.block_size)] # amax: max val of each row
        rmask = [mask[:, i:i+self.block_size].amax(dim=1, keepdims=True) for i in range(r + 1, N, self.block_size)] # if one elem masked, the whole block masked
        bmask = torch.cat(lmask + [mask[:, l:r+1]] + rmask, dim=1)
        return bmask

    def get_completion_mask(self, B, N, cdr_range):
        cmask = torch.zeros(B, N).cuda()
        for i, (l,r) in enumerate(cdr_range): 
            cmask[i, l:r+1] = 1
        return cmask

    def remove_cdr_coords(self, X, cdr_range):
        X = X.clone()
        for i, (l,r) in enumerate(cdr_range):
            X[i, l:r+1, :, :] = 0
        return X.clone()

    def forward(self, antibody_X, antibody_S, antibody_cdr, padding_mask, c_init_prob, paratope_mask, init_X):
        B, N = padding_mask.size(0), padding_mask.size(1)
        
        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in antibody_cdr]
        T_min = min([l for l,r in cdr_range])
        T_max = max([r for l,r in cdr_range])
        
        antibody_init_X = antibody_X.clone()
        # print(init_X.shape) [8, 13, 14, 3]
        # print(antibody_init_X.shape) [8, 221, 14, 3]
        # print(padding_mask)
        # print(padding_mask.shape)
        # print(antibody_init_X[padding_mask>0].shape)
        antibody_init_X[paratope_mask>0] = init_X
        antibody_init_X = self.make_X_blocks(antibody_init_X, T_min, T_max, padding_mask)

        # make blocks and encode framework
        S = antibody_S.clone() * (1 - paratope_mask.long()) # type_id, 0 for cdr-3 [7, 130]
        seq_emb = self.embedding(S)
        
        seq_emb[paratope_mask>0] = self.embedding.soft_forward(c_init_prob)
        hS = self.W_s(seq_emb)
        hS, offset, suffix = self.make_S_blocks(hS, T_min, T_max, padding_mask)
        paratope_mask = torch.cat([paratope_mask.new_zeros(B, offset), paratope_mask[:, T_min:T_max+1], paratope_mask.new_zeros(B, suffix)], dim=1) # [7, 12+14+3]

        
        # Ground truth 
        antibody_X = self.make_X_blocks(antibody_X, T_min, T_max, padding_mask) # [7, 130, 4, 3] ==> [7, 29, 4, 3]
        
        padding_mask = self.make_mask_blocks(padding_mask, T_min, T_max)

        return hS, antibody_X, paratope_mask, antibody_init_X, padding_mask
