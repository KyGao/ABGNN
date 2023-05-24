import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from fairseq_models.data.abgen_dataset import ALPHABET, ATOM_TYPES, RES_ATOM14
from fairseq_models.modules.hmpn_encoder import *
from fairseq_models.modules.framework_encoder import HierarchicalDecoder
from fairseq_models.modules.nnutils import * 
from fairseq_models.modules.utils import *

class BertonlyDecoder(nn.Module):
    def __init__(self, args):
        super(BertonlyDecoder, self).__init__()

        self.args = args
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, enc_out, paratope, epitope, antibody, pt_emb, masked_tokens):
        bind_X, _, _, _ = paratope
        bind_X = torch.zeros((bind_X.size(0), bind_X.size(1), 4, 3), device=enc_out.device)
        
        return ReturnType(xloss=0., nll=0., bind_X=bind_X)
    
    def generate(self, enc_out, paratope, epitope, antibody, pt_emb, masked_tokens, num_decode):
        
        B, paratope_N = paratope[1].size(0), paratope[1].size(1)
        prob = F.softmax(enc_out.view(-1, len(ALPHABET)), dim=-1)
        bind_I = torch.multinomial(prob, num_samples=1).squeeze(-1)
        snll = self.ce_loss(enc_out.view(-1, len(ALPHABET)), bind_I)
        sloss = snll.view(B, paratope_N).mean(dim=1)

        S = bind_I.view(B, paratope_N).tolist()
        S = [''.join([ALPHABET[S[i][j]] for j in range(paratope_N)]) for i in range(B)]
        ppl = torch.exp(sloss / paratope_N)
        return ReturnType(handle=S, ppl=ppl, bind_X=None)
        # B, paratope_N = paratope[1].size(0), paratope[1].size(1)
        # prob = F.softmax(enc_out.view(-1, len(ALPHABET)), dim=-1)
        # bind_I = torch.multinomial(prob, num_samples=num_decode, replacement=True)
        # bind_I = torch.transpose(bind_I, 0, 1)
        # repeat_logits = enc_out.view(-1, len(ALPHABET)).repeat(num_decode, 1)
        # snll = self.ce_loss(repeat_logits, bind_I.reshape(-1))
        # sloss = snll.view(num_decode, paratope_N).mean(dim=1)

        # S = bind_I.tolist()
        # S = [''.join([ALPHABET[S[i][j]] for j in range(paratope_N)]) for i in range(num_decode)]
        # ppl = torch.exp(sloss / paratope_N)
        # return ReturnType(handle=S, ppl=ppl, bind_X=None)

        # B, N = paratope[1].size(0), paratope[1].size(1)
        # bind_I = torch.zeros(B* N).cuda().long()
        # prob = F.softmax(enc_out.view(-1, len(ALPHABET)), dim=-1)


        # bind_I = torch.multinomial(prob, num_samples=1).squeeze(-1)
        
        # sloss = self.ce_loss(enc_out.view(-1, len(ALPHABET)), bind_I)
        # S = bind_I.view(B, N).tolist()
        # S = [''.join([ALPHABET[S[i][j]] for j in range(N)]) for i in range(B)]
        # ppl = torch.exp(sloss / N)
        # ReturnType(handle=S, ppl=ppl, bind_X=None)



class FullshotRefineDecoderStack(ABModel):

    def __init__(self, args):
        super(FullshotRefineDecoderStack, self).__init__(args)
        self.args = args
        self.hierarchical = args.hierarchical
        self.residue_atom14 = torch.tensor([
                [ATOM_TYPES.index(a) for a in atoms] for atoms in RES_ATOM14
        ]).cuda()

        self.W_s = nn.Linear(args.hidden_size, len(ALPHABET))
        self.W_t = nn.Linear(self.embedding.dim(), args.hidden_size)
        self.U_i = nn.Linear(self.embedding.dim(), args.hidden_size)

        # self.W_trans = nn.Linear(args.hidden_size, self.embedding.dim())
        self.coord_loss = nn.SmoothL1Loss(reduction='sum')

        if args.hierarchical:
            self.struct_mpn = HierEGNNEncoder(args)
            self.seq_mpn = HierEGNNEncoder(args, update_X=False, backbone_CA_only=False)
        else:
            self.struct_mpn = EGNNEncoder(args)
            self.seq_mpn = EGNNEncoder(args, update_X=False)

        self.framework_encoder = HierarchicalDecoder(args)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def struct_loss(self, antibody_X, epitope_X, true_V, true_R, true_D, inter_D, true_C):
        # dihedral loss
        antibody_V = self.features._dihedrals(antibody_X)
        vloss = self.mse_loss(antibody_V, true_V).sum(dim=-1)
        # local loss
        rdist = antibody_X.unsqueeze(2) - antibody_X.unsqueeze(3)
        rdist = torch.sum(rdist ** 2, dim=-1)
        rloss = self.huber_loss(rdist, true_R) + 10 * F.relu(1.5 - rdist)
        # full loss
        cdist, _ = full_square_dist(antibody_X, antibody_X, torch.ones_like(antibody_X)[..., 0], torch.ones_like(antibody_X)[..., 0])
        closs = self.huber_loss(cdist, true_C) + 10 * F.relu(1.5 - cdist)
        # alpha carbon
        antibody_X, epitope_X = antibody_X[:, :, 1], epitope_X[:, :, 1]
        # CDR self distance
        dist = antibody_X.unsqueeze(1) - antibody_X.unsqueeze(2)
        dist = torch.sum(dist ** 2, dim=-1)
        dloss = self.huber_loss(dist, true_D) + 10 * F.relu(14.4 - dist)
        # inter distance
        idist = antibody_X.unsqueeze(2) - epitope_X.unsqueeze(1)
        idist = torch.sum(idist ** 2, dim=-1)
        iloss = self.huber_loss(idist, inter_D) + 10 * F.relu(14.4 - idist)
        return dloss, vloss, rloss, iloss, closs

    def forward(
        self, init_prob, paratope, epitope, antibody,
        pretrained_embedding=None, masked_tokens=None
    ):
        # return ReturnType(xloss=3.0, nll=2., bind_X=torch.ones(masked_tokens.sum().item(), 14, 3), handle=(None, None))
        _, true_cdr_S, flatten_cdr_S, flatten_init_X = paratope
        epitope_X, epitope_S, epitope_A = epitope
        antibody_X, antibody_S, antibody_cdr, padding_mask = antibody


        # Encode target
        epitope_h = self.U_i(self.embedding(epitope_S))
        epitope_V = self.features._dihedrals(epitope_X)
        epitope_mask = epitope_A[:,:,1].clamp(max=1).float()
        
        # antibody_h_0: initialized antibody embedding with shape [B, antibody_N, 256]
        # paratope_mask: mask for paratope nodes with shape [B, antibody_N]
        # padding_mask: mask for existing nodes with shape [B, antibody_N]
        init_prob = F.softmax(init_prob, dim=-1)
        antibody_h_0, true_antibody_X, paratope_mask, antibody_X, padding_mask = self.framework_encoder(
            antibody_X, antibody_S, antibody_cdr, padding_mask, init_prob, masked_tokens, flatten_init_X.detach().clone()
        )

        B, paratope_N = true_cdr_S.size(0), true_cdr_S.size(1)
        antibody_N = paratope_mask.size(1)
        
        antibody_A = torch.zeros(B, antibody_N, 14).cuda().long()
        antibody_A[padding_mask>0] = torch.tensor(
            [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        ).cuda().expand(int(padding_mask.sum()), -1)  # backbone atoms
        
        # Refine
        dloss = vloss = rloss = iloss = sloss = closs = 0
        for t in range(self.args.refine_iteration):
            # sequence update
            antibody_V = self.features._dihedrals(antibody_X.detach())
            complex_V = torch.cat([antibody_V, epitope_V], dim=1).detach()
            complex_X = torch.cat([antibody_X, epitope_X], dim=1).detach()
            complex_A = torch.cat([antibody_A, epitope_A], dim=1).detach()

            # sequence message passing
            complex_h = self.W_i(torch.cat([antibody_h_0, epitope_h], dim=1))
            complex_h, _ = self.seq_mpn(complex_X, complex_V, complex_h, complex_A, antibody_N=antibody_N, paratope_mask=paratope_mask, pretrained_embedding=pretrained_embedding)
            logits = self.W_s(complex_h[:, :antibody_N])
            logits = logits[paratope_mask]
            antibody_h_0 = antibody_h_0.clone()

            # calculate sequence loss
            snll = self.ce_loss(logits, flatten_cdr_S)
            sloss = sloss + torch.sum(snll)

            # update paratope embedding
            probs = F.softmax(logits, dim=-1)
            antibody_h_0[paratope_mask] = self.W_t(self.embedding.soft_forward(probs))

            # structrue message passing
            complex_h = self.W_i(torch.cat([antibody_h_0, epitope_h], dim=1))
            complex_h, complex_X = self.struct_mpn(complex_X, complex_V, complex_h, complex_A, antibody_N=antibody_N, paratope_mask=paratope_mask, pretrained_embedding=pretrained_embedding)
            antibody_X = antibody_X.clone()
            antibody_X[paratope_mask] = complex_X[:, :antibody_N][paratope_mask]

            ratio = (t + 1) / self.args.refine_iteration
            label_X = true_antibody_X * ratio + antibody_X * (1 - ratio)
        
            true_V = self.features._dihedrals(label_X)
            true_R, rmask_2D = inner_square_dist(label_X, antibody_A.clamp(max=1).float())
            true_D, mask_2D = self_square_dist(label_X, paratope_mask)
            true_C, cmask_2D = full_square_dist(label_X, label_X, antibody_A, antibody_A)
            inter_D, imask_2D = cross_square_dist(label_X, epitope_X, paratope_mask, epitope_mask)

            dloss_t, vloss_t, rloss_t, iloss_t, closs_t = self.struct_loss(
                    antibody_X, epitope_X, true_V, true_R, true_D, inter_D, true_C
            )
            vloss = vloss + vloss_t * paratope_mask
            dloss = dloss + dloss_t * mask_2D
            iloss = iloss + iloss_t * imask_2D
            rloss = rloss + rloss_t * rmask_2D
            closs = closs + closs_t * cmask_2D
    

        sloss = sloss / paratope_mask.sum() # / self.args.refine_iteration
        dloss = torch.sum(dloss) / mask_2D.sum() 
        iloss = torch.sum(iloss) / imask_2D.sum() 
        vloss = torch.sum(vloss) / paratope_mask.sum() 
        # print('mask_2D', mask_2D.sum())
        if self.hierarchical:
            rloss = torch.sum(rloss) / rmask_2D.sum()
            closs = torch.sum(closs) / cmask_2D.sum()
        else:
            rloss = torch.sum(rloss[:,:,:4,:4]) / rmask_2D[:,:,:4,:4].sum()
            closs = 0


        struct_loss = (dloss + iloss + vloss + rloss + closs) / paratope_mask.sum() # / self.args.refine_iteration
        seq_loss = sloss
        return ReturnType(xloss=struct_loss, nll=seq_loss, bind_X=antibody_X[paratope_mask].detach(), handle=(epitope_X, epitope_A))

    def generate(
        self, init_prob, paratope, epitope, antibody,
        pretrained_embedding=None, masked_tokens=None, num_decode=1
    ):
        _, true_cdr_S, flatten_cdr_S, flatten_init_X = paratope
        epitope_X, epitope_S, epitope_A = epitope
        antibody_X, antibody_S, antibody_cdr, antibody_mask = antibody


        # Encode target (assumes same target)
        epitope_h = self.U_i(self.embedding(epitope_S))
        epitope_V = self.features._dihedrals(epitope_X)
        epitope_mask = epitope_A[:,:,1].clamp(max=1).float()

        # antibody_h_0: initialized antibody embedding with shape [B, antibody_N, 256]
        # paratope_mask: mask for paratope nodes with shape [B, antibody_N]
        # padding_mask: mask for existing nodes with shape [B, antibody_N]
        init_prob = F.softmax(init_prob, dim=-1)
        antibody_h_0, _, paratope_mask, antibody_X, padding_mask = self.framework_encoder(
            antibody_X, antibody_S, antibody_cdr, antibody_mask, init_prob, masked_tokens, flatten_init_X.detach().clone()
        )

        B, paratope_N = true_cdr_S.size(0), true_cdr_S.size(1)
        antibody_N = paratope_mask.size(1)
        
        antibody_A = torch.zeros(B, antibody_N, 14).cuda().long()
        antibody_A[padding_mask>0] = torch.tensor(
            [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        ).cuda().expand(int(padding_mask.sum()), -1)  # backbone atoms
        
        # Refine
        for t in range(self.args.refine_iteration):
            # sequence update
            antibody_V = self.features._dihedrals(antibody_X.detach())
            complex_V = torch.cat([antibody_V, epitope_V], dim=1).detach()
            complex_X = torch.cat([antibody_X, epitope_X], dim=1).detach()
            complex_A = torch.cat([antibody_A, epitope_A], dim=1).detach()

            # sequence message passing
            complex_h = self.W_i(torch.cat([antibody_h_0, epitope_h], dim=1))
            complex_h, _ = self.seq_mpn(complex_X, complex_V, complex_h, complex_A, antibody_N=antibody_N, paratope_mask=paratope_mask, pretrained_embedding=pretrained_embedding)
            logits = self.W_s(complex_h[:, :antibody_N])
            logits = logits[paratope_mask]
            antibody_h_0 = antibody_h_0.clone()

            # update paratope embedding
            probs = F.softmax(logits, dim=-1)
            antibody_h_0[paratope_mask] = self.W_t(self.embedding.soft_forward(probs))

            # structrue message passing
            complex_h = self.W_i(torch.cat([antibody_h_0, epitope_h], dim=1))
            complex_h, complex_X = self.struct_mpn(complex_X, complex_V, complex_h, complex_A, antibody_N=antibody_N, paratope_mask=paratope_mask, pretrained_embedding=pretrained_embedding)
            antibody_X = antibody_X.clone()
            antibody_X[paratope_mask] = complex_X[:, :antibody_N][paratope_mask]
    
        # sample new sequences
        prob = F.softmax(logits.view(-1, len(ALPHABET)), dim=-1)
        bind_I = torch.multinomial(prob, num_samples=num_decode, replacement=True)
        bind_I = torch.transpose(bind_I, 0, 1)
        repeat_logits = logits.view(-1, len(ALPHABET)).repeat(num_decode, 1)
        snll = self.ce_loss(repeat_logits, bind_I.reshape(-1))
        sloss = snll.view(num_decode, paratope_N).mean(dim=1)

        S = bind_I.tolist()
        S = [''.join([ALPHABET[S[i][j]] for j in range(paratope_N)]) for i in range(num_decode)]
        ppl = torch.exp(sloss / paratope_N)
        return ReturnType(handle=S, ppl=ppl, bind_X=antibody_X[paratope_mask].detach())
