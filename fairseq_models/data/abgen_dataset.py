import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json, copy
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from fairseq_models.modules.utils import full_square_dist
from functools import lru_cache
from pathlib import Path

from fairseq.data import (
    data_utils,
    Dictionary,
    FairseqDataset
)

from .ab_dictionary import (
    ALPHABET,
    ALPHABET_FULL,
    RESTYPE_1to3,
    ATOM_TYPES,
    RES_ATOM14,
    UNK_LIST
)


class AntibodyComplexDataset(FairseqDataset):

    def __init__(self, data_path, split, seq_vocab, tag_vocab, cdr_types, L_target, pad_idx, mask_idx, max_len):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        
        data_path = Path(data_path)
        data_file = f'{split}_data.jsonl'
        jsonl_file = data_path / data_file

        self.seq_vocab = seq_vocab
        self.tag_vocab = tag_vocab
        self.cdr_types = cdr_types
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx

        self.cdr_types = cdr_types
        self.data = []
        with open(jsonl_file) as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines):
                entry = json.loads(line)
                assert len(entry['antibody_coords']) == len(entry['antibody_seq'])
                assert len(entry['antigen_coords']) == len(entry['antigen_seq'])
                for cdr in cdr_types:
                    if entry['antibody_cdr'].count(cdr) <= 4:
                        continue
                if entry['pdb'] in UNK_LIST:
                    continue
                if len(entry['antibody_seq']) > max_len:
                    continue
                
                # paratope region
                surface = torch.tensor(
                        [i for i,v in enumerate(entry['antibody_cdr']) if v in cdr_types]
                )
                entry['paratope_surface'] = surface
                entry['cdr_string'] = entry['antibody_cdr']

                l_coord, r_coord = torch.tensor(entry['antibody_coords'])[surface[0] - 1], torch.tensor(entry['antibody_coords'])[surface[-1] + 1]
                n_span = len(surface) + 1
                coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, 14, 3) 
                coord_offsets = torch.cumsum(coord_offsets, dim=0)
                mask_cdr_coords = l_coord + coord_offsets / n_span # [cdr_len, 14, 3]
                mask_cdr_coords[:, 4:] = mask_cdr_coords[:, 1].unsqueeze(1) # regularize side chain atoms
                entry['mask_cdr_coords'] = mask_cdr_coords
                
                entry['paratope_seq'] = ''.join([entry['antibody_seq'][i] for i in surface.tolist()])
                entry['paratope_coords'] = torch.tensor(entry['antibody_coords'])[surface]
                entry['paratope_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['paratope_seq']]
                )
                if entry['paratope_coords'][0][0].norm(dim=-1) < 1e-6:
                    # print('here')
                    continue
                mask = (entry['paratope_coords'].norm(dim=-1) > 1e-6).long()
                entry['paratope_atypes'] *= mask

                # Create epitope
                entry['antigen_seq'] = entry['antigen_seq']
                entry['antigen_coords'] = torch.tensor(entry['antigen_coords'])
                entry['antigen_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['antigen_seq']]
                )
                mask = (entry['antigen_coords'].norm(dim=-1) > 1e-6).long()
                entry['antigen_atypes'] *= mask

                # Find epitope surface
                dist, _ = full_square_dist(
                        entry['antigen_coords'][None,...], 
                        entry['paratope_coords'][None,...], 
                        entry['antigen_atypes'][None,...], 
                        entry['paratope_atypes'][None,...], 
                        contact=True
                )
                K = min(len(dist[0]), L_target)
                epitope = dist[0].amin(dim=-1).topk(k=K, largest=False).indices
                entry['epitope_surface'] = torch.sort(epitope).values

                if len(entry['paratope_coords']) > 4 and len(entry['antigen_coords']) > 4 and entry['antibody_cdr'].count('001') <= 1:
                    # string to list
                    entry['antibody_seq_str'] = entry['antibody_seq']
                    entry['antigen_seq_str'] = entry['antigen_seq']
                    entry['antibody_seq'] = torch.tensor([ALPHABET_FULL.index(a) for a in entry['antibody_seq']])
                    entry['paratope_seq'] = torch.tensor([ALPHABET.index(a) for a in entry['paratope_seq']])
                    entry['antigen_seq'] = torch.tensor([ALPHABET.index(a) for a in entry['antigen_seq']])
                    entry['antibody_cdr'] = torch.tensor([self.tag_vocab.index(a) for a in entry['antibody_cdr']])
                    entry['antibody_coords'] = torch.tensor(entry['antibody_coords'])

                    # make masked dataset
                    # only implement one cdr predicting
                    for cdr in self.cdr_types:
                        cdr_mask = entry['antibody_cdr'] == self.tag_vocab.index(cdr)
                        entry['prefix_len'] = entry['cdr_string'].index(cdr)
                    # print(cdr_mask)
                    label_seq = torch.full_like(entry['antibody_seq'], self.pad_idx)
                    label_seq[cdr_mask] = entry['antibody_seq'][cdr_mask]

                    entry['label_antibody_seq'] = label_seq
                    
                    masked_seq = torch.full_like(entry['antibody_seq'], self.mask_idx)
                    masked_seq[~cdr_mask] = entry['antibody_seq'][~cdr_mask]
                    entry['masked_antibody_seq'] = masked_seq

                    # from bert vocab to decoder vocab
                    entry['antibody_seq'] = entry['antibody_seq'] - 3

                    self.data.append(entry)

        self.sizes = np.array([len(item['paratope_seq']) for item in self.data])
        self.prefix_len = np.array([item['prefix_len'] for item in self.data])
        

    def __len__(self):
        return len(self.data)

    @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['masked_antibody_seq'], item['antibody_cdr'], item['label_antibody_seq'], \
               item['paratope_seq'], item['paratope_coords'], item['paratope_atypes'], \
               item['antigen_seq'], item['antigen_coords'], item['antigen_atypes'], \
               item['paratope_surface'], item['epitope_surface'], item['mask_cdr_coords'], \
               item['antibody_coords'], item['antibody_seq'], item['cdr_string']



    def collater(self, samples):
        return self.collate_fn(samples)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]
    
    def collate_fn(self, batch):
        try:
            masked_antibody_seq, antibody_cdr, label_antibody_seq, \
            paratope_seq, paratope_coords, paratope_atypes, \
            antigen_seq, antigen_coords, antigen_atypes, \
            paratope_surface, epitope_surface, mask_cdr_coords, \
            antibody_coords, antibody_seq, cdr_string \
            = tuple(zip(*batch))
        except:
            return None

        # for bert
        batched_seq = data_utils.collate_tokens(masked_antibody_seq, self.pad_idx, left_pad=False)
        batched_tag = data_utils.collate_tokens(antibody_cdr, self.pad_idx, left_pad=False)
        batched_label = data_utils.collate_tokens(label_antibody_seq, self.pad_idx, left_pad=False)

        # for decoder 
        def featurize_antigen(seq_batch, coords_batch, atypes_batch):
            S = pad_sequence(seq_batch, batch_first=True, padding_value=0)
            X = pad_sequence(coords_batch, batch_first=True, padding_value=0)
            A = pad_sequence(atypes_batch, batch_first=True, padding_value=0)
            return X, S, A

        def featurize_paratope(seq_batch, coords_batch, mask_cdr_coords_batch):
            X = pad_sequence(coords_batch, batch_first=True, padding_value=0)
            S = pad_sequence(seq_batch, batch_first=True, padding_value=0)
            X_init = torch.cat([i for i in mask_cdr_coords_batch])
            cont_S = torch.cat([i for i in seq_batch])
            return X, S, cont_S, X_init

        def feature_framework(antibody_seq, antibody_coords, antibody_cdr):
            X = pad_sequence(antibody_coords, batch_first=True, padding_value=0)
            S = pad_sequence(antibody_seq, batch_first=True, padding_value=0)
            antibody_cdr = list(antibody_cdr)
            mask = S.bool().float()
            return X, S, antibody_cdr, mask

        def select_target(tgt_X, tgt_S, tgt_A, tgt_pos):
            max_len = max([len(pos) for pos in tgt_pos])
            xlist = [tgt_X[i, pos] for i,pos in enumerate(tgt_pos)]
            slist = [tgt_S[i, pos] for i,pos in enumerate(tgt_pos)]
            alist = [tgt_A[i, pos] for i,pos in enumerate(tgt_pos)]
            tgt_X = [F.pad(x, (0,0,0,0,0,max_len-len(x))) for x in xlist]
            tgt_S = [F.pad(s, (0,max_len-len(s))) for s in slist]
            tgt_A = [F.pad(a, (0,0,0,max_len-len(a))) for a in alist]
            return torch.stack(tgt_X, dim=0), torch.stack(tgt_S, dim=0), torch.stack(tgt_A, dim=0)

        antigen = featurize_antigen(antigen_seq, antigen_coords, antigen_atypes)
        epitope = select_target(antigen[0], antigen[1], antigen[2], [t_s for t_s in epitope_surface]) # select epitope from antigen
        paratope = featurize_paratope(paratope_seq, paratope_coords, mask_cdr_coords)
        antibody = feature_framework(antibody_seq, antibody_coords, cdr_string)
        return batched_seq, batched_tag, batched_label, paratope, epitope, antibody

        

