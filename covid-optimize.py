import os
import argparse
from fairseq_models import AntibodyRobertaModel

from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders
from fairseq import checkpoint_utils
from tqdm import tqdm
from fairseq.data import Dictionary

from fairseq_models.data.abgen_dataset import AntibodyComplexDataset
from fairseq_models.data.ab_dictionary import ALPHABET
from fairseq_models.modules.utils import compute_rmsd
from fairseq.data import data_utils

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os
from copy import deepcopy
from tqdm import tqdm, trange

from neut_model import *
from structgen import *

from igfold import IgFoldRunner


class CovidNeutralizationModel():
    
    def __init__(self):
        MODEL_PATH = "checkpoints/pretrained/ckpts/covid/neut_model1.ckpt"
        MODEL_ARGS = {
            "hidden_dim": 256,
            "n_layers": 2,
            "use_srupp": False
        }
        MODEL_ARGS.update(MultiABOnlyCoronavirusModel.add_extra_args())
        self.model = MultiABOnlyCoronavirusModel.load_from_checkpoint(
            MODEL_PATH,
            **MODEL_ARGS,  # type: ignore
        )
        self.model.cuda()
        self.model.eval()

    def predict(self, vh, vl):
        ab_sequence = vh
        ab_sequence = torch.tensor([AA_VOCAB[aa] for aa in ab_sequence]).long()
        ab_sequence = ab_sequence.unsqueeze(0).cuda()
        neut_logits = self.model(ab_sequence)
        prob = torch.sigmoid(neut_logits)
        return prob[0, 1].item()

def data_curate(batch, seq_vocab, tag_vocab, mask_idx):
    entry = batch
    # construct data for ABGNN
    surface = torch.tensor(
            [i for i,v in enumerate(entry['cdr']) if v in ['3']]
    )
    paratope_surface = surface
    
    seq_len = len(entry['seq'])
    cdr_len = len(surface)
    antibody_coords = torch.zeros((seq_len, 4, 3))
    mask_cdr_coords = torch.zeros((cdr_len, 4, 3))
    paratope_coords = mask_cdr_coords.clone()

    paratope_seq = ''.join([entry['seq'][i] for i in surface.tolist()])
    paratope_seq = torch.tensor([ALPHABET.index(a) for a in paratope_seq])

    antibody_seq = torch.tensor([seq_vocab.index(a) for a in batch['seq']])
    antibody_cdr = torch.tensor([tag_vocab.index(a) for a in entry['cdr']])

    # data for AbBERT
    cdr_mask = antibody_cdr == tag_vocab.index('3')

    masked_antibody_seq = torch.full_like(antibody_seq, mask_idx)
    masked_antibody_seq[~cdr_mask] = antibody_seq[~cdr_mask]
    
    label_antibody_seq = torch.full_like(antibody_seq, seq_vocab.pad())
    label_antibody_seq[cdr_mask] = antibody_seq[cdr_mask]

    antibody_seq = antibody_seq - 3
    
    cdr_string = entry['cdr']
    
    return masked_antibody_seq.cuda(), antibody_cdr.cuda(), label_antibody_seq.cuda(), \
            paratope_seq.cuda(), paratope_coords.cuda(), \
            paratope_surface.cuda(), mask_cdr_coords.cuda(), \
            antibody_coords.cuda(), antibody_seq.cuda(), cdr_string


def collater(batch, pad_idx):
    try:
        masked_antibody_seq, antibody_cdr, label_antibody_seq, \
        paratope_seq, paratope_coords, \
        paratope_surface, mask_cdr_coords, \
        antibody_coords, antibody_seq, cdr_string \
        = tuple(zip(*batch))
    except:
        return None

    # for bert
    batched_seq = data_utils.collate_tokens(masked_antibody_seq, pad_idx, left_pad=False)
    batched_tag = data_utils.collate_tokens(antibody_cdr, pad_idx, left_pad=False)
    batched_label = data_utils.collate_tokens(label_antibody_seq, pad_idx, left_pad=False)

    # for decoder 
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

    paratope = featurize_paratope(paratope_seq, paratope_coords, mask_cdr_coords)
    antibody = feature_framework(antibody_seq, antibody_coords, cdr_string)
    return batched_seq, batched_tag, batched_label, paratope, antibody


def extract_fn(query):
    all_viruses = [
        x.replace("(weak)", "").strip()
        for x in query.replace(" and ", ";").replace(",", ";").split(";")
    ]
    filtered = [virus for virus in all_viruses if virus in RELEVANT_VIRUSES]
    return set(filtered)


def load_data():
    # Load antibody data
    with open("dataset/exp3-covabdab/CoV-AbDab_050821.csv", "r") as f:
        full_data = list(csv.DictReader(f))

    # First filter to relevant AB's
    full_data = [
        item
        for item in full_data
        if any(
            virus in item[key]
            for virus in RELEVANT_VIRUSES
            for key in RELEVANT_KEYS
        )
        and item["Ab or Nb"] == "Ab"  # remove nanobodies
        and len(item["VH or VHH"].strip()) > 2  # ensure sequence available
        and len(item["VL"].strip()) > 2  # ensure sequence available
        and "S" in item["Protein + Epitope"]  # ensure binds to S protein
        and TYPE_MAP[item["Protein + Epitope"]] == "rbd"
    ]

    sarscov2_ab_data = []
    for item in full_data:
        bindings = set()
        if item["Binds to"]:
            bindings = extract_fn(item["Binds to"])

        non_bindings = set()
        if item["Doesn't Bind to"]:
            non_bindings = extract_fn(item["Doesn't Bind to"])

        neutralizing = set()
        if item["Neutralising Vs"]:
            neutralizing = extract_fn(item["Neutralising Vs"])

        non_neutralizing = set()
        if item["Not Neutralising Vs"]:
            non_neutralizing = extract_fn(item["Not Neutralising Vs"])

        all_viruses = bindings | non_bindings | neutralizing | non_neutralizing
        full_label = [0, 0]
        full_mask = [0, 0]
        for virus in all_viruses:
            label = [0, 0]
            mask = [0, 0]
            if virus in bindings and neutralizing:
                label = [1, 1]
                mask = [1, 1]
            elif virus in bindings and non_neutralizing:
                label = [0, 1]
                mask = [1, 1]
            elif virus in non_bindings:
                label = [0, 0]
                mask = [1, 1]
            elif virus in neutralizing:
                label = [1, 1]
                mask = [1, 1]
            elif virus in bindings:
                label = [0, 1]
                mask = [0, 1]
            elif virus in non_neutralizing:
                label = [0, 1]
                mask = [1, 0]

            idx = virus == "SARS-CoV2"
            full_label[idx] = label[0]
            full_mask[idx] = mask[0]

        # Save sars-cov2 data for later
        sarscov2_ab_data.append({
            'name': item["\ufeffName"],
            'label': full_label[1],
            'mask': full_mask[1],
            'epitope': TYPE_MAP[item["Protein + Epitope"]],
            'ab': item["VH or VHH"].replace(" ", "") + "-" + item["VL"].replace(" ", ""),
            'hcdr3': item["CDRH3"],
            'lcdr3': item["CDRL3"],
        })
    return sarscov2_ab_data

def make_entry(d, args, igfold):
    entry = {k: d[k] for k in ['name', 'hcdr3', 'lcdr3']}
    entry['VH'], entry['VL'] = d['ab'].split('-')
    assert entry['VH'].count(entry['hcdr3']) == 1
    entry['context'] = entry['VH'].replace(entry['hcdr3'], '#' * len(entry['hcdr3']))
    fw1, fw2 = entry['context'].replace('#', ' ').split()
    entry['cdr'] = '0' * len(fw1) + '3' * len(entry['hcdr3']) + '0' * len(fw2)
    
    sequences = {
        "H": entry['VH'],
        "L": entry['VL'],
    }
    pred_pdb = "checkpoints/exp3-igfold-structs/" + entry['name'] + ".pdb" # save dir

    predicted_pdb = igfold.fold(
        pred_pdb, # Output PDB file
        sequences=sequences, # Antibody sequences
        do_refine=False, # Refine the antibody structure with PyRosetta
        do_renum=False, # Renumber predicted antibody structure (Chothia)
    )
    coords = np.array(predicted_pdb.coords[0, :len(entry['VH'])].cpu()) # (N, 5, 3)
    
    entry['coords'] = {
            "N": coords[:, 0],
            "CA":coords[:, 1],
            "C": coords[:, 2],
            "O": coords[:, 4],
    }
    entry['label'] = d['label']
    entry['seq'] = entry['VH'] if args.architecture == 'hierarchical' else entry['hcdr3']
    return entry


# Decode new sequences
def decode(model, ab, args, seq_vocab, tag_vocab, mask_idx):
    ab = data_curate(ab, seq_vocab, tag_vocab, mask_idx)
    batch = [ab] * args.batch_size
    model.eval()
    model.inference = True
    with torch.no_grad():
        batched_seq, batched_tag, batched_label, paratope, antibody = collater(batch, seq_vocab.pad())
        
        masked_tokens = batched_label.ne(seq_vocab.pad())
        sample_size = masked_tokens.int().sum()
        
        out = model(
            src_tokens=batched_seq, 
            tag_tokens=batched_tag,
            paratope=paratope,
            epitope=None,
            antibody=antibody,
            masked_tokens=masked_tokens,
            num_decode=1
            )[0]
        new_seqs = out.handle
            
    return new_seqs, out.ppl.exp()


def evaluate(model, predictor, evaluator, data, args, seq_vocab, tag_vocab, mask_idx):
    succ, tot = 0, 0
    sum_ppl, tot_aa = 0., 0.
    model.eval()
    with torch.no_grad():
        for ab in tqdm(data):
            # print(ab)
            new_seqs, ppl = decode(model, ab, args, seq_vocab, tag_vocab, mask_idx)
            tot = tot + len(new_seqs)
            prior_ppl = is_natural_seq(evaluator, ab, new_seqs)
            for new_cdr, ppl in zip(new_seqs, prior_ppl):
                if is_valid_seq(new_cdr) and ppl <= args.max_prior_ppl:
                    VH = ab['VH'].replace(ab['hcdr3'], new_cdr)
                    prob = predictor.predict(VH, ab['VL'])
                    succ = succ + prob
                else:
                    succ = succ + ab['score']  # not valid, improvement=0
            sum_ppl += ppl * len(new_seqs)
            tot_aa += len(new_seqs)
    return succ / tot, sum_ppl / tot_aa


# Glycoslation is bad
def is_valid_seq(cdr):
    if '#' in cdr: return False
    charge = [CHARGE[x] for x in cdr]
    if sum(charge) >= 3 or sum(charge) <= -3:
        return False
    for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if ('N' + a + 'T') in cdr:
            return False
        if ('N' + a + 'S') in cdr:
            return False
        if (a * 4) in cdr:
            return False
    return True


def is_natural_seq(evaluator, ab, cand_cdrs):
    with torch.no_grad():
        batch = []
        for cdr in cand_cdrs:
            ab = deepcopy(ab)
            ab['seq'] = ab['VH'].replace(ab['hcdr3'], cdr)
            batch.append(ab)

        hX, hS, hL, hmask = completize(batch)
        cand_ppl1 = evaluator[0].log_prob(hS, hL, hmask).ppl.exp()

        batch = []
        for cdr in cand_cdrs:
            ab = deepcopy(ab)
            ab['seq'] = cdr
            batch.append(ab)

        (hX, hS, hL, hmask), context = featurize(batch, context=True)
        cand_ppl2 = evaluator[1].log_prob(hS, hmask, context=context).ppl.exp()
        cand_ppl3 = evaluator[2].log_prob(hS, hmask, context=context).ppl.exp()

        cand_ppl = torch.maximum(cand_ppl1, torch.maximum(cand_ppl2, cand_ppl3))
    return cand_ppl.tolist()

def main(args):
    
    modelhub = AntibodyRobertaModel.from_pretrained(
        model_name_or_path=args.cktpath,
        inference=True,
        fix_bert_param=args.fix_bert_param,
    )
    optimizer = modelhub.optimizer
    model = modelhub.model
    seq_vocab = modelhub.task.source_dictionary
    tag_vocab = modelhub.task.tag_source_dictionary
    mask_idx = modelhub.task.mask_idx
    model.cuda()

    def print_parameter_number(net):
        net_name = type(net).__name__
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'{net_name} total parameters:{total_num}, trainable: {trainable_num}')
    # print_parameter_number(model.classification_heads)
    print_parameter_number(model)

    os.makedirs(args.save_dir, exist_ok=True)
    split_map = {}
    with open(args.cluster) as f:
        for line in f:
            cdr3, fold = line.strip("\r\n ").split()
            split_map[cdr3] = fold

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    predictor = CovidNeutralizationModel()
    
    optimizer.set_lr(args.lr)

    # Language model ensemble (ensure CDR naturalness) 
    evaluator = [None, None, None]
    eval_args = deepcopy(args)
    eval_args.hidden_size = 256
    eval_args.depth = 4
    eval_args.block_size = 8
    evaluator[0] = HierarchicalDecoder(eval_args).cuda()
    evaluator[0].eval()
    model_ckpt = torch.load("checkpoints/pretrained/ckpts/covid/hieratt.ckpt")[0]
    evaluator[0].load_state_dict(model_ckpt)

    eval_args = deepcopy(args)
    eval_args.hidden_size = 128
    eval_args.depth = 1
    evaluator[1] = Seq2Seq(eval_args).cuda()
    evaluator[1].eval()
    model_ckpt = torch.load("checkpoints/pretrained/ckpts/covid/lstm.ckpt")[0]
    evaluator[1].load_state_dict(model_ckpt)

    eval_args = deepcopy(args)
    eval_args.hidden_size = 256
    eval_args.depth = 3
    evaluator[2] = Decoder(eval_args, return_coords=False).cuda()
    evaluator[2].eval()
    model_ckpt = torch.load("checkpoints/pretrained/ckpts/covid/autoreg.ckpt")[0]
    evaluator[2].load_state_dict(model_ckpt)

    # data preparation
    igfold = IgFoldRunner()
    all_ab = [make_entry(d, args, igfold) for d in load_data() if d['hcdr3'] in d['ab'] and d['mask'] == 1]
    for entry in tqdm(all_ab):
        entry['score'] = predictor.predict(entry['VH'], entry['VL'])

    train_ab = [d for d in all_ab if split_map[d['hcdr3']] == 'train' and d['label'] == 1]
    val_ab = [d for d in all_ab if split_map[d['hcdr3']] == 'val' and d['label'] == 1]
    test_ab = [d for d in all_ab if split_map[d['hcdr3']] == 'test' and d['label'] == 1]
    print("train/val/test:", len(train_ab), len(val_ab), len(test_ab))

    train_data = {entry['name'] : [entry] for entry in train_ab}
    best_score, best_epoch = -10.0, 0
    cross_entropy = nn.CrossEntropyLoss(reduction="sum", ignore_index=seq_vocab.pad())
    for e in trange(args.epochs):
        # Decode new cdrs
        ab = random.choice(train_ab)
        name = ab['name']
        new_seqs, _ = decode(model, ab, args, seq_vocab, tag_vocab, mask_idx)

        prior_ppl = is_natural_seq(evaluator, ab, new_seqs)
        for new_cdr, ppl in zip(new_seqs, prior_ppl):
            if is_valid_seq(new_cdr) and ppl <= args.max_prior_ppl:
                entry = deepcopy(ab)
                entry['VH'] = entry['VH'].replace(entry['hcdr3'], new_cdr)
                prob = predictor.predict(entry['VH'], entry['VL'])
                #print(name, entry['hcdr3'], entry['score'], '-->', new_cdr, prob)
                if prob > entry['score']: # first round: about 24 in 200
                    entry['score'] = prob
                    entry['hcdr3'] = new_cdr
                    entry['seq'] = entry['VH'] if args.architecture == 'hierarchical' else entry['hcdr3']
                    train_data[name].append(entry)
    
        if name in train_data:
            dlist = sorted(train_data[name], key=lambda d:d['score'], reverse=True)
            train_data[name] = dlist[:args.topk]
        else:
            exit(0)

        # Train model
        model.train()
        model.inference = False
        optimizer.zero_grad()
        train_keys = sorted(train_data.keys())
        
        name = random.choice(train_keys)
        batch = train_data[name]
        name_samples = []
        for name_sample in batch:
            name_samples.append(data_curate(name_sample, seq_vocab, tag_vocab, mask_idx))

        batched_seq, batched_tag, batched_label, paratope, antibody = collater(name_samples, seq_vocab.pad())
        
        masked_tokens = batched_label.ne(seq_vocab.pad())
        sample_size = masked_tokens.int().sum()
        
        out, transformer_logits, _ = model(
            src_tokens=batched_seq, 
            tag_tokens=batched_tag,
            paratope=paratope,
            epitope=None,
            antibody=antibody,
            masked_tokens=masked_tokens
            )
        
        # compute encoder loss
        if out.nll == 0.: # bertonly ablation
            targets = batched_label
            if masked_tokens is not None:
                targets = targets[masked_tokens]
            loss = cross_entropy(
                transformer_logits.view(-1, transformer_logits.size(-1)),
                targets.view(-1) - 3, 
            ) / sample_size / math.log(2)
        else:
            loss = out.nll

        loss.backward()
        optimizer.step()

        if (e + 1) % args.valid_iter == 0:
            ckpt = (model.state_dict(), optimizer.state_dict())
            torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e+1}"))
            val_score, val_ppl = evaluate(model, predictor, evaluator, val_ab, args, seq_vocab, tag_vocab, mask_idx)
            test_score, test_ppl = evaluate(model, predictor, evaluator, test_ab, args, seq_vocab, tag_vocab, mask_idx)
            print(f'Epoch {e+1}: average valid neutralization score: {val_score:.3f}')
            print(f'             average valid ppl: {val_ppl:.3f}')
            print(f'Epoch {e+1}: average test neutralization score: {test_score:.3f}')
            print(f'             average test ppl: {test_ppl:.3f}')
            if val_score > best_score:
                best_epoch = e + 1
                best_score = val_score

    # best_epoch = 10000
    if best_epoch > 0:
        best_ckpt = os.path.join(args.save_dir, f"model.ckpt.{best_epoch}")
        model.load_state_dict(torch.load(best_ckpt)[0])

    test_score, test_ppl = evaluate(model, predictor, evaluator, test_ab, args, seq_vocab, tag_vocab, mask_idx)
    print(f'Test average neutralization score: {test_score:.3f}')
    print(f'             average test ppl: {test_ppl:.3f}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data and ckpts
    parser.add_argument("--cluster", type=str, default="dataset/exp3-covabdab/cdrh3_split.txt")
    parser.add_argument("--cktpath", type=str, default="checkpoints/0301-exp3-sabdab/checkpoint_best.pt")
    parser.add_argument("--save_dir", type=str, default='checkpoints/exp3-ckpts/temp')
    # training settings    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--valid_iter", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--fix_bert_param", action="store_true", default=False)
    # evaluate naturalness (and relevant refinegnn setting)  ===  fixed
    parser.add_argument("--load_model", type=str, default="checkpoints/pretrained/RefineGNN-rabd/model.best")
    parser.add_argument("--architecture", type=str, default="hierarchical")
    parser.add_argument("--cdr_type", type=str, default="3")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--k_neighbors", type=int, default=9)
    parser.add_argument("--update_freq", type=int, default=1)
    parser.add_argument("--augment_eps", type=float, default=3.0)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=21)
    parser.add_argument("--num_rbf", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_prior_ppl", type=int, default=10)
    parser.add_argument("--context", type=bool, default=True)
    
    args = parser.parse_args()
    
    assert os.path.exists(args.cktpath)
    
    main(args)
