# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging
import os

import fairseq
import numpy as np
import torch
import warnings


from fairseq import utils
from fairseq.data import (
    data_utils,
    Dictionary,
    SortDataset,
)
from fairseq.tasks import LegacyFairseqTask, register_task

from fairseq_models.data.abgen_dataset import AntibodyComplexDataset
from fairseq_models.data.abgen_dataset_noantigen import AntibodyOnlyDataset

logger = logging.getLogger(__name__)


@register_task('antibody_generation_task')
class AntibodyGenerationTask(LegacyFairseqTask):
    """
    Notice: till now, one of the MLM/PSSM Pred. is must needed for build & load the dataset


    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--sabdab-data', help="sabdab data path")
        parser.add_argument(
            '--cdr-type', 
            default='3', 
            choices=['1', '2', '3'],
            help="which part to predict"
        )
        parser.add_argument(
            '--L-target', 
            default=20,
            type=int,
            help="number of antigen residues to be considered as epitope"
        )

        parser.add_argument('--noantigen', action='store_true', default=False)
        

    def __init__(self, args, seq_dict, tag_dict):
        super().__init__(args)
        self.args = args
        self.seq_dict = seq_dict
        self.tag_dict = tag_dict
        self.seed = args.seed

        self.mask_idx = seq_dict.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        seq_dict = Dictionary.load(os.path.join('data', 'seq_dict.txt'))
        tag_dict = Dictionary.load(os.path.join('data', 'tag_dict.txt'))

        logger.info("[input] dictionary: {} types".format(len(seq_dict)))
        logger.info("[input] dictionary: {} types".format(len(tag_dict)))

        
        return cls(args, seq_dict, tag_dict)  # Done: needs refine to TAPE's tokenizer
        

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        ### downstream tasks
        if self.args.noantigen:
            dataset = AntibodyOnlyDataset(
                data_path=self.args.sabdab_data,
                split=split,
                seq_vocab=self.source_dictionary,
                tag_vocab=self.tag_source_dictionary,
                cdr_types=[*self.args.cdr_type],
                L_target=self.args.L_target,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                max_len=self.args.max_positions
            )
        else:
            dataset = AntibodyComplexDataset(
                data_path=self.args.sabdab_data,
                split=split,
                seq_vocab=self.source_dictionary,
                tag_vocab=self.tag_source_dictionary,
                cdr_types=[*self.args.cdr_type],
                L_target=self.args.L_target,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                max_len=self.args.max_positions
            )
        

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))
            
        self.datasets[split] = SortDataset(dataset, sort_order=[shuffle, dataset.prefix_len, dataset.sizes])
        
        return None  # return in advance

    def build_model(self, args):
 
        model = super().build_model(args)

        def inplace_relu(m):
            classname = m.__class__.__name__
            if classname.find('ReLU') != -1:
                m.inplace=True
        model.apply(inplace_relu)

        return model

    @property
    def source_dictionary(self):
        return self.seq_dict
    
    @property
    def tag_source_dictionary(self):
        return self.tag_dict

    @property
    def target_dictionary(self):
        return self.seq_dict