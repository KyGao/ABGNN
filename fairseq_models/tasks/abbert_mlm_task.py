# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task

from fairseq_models.tasks.abbert_mask_tokens_dataset import AntibodyMaskTokensDataset

logger = logging.getLogger(__name__)


@register_task("antibody_masked_lm")
class AntibodyMaskedLMTask(LegacyFairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # data: seq data-bin
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--tag-data", 
            help='name of the tag data'
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            default=False,
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )

        # parser.add_argument(
        #     "--mask-aa-pieces",
        #     default=False,
        #     action="store_true",
        #     help="mask whole A.A. pieces",
        # )

        parser.add_argument(
            "--mask-multiple-length",
            default=1,
            type=int,
            help="repeat the mask indices multiple times",
        )
        parser.add_argument(
            "--mask-stdev", default=0.0, type=float, help="stdev of the mask length"
        )
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )

        # parser.add_argument(
        #     "--sentencepiece-model-path",
        #     type=str,
        #     default=None,
        #     help="path to the used sentencepiece tokenizer",
        # )


    def __init__(self, args, seq_dict, tag_dict):
        super().__init__(args)
        self.seq_dict = seq_dict
        self.tag_dict = tag_dict
        self.seed = args.seed

        # add mask token
        self.mask_idx = seq_dict.add_symbol("<mask>")
        # self.mask_idx = tag_dict.add_symbol('<mask>')

    @classmethod
    def setup_task(cls, args, **kwargs):
        seq_dict = Dictionary.load(os.path.join(args.data, 'dict.txt'))
        tag_dict = Dictionary.load(os.path.join(args.tag_data, 'dict.txt'))

        logger.info("[input] dictionary: {} types".format(len(seq_dict)))
        logger.info("[input] dictionary: {} types".format(len(tag_dict)))

        # AAPieceFromUniprot21.set_model_path(args.sentencepiece_model_path)
        return cls(args, seq_dict, tag_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
        def curate_dataset(dataset_path, dictionary):
            paths = utils.split_paths(dataset_path)
            assert len(paths) > 0
            data_path = paths[(epoch - 1) % len(paths)]
            split_path = os.path.join(data_path, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )


            dataset = maybe_shorten_dataset(
                dataset,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                self.args.tokens_per_sample,
                self.args.seed,
            )

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=dictionary.pad(),
                eos=dictionary.eos(),
                break_mode=self.args.sample_break_mode,
            )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, dictionary.bos())

            return dataset

        seq_dataset = curate_dataset(self.args.data, self.source_dictionary)
        tag_dataset = curate_dataset(self.args.tag_data, self.tag_source_dictionary)
        # print(max([len(seq_dataset[i]) for i in range(900)]))
        # print(tag_dataset[5])
        # exit(0)

        # aa_piece_dataset=AAPieceDataset(dataset,enabled=self.args.mask_aa_pieces)

        # homology_dataset=HomologyDataset(dataset,enabled=self.args.contrastive_learning,negative_sample=(self.args.criterion=='masked_lm_with_msa'))

        # create masked input and targets

        src_dataset, tgt_dataset = AntibodyMaskTokensDataset.apply_mask(
            seq_dataset,
            tag_dataset,
            self.source_dictionary,
            self.tag_source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            # random_token_prob=0,     ###   self.args.random_token_prob,   不随机替换token, for BPE training
            freq_weighted_replacement=self.args.freq_weighted_replacement,

            # mask_aa_pieces=self.args.mask_aa_pieces,
            # aa_piece_dataset=aa_piece_dataset,

            mask_multiple_length=self.args.mask_multiple_length,
            mask_stdev=self.args.mask_stdev,
        )

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input0": {
                        "src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "net_input1": {
                        "src_tokens": RightPadDataset(
                            tag_dataset,
                            pad_idx=self.tag_source_dictionary.pad(),
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "target": RightPadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset=TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            )

        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())

        # aa_piece_dataset = AAPieceDataset(src_dataset,enabled=self.args.mask_aa_pieces)

        src_dataset = RightPadDataset(
            src_dataset,
            pad_idx=self.source_dictionary.pad(),
        )


        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),

                    # "aa_piece_handlers": aa_piece_dataset,
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.seq_dict
    
    @property
    def tag_source_dictionary(self):
        return self.tag_dict

    @property
    def target_dictionary(self):
        return self.seq_dict


