# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils
from fairseq.data import BaseWrapperDataset, LRUCacheDataset

# from fairseq_models.aa_piece_dataset import AAPieceDataset


class AntibodyMaskTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
        mask_multiple_length : repeat each mask index multiple times. Default
            value is 1.
        mask_stdev : standard deviation of masks distribution in case of
            multiple masking. Default value is 0.
    """

    @classmethod
    def apply_mask(cls, dataset1: torch.utils.data.Dataset, dataset2: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset1 = LRUCacheDataset(dataset1)
        dataset2 = LRUCacheDataset(dataset2)
        return (
            LRUCacheDataset(cls(dataset1, dataset2, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset1, dataset2, *args, **kwargs, return_masked_tokens=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        tag_dataset: torch.utils.data.Dataset,
        seq_vocab: Dictionary,
        tag_vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        # mask_aa_pieces:bool=False,
        # aa_piece_dataset:AAPieceDataset=None,
        mask_multiple_length: int = 1,
        mask_stdev: float = 0.0,
    ):
        assert 0.0 < mask_prob <= 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0
        assert mask_multiple_length >= 1
        assert mask_stdev >= 0.0

        self.dataset = dataset
        self.tag_dataset = tag_dataset
        self.seq_vocab = seq_vocab
        self.tag_vocab = tag_vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        # self.mask_aa_pieces=mask_aa_pieces
        # self.aa_piece_dataset=aa_piece_dataset

        self.mask_multiple_length = mask_multiple_length
        self.mask_stdev = mask_stdev

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.seq_vocab.count)
            else:
                weights = np.ones(len(self.seq_vocab))
            weights[: self.seq_vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        # if mask_aa_pieces:
        #     # self.aa_pieces=AAPieceFromUniprot21()
        #     self.aa_piece_dataset=aa_piece_dataset

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            tag = self.tag_dataset[index]
            cdr1_idx = self.tag_vocab.index('1')
            cdr2_idx = self.tag_vocab.index('2')
            cdr3_idx = self.tag_vocab.index('3')
            cdr_mask = torch.logical_or(
                torch.logical_or(tag==cdr1_idx, tag==cdr2_idx),
                tag==cdr3_idx
            )
            sz = cdr_mask.sum()
            

            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )


            # if self.mask_aa_pieces:
            #     # word_begins_mask = self.mask_whole_words.gather(0, item)
            #     # word_begins_idx = word_begins_mask.nonzero().view(-1)
            #     # sz = len(word_begins_idx)
            #     # words = np.split(word_begins_mask, word_begins_idx)[1:]
            #     # assert len(words) == sz
            #     # word_lens = list(map(len, words))

            #     # 对item做BPE粒度的mask，注意1个item只含1条序列，即sample-break-mode选项必须为eos
            #     # pieces = self.aa_pieces.encode_pieces(item)
            #     pieces = self.aa_piece_dataset[index].get_pieces()

            #     word_lens = list(map(len, pieces))
            #     word_lens=[1]+word_lens+[1]
            #     sz = len(word_lens)

            if self.mask_prob == 1.0:
                mask = np.full(sz, True)
                if self.return_masked_tokens:
                    # exit early if we're just returning the masked tokens
                    # (i.e., the targets for masked LM training)
                    # if self.mask_aa_pieces:
                    #     mask = np.repeat(mask, word_lens)
                        
                    new_item = np.full(len(item), self.pad_idx)
                    pre_mask = np.full(sz, self.pad_idx)
                    pre_mask[mask] = item[cdr_mask][torch.from_numpy(mask.astype(np.uint8)) == 1]
                    new_item[cdr_mask] = pre_mask

                    return torch.from_numpy(new_item)

                # decide unmasking and random replacement
                rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
                if rand_or_unmask_prob > 0.0:
                    rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                    if self.random_token_prob == 0.0:
                        unmask = rand_or_unmask
                        rand_mask = None
                    elif self.leave_unmasked_prob == 0.0:
                        unmask = None
                        rand_mask = rand_or_unmask
                    else:
                        unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                        decision = np.random.rand(sz) < unmask_prob
                        unmask = rand_or_unmask & decision
                        rand_mask = rand_or_unmask & (~decision)
                else:
                    unmask = rand_mask = None

                if unmask is not None:
                    mask = mask ^ unmask

                # if self.mask_aa_pieces:
                #     mask = np.repeat(mask, word_lens)

                new_item = np.copy(item)
                pre_mask = np.full(sz, self.pad_idx)
                pre_mask[mask] = self.mask_idx
                if rand_mask is not None:
                    num_rand = rand_mask.sum()
                    if num_rand > 0:
                        # if self.mask_aa_pieces:
                        #     rand_mask = np.repeat(rand_mask, word_lens)
                        #     num_rand = rand_mask.sum()

                        pre_mask[rand_mask] = np.random.choice(
                            len(self.seq_vocab),
                            num_rand,
                            p=self.weights,
                        )
                new_item[cdr_mask] = pre_mask

                # print("sentence: ", item)
                # print("tag: ", tag)
                # print("tag seq", self.tag_vocab.string(tag))
                # print("new sent: ", torch.from_numpy(new_item))
                # print('tag_vocab: ', self.tag_vocab)
                # exit(0)

                return torch.from_numpy(new_item)
                


            # decide elements to mask
            mask = np.full(sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz
                + np.random.rand()
            )

            # multiple masking as described in the vq-wav2vec paper (https://arxiv.org/abs/1910.05453)
            mask_idc = np.random.choice(sz, num_mask, replace=False)
            
            mask_idc = mask_idc[mask_idc < len(mask)]
            try:
                mask[mask_idc] = True
            except:  # something wrong
                print(
                    "Assigning mask indexes {} to mask {} failed!".format(
                        mask_idc, mask
                    )
                )
                raise

            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                # if self.mask_aa_pieces:
                #     mask = np.repeat(mask, word_lens)
                
                new_item = np.full(len(item), self.pad_idx)
                pre_mask = np.full(sz, self.pad_idx)
                pre_mask[mask] = item[cdr_mask][torch.from_numpy(mask.astype(np.uint8)) == 1]
                new_item[cdr_mask] = pre_mask

                return torch.from_numpy(new_item)

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            # if self.mask_aa_pieces:
            #     mask = np.repeat(mask, word_lens)

            new_item = np.copy(item)
            pre_mask = np.full(sz, self.pad_idx)
            pre_mask[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    # if self.mask_aa_pieces:
                    #     rand_mask = np.repeat(rand_mask, word_lens)
                    #     num_rand = rand_mask.sum()

                    pre_mask[rand_mask] = np.random.choice(
                        len(self.seq_vocab),
                        num_rand,
                        p=self.weights,
                    )
            new_item[cdr_mask] = pre_mask

            # print("sentence: ", item)
            # print("tag: ", tag)
            # print("tag seq", self.tag_vocab.string(tag))
            # print("new sent: ", torch.from_numpy(new_item))
            # print('tag_vocab: ', self.tag_vocab)
            # exit(0)

            return torch.from_numpy(new_item)
