# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from fairseq import utils
from fairseq import metrics
from fairseq.criterions import LegacyFairseqCriterion, register_criterion

from fairseq_models.modules.utils import compute_rmsd

@register_criterion("antibody_generation_loss")
class AntibodyGenerationLoss(LegacyFairseqCriterion):
    """ 
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task, tpu=False):
        super().__init__(args, task)
        self.tpu = tpu
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.padding_idx)
        self.update_num = 0
        self.args = args

    @staticmethod
    def add_args(parser):
        parser.add_argument('--loss-scale-enc', type=float, default=1.,
                            help='loss scale of encoder loss')
        parser.add_argument('--loss-scale-dec-sloss', type=float, default=1.,
                            help='loss scale of decoder sloss')
        parser.add_argument('--loss-scale-dec-xloss', type=float, default=1.,
                            help='loss scale of decoder xloss')

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        batched_seq, batched_tag, batched_label, paratope, epitope, antibody = sample

        masked_tokens = batched_label.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()
        
        out, transformer_logits, _ = model(
            src_tokens=batched_seq, 
            tag_tokens=batched_tag,
            paratope=paratope,
            epitope=epitope,
            antibody=antibody,
            masked_tokens=masked_tokens,
        )

        # compute encoder loss
        targets = batched_label
        if masked_tokens is not None:
            targets = targets[masked_tokens]
        if transformer_logits is not None:
            sloss = self.cross_entropy(
                transformer_logits.view(-1, transformer_logits.size(-1)),
                targets.view(-1) - 3, 
            ) / sample_size / math.log(2)
        else:
            sloss = 0

        # compute RMSD score
        bind_X, bind_S, _, _ = paratope
        bind_mask = bind_S > 0
        if self.args.noantigen: # noantigen version has no side chains
            ret = torch.zeros((bind_X.size(0), bind_X.size(1), 4, 3), device=batched_seq.device)
        else:
            ret = torch.zeros((bind_X.size(0), bind_X.size(1), 14, 3), device=batched_seq.device)
        cnt = 0
        for i, v in enumerate([sum(mask) for mask in masked_tokens]):
            ret[i, :v] = out.bind_X[cnt:cnt+v]
            cnt += v
        rmsd = compute_rmsd(
                ret[:, :, 1], bind_X[:, :, 1], bind_mask
            )
        rmsd = sum(rmsd) / len(rmsd)

        # loss sum
        loss =  self.args.loss_scale_enc * sloss + \
                self.args.loss_scale_dec_xloss * out.xloss + \
                self.args.loss_scale_dec_sloss * out.nll
        
        logging_output = {
            "encoder_sloss": sloss.item(),
            "decoder_sloss": out.nll,
            "decoder_xloss": out.xloss,
            "sample_size": sample_size.item(),
            "loss": loss.item(),
            "rmsd": rmsd.item()
        }
        return loss, sample_size.item(), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        encoder_sloss = sum(log.get("encoder_sloss", 0) for log in logging_outputs)
        decoder_sloss = sum(log.get("decoder_sloss", 0) for log in logging_outputs)
        decoder_xloss = sum(log.get("decoder_xloss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        rmsd = sum(log.get("rmsd", 0) for log in logging_outputs)
        
        metrics.log_scalar(
            "encoder_sloss", encoder_sloss, sample_size, round=3
        )
        metrics.log_derived(
            "encoder_ppl", lambda meters: utils.get_perplexity(meters["encoder_sloss"].avg)
        )
        metrics.log_scalar(
            "decoder_sloss", decoder_sloss, sample_size, round=3
        )
        metrics.log_derived(
            "decoder_ppl", lambda meters: utils.get_perplexity(meters["decoder_sloss"].avg)
        )
        metrics.log_scalar(
            "decoder_xloss", decoder_xloss, sample_size, round=3
        )
        metrics.log_scalar(
            "loss", loss, sample_size, round=3
        )
        metrics.log_scalar(
            "rmsd", rmsd, sample_size, round=3
        )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
