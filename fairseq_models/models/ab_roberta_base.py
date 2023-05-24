# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq_models.modules.ab_transformer_sentence_encoder import AntibodyTransformerSentenceEncoder
from fairseq_models.models.ab_decoder import FullshotRefineDecoder
from fairseq_models.models.ab_decoder_ablation import FullshotRefineDecoderStack, BertonlyDecoder
from fairseq_models.models.ab_decoder_noantigen import NoAntigenFullshotRefineDecoder
from fairseq_models.data.ab_dictionary import ALPHABET, ALPHABET_FULL, TAG_FULL

from fairseq.data import Dictionary


logger = logging.getLogger(__name__)


class PrefixEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.prefix_projection = args.prefix_projection
        prefix_hidden_size = args.encoder_embed_dim
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(args.pre_seq_len, args.encoder_embed_dim)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(args.encoder_embed_dim, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, args.encoder_layers * 2 * args.encoder_embed_dim)
            )
        else:
            self.embedding = torch.nn.Embedding(args.pre_seq_len, args.encoder_layers * 2 * args.encoder_embed_dim)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
            # print(self.trans[0].weight)
            # exit(0)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


@register_model("antibody_roberta")
class AntibodyRobertaModel(FairseqEncoderModel):
    def __init__(self, args, encoder, dictionary, inference=False):
        super().__init__(encoder)
        self.args = args
        self.dictionary = dictionary
        self.inference=inference
        
        if self.args.finetune:
            if self.args.finetune_bert_scheme == 'prefix_tuning' and self.args.pre_seq_len > 0:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.pre_seq_len = args.pre_seq_len
                self.n_layer = args.encoder_layers
                self.n_head = args.encoder_attention_heads
                self.n_embd = args.encoder_embed_dim // args.encoder_attention_heads
                self.prefix_tokens = torch.arange(self.pre_seq_len).long()
                self.prefix_encoder = PrefixEncoder(args)
                # print(self.n_layer)
            elif self.args.finetune_bert_scheme == 'fixed':
                for param in self.encoder.parameters():
                    param.requires_grad = False

            if self.args.bertonly:
                self.structure_decoder = BertonlyDecoder(self.args)
            elif self.args.noantigen:
                self.structure_decoder = NoAntigenFullshotRefineDecoder(self.args)
            elif self.args.stack:
                self.structure_decoder = FullshotRefineDecoderStack(self.args)
            else:
                self.structure_decoder = FullshotRefineDecoder(self.args)
    
        # We follow BERT's random weight initialization
        self.apply(init_bert_params)




    def print_parameter_number(self, net):
        net_name = type(net).__name__
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'{net_name} total parameters:{total_num}, trainable: {trainable_num}')

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        # args for Hierarchical Equivariant Refinement Network (HERN)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--k_neighbors', type=int, default=9)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--clash_step', type=int, default=10)
        parser.add_argument('--num_rbf', type=int, default=16)
        parser.add_argument('--hierarchical', type=bool, default=True)
        # new addedargs
        parser.add_argument('--finetune-bert-scheme', type=str, default='fixed', choices=['prefix_tuning', 'fixed', 'all_tuning'])
        parser.add_argument('--pre-seq-len', type=int, default=1)
        parser.add_argument('--prefix-projection', type=bool, default=True)

        parser.add_argument('--refine-iteration', type=int, default=5)

        parser.add_argument('--block_size', type=int, default=8)

        parser.add_argument('--use-no-pretrain', type=bool, default=False)

        # finetune or pretrain
        parser.add_argument('--finetune', action='store_true', default=False)
        parser.add_argument('--stack', action='store_true', default=False)
        parser.add_argument('--bertonly', action='store_true', default=False)

    @classmethod
    def build_model(cls, args, task, inference=False):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary, task.tag_source_dictionary)
        return cls(args, encoder, task.source_dictionary, inference)

    def get_prompt(self, batch_size, data_device):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(data_device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # (key, value) n_layer x batch_size x n_head x pre_seq_len x n_embd
        # n_layer * [2, bsz, head, len, dim//head]
        return past_key_values


    def forward(
        self,
        src_tokens,
        tag_tokens,
        paratope=None,
        epitope=None,
        antibody=None,
        features_only=False, 
        return_all_hiddens=False,
        masked_tokens=None,
        num_decode=1,
        **kwargs
    ):
        if self.args.finetune:
            batch_size = src_tokens.shape[0]
            if self.args.finetune_bert_scheme == 'prefix_tuning':
                past_key_values = self.get_prompt(batch_size=batch_size, data_device=src_tokens.device)
            else:
                past_key_values = None
            enc_out, pt_emb, extra = self.encoder(        
                src_tokens=src_tokens,
                tag_tokens=tag_tokens,
                features_only=features_only,
                return_all_hiddens=return_all_hiddens,
                past_key_values=past_key_values, 
                masked_tokens=masked_tokens,
                **kwargs
            )
            # for ablation
            # extra = None
            # enc_out = torch.zeros((masked_tokens.sum(), 21), device=masked_tokens.device)
            # pt_emb = torch.zeros((masked_tokens.sum(), 768), device=masked_tokens.device)
            if self.inference:
                decoder_out = self.structure_decoder.generate(
                    enc_out, paratope, epitope, antibody, pt_emb, masked_tokens, num_decode
                )
            else:
                decoder_out = self.structure_decoder( # feed into the generation model
                    enc_out, paratope, epitope, antibody, pt_emb, masked_tokens
                )
            
            return decoder_out, enc_out, extra
        
        else:
            x, _, extra = self.encoder(src_tokens, tag_tokens, features_only, return_all_hiddens, masked_tokens=masked_tokens, **kwargs)
            return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
    

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != "" else ""

        super().upgrade_state_dict_named(state_dict, name)

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "structure_decoder"):
            cur_state = self.structure_decoder.state_dict()
            for k, v in cur_state.items():
                if prefix + "structure_decoder." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "structure_decoder." + k)
                    state_dict[prefix + "structure_decoder." + k] = v
            
        if self.args.finetune_bert_scheme == 'prefix_tuning' and self.args.pre_seq_len > 0:
            cur_state = self.prefix_encoder.state_dict()
            for k, v in cur_state.items():
                if prefix + "prefix_encoder." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "prefix_encoder." + k)
                    state_dict[prefix + "prefix_encoder." + k] = v

    @property
    def supported_targets(self):
        return {"self"}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        inference=False,
        fix_bert_param=False,
        **kwargs
    ):
        from fairseq.checkpoint_utils import load_checkpoint_to_cpu
        import os
        
        state = load_checkpoint_to_cpu(model_name_or_path, arg_overrides=None)
        
        args = state["args"]
        from fairseq import models, quantization_utils

        class TaskConfig:
            def __init__(self):
                self.source_dictionary = ALPHABET_FULL
                self.mask_idx = self.source_dictionary.add_symbol("<mask>")
                self.tag_source_dictionary =  TAG_FULL
        task = TaskConfig()
        model = cls.build_model(args, task, inference)
        
        # model.add_state_dict_named(state['model'], "")
        model.load_state_dict(state["model"], strict=True, args=args)
        
        from itertools import chain
        from fairseq.optim.adam import FairseqAdam
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(model.parameters()),
            )
        )
        optimizer = FairseqAdam(args, params)
        
        last_optim_state = state["last_optimizer_state"]
        optimizer.load_state_dict(last_optim_state)
        if fix_bert_param:
            for param in model.prefix_encoder.parameters():
                param.requires_grad = False

        cls.upgrade_args(args)
        
        logger.info(args)
        return HubInterface(args, model, task, optimizer)

class HubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, args, model, task, optimizer):
        super().__init__()
        self.args = args
        self.model = model
        self.task = task
        self.optimizer = optimizer


        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device



class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None, finetune=False):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        self.finetune = finetune


        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        
        if self.finetune:
            x = torch.cat((torch.zeros((x.shape[0], 1), device=x.device), x[:, 4:24]), dim=1)
        
        return x


class RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary, tag_dict):
        super().__init__(dictionary)
        self.args = args

        tt = torch.tensor([i for i in range(len(dictionary))])
        print('dictionary string: ', dictionary.string(tt))

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        self.sentence_encoder = AntibodyTransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            segments_vocab=tag_dict,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
        )
        args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)

        self.lm_head = RobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            ),
            finetune=args.finetune
        )

    def forward(
        self,
        src_tokens,
        tag_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        past_key_values=None,
        **unused
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, tag_tokens, return_all_hiddens=return_all_hiddens, past_key_values=past_key_values,
        )

        if self.args.finetune:
            pt_emb = x.clone().detach()
            if masked_tokens is not None:
                pt_emb = pt_emb[masked_tokens, :]
        else:
            pt_emb = None
        
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)

        return x, pt_emb, extra

    def extract_features(self, src_tokens, tag_tokens, return_all_hiddens=False, past_key_values=None, **kwargs):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            segment_labels=tag_tokens,
            last_state_only=not return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
            past_key_values=past_key_values,
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {"inner_states": inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture("antibody_roberta", "antibody_roberta")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.spectral_norm_classification_head = getattr(
        args, "spectral_nrom_classification_head", False
    )


@register_model_architecture("antibody_roberta", "antibody_roberta_base")
def roberta_base_architecture(args):
    base_architecture(args)

