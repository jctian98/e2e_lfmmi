# Author: Jinchuan Tian; tianjinchuan@stu.pku.edu.cn ; tyriontian@tencent.com
# Neural Transducer model for code-switch (bilingual problem)

from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import asdict

import torch
import chainer
import numpy
import math
import logging
from itertools import groupby
from typing import Tuple, List
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transducer.arguments import (
    add_encoder_general_arguments,  # noqa: H301
    add_custom_encoder_arguments,  # noqa: H301
    add_decoder_general_arguments,  # noqa: H301
    add_rnn_decoder_arguments,  # noqa: H301
    add_custom_training_arguments,  # noqa: H301
    add_transducer_arguments,  # noqa: H301
    add_auxiliary_task_arguments,  # noqa: H301
    add_att_scorer_arguments,
    add_transducer_code_switch_arguments,
)
from espnet.nets.pytorch_backend.transducer.custom_encoder import CustomEncoder
from espnet.nets.pytorch_backend.transducer.error_calculator import ErrorCalculator
from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoder import DecoderRNNT
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.transducer_decoder_interface import Hypothesis


class Reporter(chainer.Chain):
    """A chainer reporter wrapper for transducer models."""

    def report(
        self,
        loss,
        loss_trans,
        loss_ctc,
        loss_lm,
        loss_aux_trans,
        loss_aux_symm_kl,
        loss_mbr,
        loss_mmi,
        loss_att,
        loss_lang,
        cer,
        wer,
    ):
        """Instantiate reporter attributes."""
        chainer.reporter.report({"loss": loss}, self)
        chainer.reporter.report({"loss_trans": loss_trans}, self)
        chainer.reporter.report({"loss_ctc": loss_ctc}, self)
        chainer.reporter.report({"loss_lm": loss_lm}, self)
        chainer.reporter.report({"loss_aux_trans": loss_aux_trans}, self)
        chainer.reporter.report({"loss_aux_symm_kl": loss_aux_symm_kl}, self)
        chainer.reporter.report({"loss_mbr": loss_mbr}, self)
        chainer.reporter.report({"loss_mmi": loss_mmi}, self)
        chainer.reporter.report({"loss_att": loss_att}, self)
        chainer.reporter.report({"loss_lang": loss_lang}, self)
        chainer.reporter.report({"cer": cer}, self)
        chainer.reporter.report({"wer": wer}, self)

        logging.info("loss:" + str(loss))

class E2E(ASRInterface, torch.nn.Module):
    """E2E module for transducer models.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (Namespace): argument Namespace containing options
        ignore_id (int): padding symbol id
        blank_id (int): blank symbol id

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments for transducer model."""
        E2E.encoder_add_general_arguments(parser)
        E2E.encoder_add_custom_arguments(parser)

        E2E.decoder_add_general_arguments(parser)
        E2E.decoder_add_rnn_arguments(parser)

        E2E.training_add_custom_arguments(parser)
        E2E.transducer_add_arguments(parser)
        E2E.auxiliary_task_add_arguments(parser)

        E2E.transducer_add_code_switch_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_general_arguments(parser):
        """Add general arguments for encoder."""
        group = parser.add_argument_group("Encoder general arguments")
        group = add_encoder_general_arguments(group)

        return parser

    @staticmethod
    def encoder_add_custom_arguments(parser):
        """Add arguments for Custom encoder."""
        group = parser.add_argument_group("Custom encoder arguments")
        group = add_custom_encoder_arguments(group)

        return parser

    @staticmethod
    def decoder_add_general_arguments(parser):
        """Add general arguments for decoder."""
        group = parser.add_argument_group("Decoder general arguments")
        group = add_decoder_general_arguments(group)

        return parser

    @staticmethod
    def decoder_add_rnn_arguments(parser):
        """Add arguments for RNN decoder."""
        group = parser.add_argument_group("RNN decoder arguments")
        group = add_rnn_decoder_arguments(group)

        return parser

    @staticmethod
    def training_add_custom_arguments(parser):
        """Add arguments for Custom architecture training."""
        group = parser.add_argument_group("Training arguments for custom archictecture")
        group = add_custom_training_arguments(group)

        return parser

    @staticmethod
    def transducer_add_arguments(parser):
        """Add arguments for transducer model."""
        group = parser.add_argument_group("Transducer model arguments")
        group = add_transducer_arguments(group)

        return parser

    @staticmethod
    def transducer_add_code_switch_arguments(parser):
        """Add arguments for transducer model."""
        group = parser.add_argument_group("Transducer code switch arguments")
        group = add_transducer_code_switch_arguments(group)
        
        return parser

    @staticmethod
    def auxiliary_task_add_arguments(parser):
        """Add arguments for auxiliary task."""
        group = parser.add_argument_group("Auxiliary task arguments")
        group = add_auxiliary_task_arguments(group)

        return parser

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        if self.shared_encoder:
            return self.shared_encoder.conv_subsampling_factor * int(
                numpy.prod(self.subsample)
        )
        else:
            return self.chn_encoder.conv_subsampling_factor * int(
                numpy.prod(self.subsample)
        )

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0, training=True):
        """Construct an E2E object for transducer model."""
        """ By default we only adopt Custom Encoder and RNN Decoder """
        torch.nn.Module.__init__(self)

        args = fill_missing_args(args, self.add_arguments)

        ### Commom transducer configs ###
        self.is_rnnt = True # legacy
        self.transducer_weight = args.transducer_weight
        self.etype = "custom" # legacy
        self.dtype = "rnn"

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        ### code-switch parameters ###
        self.chn_id = odim
        self.eng_id = odim + 1
        self.cs_id = odim + 2
        self.chn_start = args.cs_chn_start
        self.eng_start = args.cs_eng_start
        
        self.use_adversial_examples = args.cs_use_adversial_examples
        self.is_ctc_decoder = args.cs_is_ctc_decoder
        self.is_pretrain = args.cs_is_pretrain
        self.use_decoder_expert = args.cs_decoder_expert       
 
        self.aux_ctc_weight = args.aux_ctc_weight
        self.lang_weight = args.cs_lang_weight

        self.space = args.sym_space
        self.blank = args.sym_blank
        self.odim = odim
        self.reporter = Reporter()
        self.error_calculator = None

        ### Modules ###
        if args.cs_share_encoder:
            self.shared_encoder = CustomEncoder(
                idim=idim,
                enc_arch=args.enc_block_arch,
                input_layer=args.custom_enc_input_layer,
                repeat_block=args.cs_share_encoder_layers,
                self_attn_type=args.custom_enc_self_attn_type,
                positional_encoding_type=args.custom_enc_positional_encoding_type,
                positionwise_activation_type=args.custom_enc_pw_activation_type,
                conv_mod_activation_type=args.custom_enc_conv_mod_activation_type,
            )
        else:
            self.shared_encoder = None
       
        # When use shared_encoder, there is no cnn layers in chn/eng encoder 
        enc_params = dict(
            idim=idim if not args.cs_share_encoder else self.shared_encoder.enc_out,
            enc_arch=args.enc_block_arch,
            input_layer=args.custom_enc_input_layer if not args.cs_share_encoder else "null",
            repeat_block=args.enc_block_repeat,
            self_attn_type=args.custom_enc_self_attn_type,
            positional_encoding_type=args.custom_enc_positional_encoding_type,
            positionwise_activation_type=args.custom_enc_pw_activation_type,
            conv_mod_activation_type=args.custom_enc_conv_mod_activation_type,
        )
        # Make sure identical settings
        self.chn_encoder = CustomEncoder(**enc_params)
        self.eng_encoder = CustomEncoder(**enc_params)

        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        encoder_out = self.chn_encoder.enc_out

        self.most_dom_list = args.enc_block_arch[:]
        self.most_dom_dim = sorted(
            Counter(
                d["d_hidden"] for d in self.most_dom_list if "d_hidden" in d
            ).most_common(),
            key=lambda x: x[0],
            reverse=True,
        )[0][0]
           

        dec_param = (
                odim,
                args.dtype,
                args.dlayers,
                args.dunits,
                blank_id,
                args.dec_embed_dim,
                args.dropout_rate_decoder,
                args.dropout_rate_embed_decoder,
        )
        if self.use_decoder_expert:
            raise NotImplementedError
        else:
            self.dec = DecoderRNNT(*dec_param) 
 
        decoder_out = args.dunits

        self.joint_network = JointNetwork(
            odim, encoder_out, decoder_out, args.joint_dim, args.joint_activation_type
        )

        if self.lang_weight > 0.0:
            self.lang_classifer = torch.nn.Sequential(
                                        torch.nn.Linear(encoder_out, 2 * encoder_out),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(2 * encoder_out, 3),
                                      )
 
        self.default_parameters(args)

        ### Criterion ###
        self.criterion = TransLoss(args.trans_type, self.blank_id)
        self.aux_ctc = ctc_for(
                Namespace(
                    num_encs=1,
                    eprojs=encoder_out,
                    dropout_rate=args.aux_ctc_dropout_rate,
                    ctc_type="warpctc",
                ),
                odim,
                reduce=False,
        )
        self.decoder_ctc = ctc_for(
                Namespace(
                    num_encs=1,
                    eprojs=encoder_out,
                    dropout_rate=args.aux_ctc_dropout_rate,
                     ctc_type="warpctc",
                ),
                odim,
                reduce=False,
        )
        self.lang_cls_criterion = torch.nn.CrossEntropyLoss()

        self.loss = None
        self.rnnlm = None
        self.lms = {} # ngram LMs. Set externally during decoding

    def default_parameters(self, args):
        """Initialize/reset parameters for transducer.

        Args:
            args (Namespace): argument Namespace containing options

        """
        initializer(self, args)

    ### Training Implementation  ###
    def forward(self, xs_pad, ilens, ys_pad, texts, xs_pad_orig):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): transducer loss value
        """
        # 0. process labels
        ys_pad, cls_ids = ys_pad[:, 1:], ys_pad[:, 0].squeeze(0)

        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

        if self.shared_encoder:
            hs_pad, hs_mask = self.shared_encoder(xs_pad, src_mask,
                                                  return_as_intermidiate=True)
        else:
            hs_pad, hs_mask = xs_pad, src_mask

        chn_hs_pad, chn_hs_mask = self.chn_encoder(hs_pad, hs_mask)
        eng_hs_pad, eng_hs_mask = self.eng_encoder(hs_pad, hs_mask)

        hs_pad, hs_mask = self.combine_fn(chn_hs_pad, eng_hs_pad,
                                          chn_hs_mask, eng_hs_mask)

        # 2. Decoder loss: either RNNT or CTC
        if not self.is_pretrain:
            if not self.is_ctc_decoder:    
                ys_in_pad, ys_out_pad, target, pred_len, target_len = \
                    prepare_loss_inputs(ys_pad, hs_mask
                )
        
                pred_pad = self.dec(hs_pad, ys_in_pad)
    
                z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))
                loss_dec = self.criterion(z, target, pred_len, target_len)
            else:
                hlen = torch.IntTensor([h.size(1) for h in hs_mask]).to(hs_mask.device)
                loss_dec = self.decoder_ctc(hs_pad, hlen, ys_pad, texts).sum()
        else:
            loss_dec = 0.0

        # 3. auxiliary CTC
        if self.aux_ctc_weight > 0.0:
            chn_ys_pad, eng_ys_pad = self.monolingual_mask(ys_pad)
            # print(chn_ys_pad, eng_ys_pad)
            hlen = torch.IntTensor([h.size(1) for h in chn_hs_mask]).to(chn_hs_mask.device)
            loss_ctc_chn = self.aux_ctc(chn_hs_pad, hlen, chn_ys_pad, texts)
            loss_ctc_eng = self.aux_ctc(eng_hs_pad, hlen, eng_ys_pad, texts)
        
            # In fine-tuning we must compute two ctc loss for each utt
            if self.use_adversial_examples:
                loss_ctc = (loss_ctc_chn + loss_ctc_eng).sum() / 2
            else:
                chn_indices = torch.nonzero(cls_ids != self.eng_id).squeeze(1)
                eng_indices = torch.nonzero(cls_ids != self.chn_id).squeeze(1)
                loss_ctc = loss_ctc_chn[chn_indices].sum() + \
                           loss_ctc_eng[eng_indices].sum()
        else:
            loss_ctc = 0.0

        # 4. language prediction loss
        if self.lang_weight > 0.0:
            loss_lang = self.lang_cls_criterion(
                        self.lang_classifer(hs_pad.mean(dim=1)),
                        cls_ids - self.chn_id
                        )
        else:
            loss_lang = 0.0

        # 5. aggregate loss and report
        loss = loss_dec + \
               loss_ctc  * self.aux_ctc_weight + \
               loss_lang * self.lang_weight
 
        self.loss = loss
        loss_data = float(loss)

        # Some reprot keys are not used here
        if not math.isnan(loss_data):
            self.reporter.report(
                loss_data,
                float(loss_dec),
                float(loss_ctc),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                float(loss_lang),
                0.0,
                0.0,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    # You may want to revise this function to combine encoder_output differently
    def combine_fn(self, chn_hs_pad, eng_hs_pad, chn_hs_mask, eng_hs_mask):
        return chn_hs_pad + eng_hs_pad, chn_hs_mask

    def monolingual_mask(self, ys_pad):
        # <chn> 2 ; <eng> 3
        ys_pad_chn = torch.where(torch.logical_and(
            ys_pad >= self.eng_start, ys_pad < self.odim),
            3, ys_pad)

        ys_pad_eng = torch.where(torch.logical_and(
            ys_pad >= self.chn_start, ys_pad < self.eng_start),
            2, ys_pad)

        return ys_pad_chn, ys_pad_eng

    ### Decoding Implementation ###
    def encoder_forward(self, x):
         # Inference all
        self.eval()
        device = next(self.parameters()).device
        x = torch.Tensor(x).to(device).unsqueeze(0)

        if self.shared_encoder:
            hs, _ = self.shared_encoder(x, None, return_as_intermidiate=True)
        else:
            hs = x

        chn_hs, _ = self.chn_encoder(hs, None)
        eng_hs, _ = self.eng_encoder(hs, None)

        hs, _ = self.combine_fn(chn_hs, eng_hs, None, None)

        # temporary code:
        if hasattr(self, "lang_classifer"):
            pred = torch.argmax(self.lang_classifer(hs.mean(dim=1))).item()
            print("language classification results: ", pred, flush=True)

        return hs, chn_hs, eng_hs

    def recognize(self, x, beam_search=None, decode_feature="combine"):
        hs, chn_hs, eng_hs = self.encoder_forward(x)
        if decode_feature == "combine":
            feature = hs
        elif decode_feature == "chn":
            feature = chn_hs
        elif decode_feature == "eng":
            feature = eng_hs
        else:
            raise NotImplementedError

        nbest_hyps = beam_search(feature)
        return [asdict(n) for n in nbest_hyps] 

    # legacy, not used  
    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, texts, xs_pad_orig):
        """E2E attention calculation.

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax)

        Returns:
            ret (ndarray): attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).

        """
        self.eval()

        if "custom" not in self.etype and "custom" not in self.dtype:
            return []
        else:
            with torch.no_grad():
                self.forward(xs_pad, ilens, ys_pad, texts, xs_pad_orig)

            ret = dict()
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) or isinstance(
                    m, RelPositionMultiHeadedAttention
                ):
                    ret[name] = m.attn.cpu().numpy()

        self.train()

        return ret
