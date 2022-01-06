"""Transducer speech recognition model (pytorch)."""

from argparse import Namespace
from collections import Counter
from dataclasses import asdict
from functools import partial
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import logging
import math
import numpy
import functools
import chainer
import torch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transducer.arguments import (
    add_encoder_general_arguments,  # noqa: H301
    add_rnn_encoder_arguments,  # noqa: H301
    add_custom_encoder_arguments,  # noqa: H301
    add_decoder_general_arguments,  # noqa: H301
    add_rnn_decoder_arguments,  # noqa: H301
    add_custom_decoder_arguments,  # noqa: H301
    add_custom_training_arguments,  # noqa: H301
    add_transducer_arguments,  # noqa: H301
    add_auxiliary_task_arguments,  # noqa: H301
    add_att_scorer_arguments,
)
from espnet.nets.pytorch_backend.transducer.auxiliary_task import AuxiliaryTask
from espnet.nets.pytorch_backend.transducer.custom_decoder import CustomDecoder
from espnet.nets.pytorch_backend.transducer.custom_encoder import CustomEncoder
from espnet.nets.pytorch_backend.transducer.error_calculator import ErrorCalculator
from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoder import DecoderRNNT
from espnet.nets.pytorch_backend.transducer.rnn_encoder import encoder_for
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs
from espnet.nets.pytorch_backend.transducer.utils import valid_aux_task_layer_list
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.snowfall.warpper.warpper_mmi import K2MMI
from espnet.snowfall.warpper.warpper_ctc import K2CTC
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)

import editdistance

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
        E2E.encoder_add_rnn_arguments(parser)
        E2E.encoder_add_custom_arguments(parser)

        E2E.decoder_add_general_arguments(parser)
        E2E.decoder_add_rnn_arguments(parser)
        E2E.decoder_add_custom_arguments(parser)

        E2E.training_add_custom_arguments(parser)
        E2E.transducer_add_arguments(parser)
        E2E.auxiliary_task_add_arguments(parser)

        E2E.att_scorer_arguments(parser)
        return parser

    @staticmethod
    def att_scorer_arguments(parser):
        """Add attention scorer argument."""
        group = parser.add_argument_group("Attention scorer arguments")
        group = add_att_scorer_arguments(group)

        return parser

    @staticmethod
    def encoder_add_general_arguments(parser):
        """Add general arguments for encoder."""
        group = parser.add_argument_group("Encoder general arguments")
        group = add_encoder_general_arguments(group)

        return parser

    @staticmethod
    def encoder_add_rnn_arguments(parser):
        """Add arguments for RNN encoder."""
        group = parser.add_argument_group("RNN encoder arguments")
        group = add_rnn_encoder_arguments(group)

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
    def decoder_add_custom_arguments(parser):
        """Add arguments for Custom decoder."""
        group = parser.add_argument_group("Custom decoder arguments")
        group = add_custom_decoder_arguments(group)

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
    def auxiliary_task_add_arguments(parser):
        """Add arguments for auxiliary task."""
        group = parser.add_argument_group("Auxiliary task arguments")
        group = add_auxiliary_task_arguments(group)

        return parser

    @property
    def attention_plot_class(self):
        """Get attention plot class."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        if self.etype == "custom":
            return self.encoder.conv_subsampling_factor * int(
                numpy.prod(self.subsample)
            )
        else:
            return self.enc.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0, training=True):
        """Construct an E2E object for transducer model."""
        torch.nn.Module.__init__(self)
        
        args = fill_missing_args(args, self.add_arguments)

        self.is_rnnt = True
        self.transducer_weight = args.transducer_weight

        self.use_aux_task = (
            True if (args.aux_task_type is not None and training) else False
        )

        self.use_aux_ctc = args.aux_ctc #and training
        self.aux_ctc_weight = args.aux_ctc_weight

        self.use_aux_mmi = args.aux_mmi #and training
        self.aux_mmi_weight = args.aux_mmi_weight

        self.use_aux_cross_entropy = args.aux_cross_entropy #and training
        self.aux_cross_entropy_weight = args.aux_cross_entropy_weight

        self.use_aux_mbr = args.aux_mbr
        self.aux_mbr_weight = args.aux_mbr_weight
        self.aux_mbr_beam = args.aux_mbr_beam

        self.use_att_scorer = args.att_scorer_weight > 0.0
        self.att_scorer_weight = args.att_scorer_weight

        if self.use_aux_task:
            n_layers = (
                (len(args.enc_block_arch) * args.enc_block_repeat - 1)
                if args.enc_block_arch is not None
                else (args.elayers - 1)
            )

            aux_task_layer_list = valid_aux_task_layer_list(
                args.aux_task_layer_list,
                n_layers,
            )
        else:
            aux_task_layer_list = []

        if "custom" in args.etype:
            if args.enc_block_arch is None:
                raise ValueError(
                    "When specifying custom encoder type, --enc-block-arch"
                    "should also be specified in training config. See"
                    "egs/vivos/asr1/conf/transducer/train_*.yaml for more info."
                )

            self.subsample = get_subsample(args, mode="asr", arch="transformer")

            self.encoder = CustomEncoder(
                idim,
                args.enc_block_arch,
                input_layer=args.custom_enc_input_layer,
                repeat_block=args.enc_block_repeat,
                self_attn_type=args.custom_enc_self_attn_type,
                positional_encoding_type=args.custom_enc_positional_encoding_type,
                positionwise_activation_type=args.custom_enc_pw_activation_type,
                conv_mod_activation_type=args.custom_enc_conv_mod_activation_type,
                aux_task_layer_list=aux_task_layer_list,
            )
            encoder_out = self.encoder.enc_out

            self.most_dom_list = args.enc_block_arch[:]
        else:
            self.subsample = get_subsample(args, mode="asr", arch="rnn-t")

            self.enc = encoder_for(
                args,
                idim,
                self.subsample,
                aux_task_layer_list=aux_task_layer_list,
            )
            encoder_out = args.eprojs

        if "custom" in args.dtype:
            if args.dec_block_arch is None:
                raise ValueError(
                    "When specifying custom decoder type, --dec-block-arch"
                    "should also be specified in training config. See"
                    "egs/vivos/asr1/conf/transducer/train_*.yaml for more info."
                )

            self.decoder = CustomDecoder(
                odim,
                args.dec_block_arch,
                input_layer=args.custom_dec_input_layer,
                repeat_block=args.dec_block_repeat,
                positionwise_activation_type=args.custom_dec_pw_activation_type,
                dropout_rate_embed=args.dropout_rate_embed_decoder,
            )
            decoder_out = self.decoder.dunits

            if "custom" in args.etype:
                self.most_dom_list += args.dec_block_arch[:]
            else:
                self.most_dom_list = args.dec_block_arch[:]
        else:
            self.dec = DecoderRNNT(
                odim,
                args.dtype,
                args.dlayers,
                args.dunits,
                blank_id,
                args.dec_embed_dim,
                args.dropout_rate_decoder,
                args.dropout_rate_embed_decoder,
            )
            decoder_out = args.dunits

        self.joint_network = JointNetwork(
            odim, encoder_out, decoder_out, args.joint_dim, args.joint_activation_type
        )

        # Attention Rescore
        if self.use_att_scorer > 0.0:
            self.att_scorer = Decoder(
                odim=odim,
                selfattention_layer_type=args.att_decoder_selfattn_layer_type,
                attention_dim=args.att_adim,
                attention_heads=args.att_aheads,
                conv_wshare=args.att_wshare,
                conv_kernel_length=args.att_ldconv_decoder_kernel_length,
                conv_usebias=args.att_ldconv_usebias,
                linear_units=args.att_dunits,
                num_blocks=args.att_dlayers,
                dropout_rate=args.att_dropout_rate,
                positional_dropout_rate=args.att_dropout_rate,
                self_attention_dropout_rate=args.att_attn_dropout_rate,
                src_attention_dropout_rate=args.att_attn_dropout_rate,
            )
            self.att_scorer_criterion = LabelSmoothingLoss(
                odim,
                ignore_id,
                args.lsm_weight,
                args.att_length_normalized_loss,
            )
        else:
            self.attention_scorer = None
            self.att_scorer_criterion = None

        if hasattr(self, "most_dom_list"):
            self.most_dom_dim = sorted(
                Counter(
                    d["d_hidden"] for d in self.most_dom_list if "d_hidden" in d
                ).most_common(),
                key=lambda x: x[0],
                reverse=True,
            )[0][0]

        self.etype = args.etype
        self.dtype = args.dtype

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim

        self.reporter = Reporter()

        self.error_calculator = None

        self.default_parameters(args)

        self.criterion = TransLoss(args.trans_type, self.blank_id)
        if training:

            decoder = self.decoder if self.dtype == "custom" else self.dec

            if args.report_cer or args.report_wer:
                self.error_calculator = ErrorCalculator(
                    decoder,
                    self.joint_network,
                    args.char_list,
                    args.sym_space,
                    args.sym_blank,
                    args.report_cer,
                    args.report_wer,
                )

            if self.use_aux_task:
                self.auxiliary_task = AuxiliaryTask(
                    decoder,
                    self.joint_network,
                    self.criterion,
                    args.aux_task_type,
                    args.aux_task_weight,
                    encoder_out,
                    args.joint_dim,
                )

        if self.use_aux_ctc:
            self.aux_ctc = ctc_for(
                Namespace(
                    num_encs=1,
                    eprojs=encoder_out,
                    dropout_rate=args.aux_ctc_dropout_rate,
                    ctc_type="warpctc",
                ),
                odim,
            )

        if self.use_aux_mmi:
            # assert self.use_aux_ctc # ctc is needed for aishell-1 but not for librispeech
            device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
            aux_mmi_module = K2MMI if args.aux_mmi_type == "mmi" else K2CTC
            self.aux_mmi=aux_mmi_module(idim=encoder_out,
                         lang=args.lang,
                         char_list=args.char_list,
                         device=device,
                         dropout=args.aux_mmi_dropout_rate,
                         den_scale=args.den_scale,
                         eos_id=self.eos,
                         use_segment=args.use_segment)

        if self.use_aux_cross_entropy:
            self.aux_decoder_output = torch.nn.Linear(decoder_out, odim)

            self.aux_cross_entropy = LabelSmoothingLoss(
                odim, ignore_id, args.aux_cross_entropy_smoothing
            )

        if self.use_aux_mbr:
            assert args.resume is not None # need a seed model
            self.beam_search = BeamSearchTransducer(
                decoder=self.decoder if "custom" in self.dtype else self.dec,
                joint_network=self.joint_network,
                beam_size=self.aux_mbr_beam,
                nbest=self.aux_mbr_beam,
                search_type='alsd',
            ) 
            self.char_list = args.char_list

            self.mbr_trans_type = args.trans_type
            if args.trans_type == "warp-transducer":
                from warprnnt_pytorch import RNNTLoss
                self.mbr_trans_loss = RNNTLoss(blank=self.blank_id, reduction="none")
            elif args.trans_type == "warp-rnnt":
                from warp_rnnt import rnnt_loss
                self.mbr_trans_loss = rnnt_loss
            print("built beam search decoder for MBR") 

        self.loss = None
        self.rnnlm = None

    def default_parameters(self, args):
        """Initialize/reset parameters for transducer.

        Args:
            args (Namespace): argument Namespace containing options

        """
        initializer(self, args)

    def forward(self, xs_pad, ilens, ys_pad, texts, xs_pad_orig):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): transducer loss value

        """
        # 1. encoder
        xs_pad = xs_pad[:, : max(ilens)]

        if "custom" in self.etype:
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            _hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            _hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)

        if self.use_aux_task:
            hs_pad, aux_hs_pad = _hs_pad[0], _hs_pad[1]
        else:
            hs_pad, aux_hs_pad = _hs_pad, None

        # 1.5. transducer preparation related
        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            ys_pad, hs_mask
        )
        """
        ys_in_pad : ys with blank_id in head. For decoder forward
        ys_out_pad : ys with ignore_id in tail. For aux task 
        target: ys with padding only, for RNNT loss computation
        pred_len: real length of hs_mask
        target_len: real length of target 
        """

        if self.use_aux_mbr:
            loss_mbr = self.mbr_forward(xs_pad_orig, ilens, ys_pad, hs_pad)
            loss_mbr *= self.aux_mbr_weight
        else:
            loss_mbr = 0.0

        # 2. decoder
        if "custom" in self.dtype:
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask)
        else:
            pred_pad = self.dec(hs_pad, ys_in_pad)

        z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))

        # 3. loss computation
        loss_trans = self.criterion(z, target, pred_len, target_len)

        if self.use_aux_task and aux_hs_pad is not None:
            loss_aux_trans, loss_aux_symm_kl = self.auxiliary_task(
                aux_hs_pad, pred_pad, z, target, pred_len, target_len
            )
        else:
            loss_aux_trans, loss_aux_symm_kl = 0.0, 0.0

        if self.use_aux_ctc or self.use_aux_mmi:
            if "custom" in self.etype:
                hlen = torch.IntTensor(
                    [h.size(1) for h in hs_mask],
                ).to(hs_mask.device)

        if self.use_aux_ctc:
            loss_ctc = self.aux_ctc_weight * self.aux_ctc(hs_pad, hlen, ys_pad, texts)
        else:
            loss_ctc = 0.0

        if self.use_aux_mmi:
            loss_mmi = self.aux_mmi_weight * self.aux_mmi(hs_pad, hlen, ys_pad, texts)
        else:
            loss_mmi = 0.0

        if self.use_aux_cross_entropy:
            loss_lm = self.aux_cross_entropy_weight * self.aux_cross_entropy(
                self.aux_decoder_output(pred_pad), ys_out_pad
            )
        else:
            loss_lm = 0.0

        if self.use_att_scorer:
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, _ = self.att_scorer(ys_in_pad, ys_mask, hs_pad, hs_mask)
            loss_att = self.att_scorer_criterion(pred_pad, ys_out_pad)
            loss_att *= self.att_scorer_weight
        else:
            loss_att = 0.0

        loss = (
            loss_trans
            + self.transducer_weight * (loss_aux_trans + loss_aux_symm_kl)
            + loss_ctc
            + loss_mmi
            + loss_lm
            + loss_mbr
            + loss_att
        )

        self.loss = loss
        loss_data = float(loss)

        # 4. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad, ys_pad)

        if not math.isnan(loss_data):
            self.reporter.report(
                loss_data,
                float(loss_trans),
                float(loss_ctc),
                float(loss_lm),
                float(loss_aux_trans),
                float(loss_aux_symm_kl),
                float(loss_mbr),
                float(loss_mmi),
                float(loss_att),
                cer,
                wer,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def mbr_forward(self, xs_pad_orig, ilens, ys_pad, hs_pad):
        # torch.set_printoptions(sci_mode=False)

        self.eval()
        batch_size = len(ilens)
        
        # (1) on-the-fly decoding
        with torch.no_grad():
            # decode without data augmentation (a.k.a., xs_pad_orig)
            if "custom" in self.etype:
                src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad_orig.device).unsqueeze(-2)
                specs, hs_mask = self.encoder(xs_pad_orig, src_mask)
            else:
                specs, hs_mask, _ = self.enc(xs_pad_orig, ilens)           


            hs = [h[h != 0] for h in hs_mask]
            hlens = list(map(int, [h.size(0) for h in hs]))
            specs = [h[:l] for h, l in zip(specs, hlens)]

            # multi-thread on-the-fly decoding on GPU
            """
            It is very inefficient to do on-the-fly decoding.
            We've tried multi-process but failed since the dataloader cannot work
              in forked process
            Multi-thread is used. Remember to use 'export OMP_NUM_THREADS=<ncpu>'
              to achieve faster decoding speed
            """
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(self.beam_search, h) for h in specs]
                wait(futures, return_when=ALL_COMPLETED)
            
                hyps = []
                for future in futures:
                    hyps.extend(future.result()) 
                hyps = [h.yseq[1:] for h in hyps] # exclude <sos>

                # for debug
                # for i, y in enumerate(ys_pad):
                #     ref_text = "".join([self.char_list[x] for x in y if x != self.ignore_id])
                #     print(f"ref_text: {ref_text}")
                #     for y in hyps[i * self.aux_mbr_beam: (i+1) * self.aux_mbr_beam]:
                #         hyp_text = "".join([self.char_list[x] for x in y if x != self.blank_id])
                #         print(f"hyp_text: {hyp_text}")
        
        self.train()

        # (2) compute edit distance
        dist = self.compute_edit_distance(hyps, ys_pad)
 
        if dist is None:
            print("Warning: An error encountered when editing distance", flush=True)
            return 0.0 # fail in editdistance.  

        # (3) RNN-T loss computation
        # prepare many inputs
        hyp_maxlen = max([len(hyp) for hyp in hyps])
        hyps_pad = [hyp + [self.ignore_id] * (hyp_maxlen - len(hyp)) for hyp in hyps]
        hyps_pad = torch.Tensor(hyps_pad).to(ys_pad.device).to(ys_pad.dtype)

        hyps_in_pad, hyps_out_pad, target, pred_len, target_len = prepare_loss_inputs(
                                                              hyps_pad, hs_mask) 
        
        idx = torch.arange(self.aux_mbr_beam * batch_size) // self.aux_mbr_beam
        pred_len = pred_len[idx]
        hs_pad = hs_pad[idx]

        # decoder and joint-net forward
        """ We are not sure which hs_pad should be used in decoder forward 
            Currently we are using the hs_pad from xs_pad, since we consider
            the encoder should also receive the gradient from denominator
        """
        if "custom" in self.dtype:
            hyps_mask = target_mask(hyps_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(hyps_in_pad, hyps_mask, hs_pad)
        else:
            pred_pad = self.dec(hs_pad, hyps_in_pad)

        z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))

        # loss computation
        # we need reduction = 'none' for utt-level probability
        # code for warp-rnnt is not tested
        if self.mbr_trans_type == "warp-rnnt":
            log_prob = torch.log_softmax(z, dim=-1)
            loss_trans = self.mbr_trans_loss(
                log_probs,
                target,
                pred_len,
                target_len,
                reduction=None,
                blank=self.blank_id,
                gather=True,
            )
        elif self.mbr_trans_type == "warp-transducer":
            loss_trans = self.mbr_trans_loss(z, target, pred_len, target_len)

        # This is exactly posterior P(W|O) 
        loss_trans = (-loss_trans).exp()
  
        # (4) MBR loss. 
        # Can also change to MMI loss if LM scores are provided 
        num = (loss_trans * dist).view(batch_size, self.aux_mbr_beam)
        den = loss_trans.view(batch_size, self.aux_mbr_beam)
        loss_mbr = num.sum(dim=-1) / den.sum(dim=-1)
        loss_mbr = loss_mbr.mean()
        
        # as cweng is using frame-level loss 
        # loss_mbr *= max(ilens)    
 
        return loss_mbr 
 
    def compute_edit_distance(self, hyps, refs):
        # hyps: list of list with number batch * beam
        # refs: 2-D tensor of labels. -1 means padding
  
        # convert refs into list and remove padding 
        refs_device = refs.device
        refs = refs.cpu().tolist()
        refs = [[x for x in t if x != self.ignore_id] for t in refs]
         
        if not len(hyps) % len(refs) == 0:
            raise ValueError("The number of hypotheses is not correct")

        beam = int(len(hyps) / len(refs))

        dist = [editdistance.eval(hyp, refs[i//beam]) 
                for i, hyp in enumerate(hyps)
               ] 
        dist = torch.IntTensor(dist).to(refs_device)
        return dist

    def encode_custom(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, D_enc)

        """
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)

        return enc_output.squeeze(0)

    def encode_rnn(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, D_enc)

        """
        p = next(self.parameters())

        ilens = [x.shape[0]]
        x = x[:: self.subsample[0], :]

        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        hs = h.contiguous().unsqueeze(0)

        hs, _, _ = self.enc(hs, ilens)

        return hs.squeeze(0)

    def recognize(self, x, beam_search):
        """Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            beam_search (class): beam search class

        Returns:
            nbest_hyps (list): n-best decoding results

        """
        self.eval()

        if "custom" in self.etype:
            h = self.encode_custom(x)
        else:
            h = self.encode_rnn(x)

        nbest_hyps = beam_search(h)
        return [asdict(n) for n in nbest_hyps]

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
