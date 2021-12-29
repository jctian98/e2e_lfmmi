# Author: Jinchuan Tian
# tyriontian@tencent.com

# A simple example to show the inference process of RNN-T modules
# The only dependency for this script is pytorch (torch1.7.1+cuda101)
#
# run: python3 trace_rnnt.py <resources-dir>
# egs: python3 trace_rnnt.py ./resources


import sys
import os
import torch
import json
from argparse import Namespace

# If you do not need the Espnet dependency, you can just copy the transducer directory
from espnet.nets.pytorch_backend.transducer.custom_encoder import CustomEncoder
from espnet.nets.pytorch_backend.transducer.rnn_decoder import DecoderRNNT
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork


def main():
    """ parse configs """
    export_dir = sys.argv[1]
    json_file = os.path.join(export_dir, "model.json")
    idim, odim, args = json.load(open(json_file))
    args = Namespace(**args) 
    device = torch.device("cuda") # also works for CPU 
    
    """ load modules """
    encoder = CustomEncoder(
                idim,
                args.enc_block_arch,
                input_layer=args.custom_enc_input_layer,
                repeat_block=args.enc_block_repeat,
                self_attn_type=args.custom_enc_self_attn_type,
                positional_encoding_type=args.custom_enc_positional_encoding_type,
                positionwise_activation_type=args.custom_enc_pw_activation_type,
                conv_mod_activation_type=args.custom_enc_conv_mod_activation_type,
                aux_task_layer_list=args.aux_task_layer_list,
    )
    enc_pt = os.path.join(export_dir, "encoder.pt")
    encoder.load_state_dict(torch.load(enc_pt))
    encoder.eval().to(device)

    decoder = DecoderRNNT(
                odim,
                args.dtype,
                args.dlayers,
                args.dunits,
                args.char_list.index("<blank>"),
                args.dec_embed_dim,
                args.dropout_rate_decoder,
                args.dropout_rate_embed_decoder,
    )
    dec_pt = os.path.join(export_dir, "decoder.pt")
    decoder.load_state_dict(torch.load(dec_pt))
    decoder.eval().to(device)

    joint_network = JointNetwork(
        odim, 
        encoder.enc_out, 
        args.dunits, 
        args.joint_dim, 
        args.joint_activation_type
    )
    joint_pt = os.path.join(export_dir, "joint_net.pt")
    joint_network.load_state_dict(torch.load(joint_pt))
    joint_network.eval().to(device)
    print("INFO: Successfully load encoder, decoder and joint-network")

    """ Module Inference """
    B = 2                        # Batch_size
    T = 400                      # Maximum time index
    U = 4                        # Maximum word index
    enc_idim = idim
    enc_odim = encoder.enc_out
    n_vocab = odim
    dec_odim = args.dunits

    # For batch-inference, you may want to pass masks to the encoder and call it like 
    # 'encoder(enc_in, masks)'. In this case, the paddings will not be considered.
    # See espnet/nets/pytorch_backend/nets_utils.py:make_non_pad_mask for details.
    # but it's ok if B = 1.
    enc_in = torch.rand([B, T, enc_idim]).to(device)
    enc_out, _  = encoder(enc_in, None)
    print("encoder_out size: ", enc_out.size())  # enc_out: [B, sub(T), enc_odim], T is sub-sumpled by a factor of 6
     
    # decoder inference
    decoder.set_device(enc_out.device) # needed before inference
    decoder.set_data_type(enc_out.dtype) # needed before inference
    # The LSTM should work as long as the 'ey' is consistent with 'states'.
    # So you may use a cache and a state-select method to save computation.
    states = decoder.init_state(B)
    for _ in range(U):
        tokens = torch.randint(low=0, high=n_vocab, size=[B, 1]).to(device)
        ey = decoder.embed(tokens)
        dec_out, states = decoder.rnn_forward(ey, states)
    print("decoder_out size: ", dec_out.size()) # dec_out: [B, 1, dec_odim]

    # joint-network inference
    # It is safe to feed two 4-dim tensors.
    # However, the joint network should work as long as two conditions are met.
    # (1) element-wise addtion of enc_out and dec_out will not raise shape error (allow broadcastable)
    # (2) enc_out.size()[-1] == dec_out.size()[-1] == size_of_joint_net
    # The size of output should be the same with enc_out except the last dimention:
    # the last dimention is n_vocab
    enc_out = enc_out.unsqueeze(2) # [B, sub(T), 1, enc_odim]
    dec_out = dec_out.unsqueeze(1) # [B, 1, U, dec_odim]
    joint_out = joint_network(enc_out, dec_out)
    print("joint_out size: ", joint_out.size()) # [B, T, U, n_vocab] 
    
    # the output distribution is over this char_list: args.char_list
 
if __name__ == "__main__":
    main()
