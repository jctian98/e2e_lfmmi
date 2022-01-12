#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0
import argparse
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple, Union

import k2
import kaldialign
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel

from snowfall.models import AcousticModel

Pathlike = Union[str, Path]


def setup_logger(log_filename: Pathlike, log_level: str = 'info', use_console: bool = True) -> None:
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d-%H-%M-%S')
    log_filename = '{}-{}'.format(log_filename, date_time)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f'%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s'
    else:
        formatter = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'

    level = logging.ERROR
    if log_level == 'debug':
        level = logging.DEBUG
    elif log_level == 'info':
        level = logging.INFO
    elif log_level == 'warning':
        level = logging.WARNING
    logging.basicConfig(filename=log_filename,
                        format=formatter,
                        level=level,
                        filemode='w')
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger('').addHandler(console)


def load_checkpoint(
        filename: Pathlike,
        model: AcousticModel,
        optimizer: Optional[object] = None,
        scheduler: Optional[object] = None,
        scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    logging.info('load checkpoint from {}'.format(filename))

    checkpoint = torch.load(filename, map_location='cpu')

    keys = [
        'state_dict', 'optimizer', 'scheduler', 'epoch', 'learning_rate', 'objf', 'valid_objf',
        'num_features', 'num_classes', 'subsampling_factor',
        'global_batch_idx_train'
    ]
    missing_keys = set(keys) - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(f"Missing keys in checkpoint: {missing_keys}")

    if isinstance(model, DistributedDataParallel):
        model = model.module

    if not list(model.state_dict().keys())[0].startswith('module.') \
            and list(checkpoint['state_dict'])[0].startswith('module.'):
        # the checkpoint was saved by DDP
        logging.info('load checkpoint from DDP')
        dst_state_dict = model.state_dict()
        src_state_dict = checkpoint['state_dict']
        for key in dst_state_dict.keys():
            src_key = '{}.{}'.format('module', key)
            dst_state_dict[key] = src_state_dict.pop(src_key)
        assert len(src_state_dict) == 0
        model.load_state_dict(dst_state_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])

    model.num_features = checkpoint['num_features']
    model.num_classes = checkpoint['num_classes']
    model.subsampling_factor = checkpoint['subsampling_factor']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if scaler is not None:
        scaler.load_state_dict(checkpoint['grad_scaler'])

    return checkpoint


def average_checkpoint(filenames: List[Pathlike], model: AcousticModel) -> Dict[str, Any]:
    logging.info('average over checkpoints {}'.format(filenames))

    avg_model = None

    # sum
    for filename in filenames:
        checkpoint = torch.load(filename, map_location='cpu')
        checkpoint_model = checkpoint['state_dict']
        if avg_model is None:
            avg_model = checkpoint_model
        else:
            for k in avg_model.keys():
                avg_model[k] += checkpoint_model[k]
    # average
    for k in avg_model.keys():
        if avg_model[k] is not None:
            if avg_model[k].is_floating_point():
                avg_model[k] /= len(filenames)
            else:
                avg_model[k] //= len(filenames)

    checkpoint['state_dict'] = avg_model

    keys = [
        'state_dict', 'optimizer', 'scheduler', 'epoch', 'learning_rate', 'objf', 'valid_objf',
        'num_features', 'num_classes', 'subsampling_factor',
        'global_batch_idx_train'
    ]
    missing_keys = set(keys) - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(f"Missing keys in checkpoint: {missing_keys}")

    if not list(model.state_dict().keys())[0].startswith('module.') \
            and list(checkpoint['state_dict'])[0].startswith('module.'):
        # the checkpoint was saved by DDP
        logging.info('load checkpoint from DDP')
        dst_state_dict = model.state_dict()
        src_state_dict = checkpoint['state_dict']
        for key in dst_state_dict.keys():
            src_key = '{}.{}'.format('module', key)
            dst_state_dict[key] = src_state_dict.pop(src_key)
        assert len(src_state_dict) == 0
        model.load_state_dict(dst_state_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])

    model.num_features = checkpoint['num_features']
    model.num_classes = checkpoint['num_classes']
    model.subsampling_factor = checkpoint['subsampling_factor']

    return checkpoint


def save_checkpoint(
        filename: Pathlike,
        model: Union[AcousticModel, DistributedDataParallel],
        optimizer: object,
        scheduler: object,
        epoch: int,
        learning_rate: float,
        objf: float,
        valid_objf: float,
        global_batch_idx_train: int,
        local_rank: int = 0,
        scaler: Optional[GradScaler] = None
) -> None:
    if local_rank is not None and local_rank != 0:
        return
    if isinstance(model, DistributedDataParallel):
        model = model.module
    logging.info(f'Save checkpoint to {filename}: epoch={epoch}, '
                 f'learning_rate={learning_rate}, objf={objf}, valid_objf={valid_objf}')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'grad_scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'learning_rate': learning_rate,
        'objf': objf,
        'valid_objf': valid_objf,
        'global_batch_idx_train': global_batch_idx_train,
        'num_features': model.num_features,
        'num_classes': model.num_classes,
        'subsampling_factor': model.subsampling_factor,
    }
    torch.save(checkpoint, filename)


def save_training_info(
        filename: Pathlike,
        model_path: Pathlike,
        current_epoch: int,
        learning_rate: float,
        objf: float,
        best_objf: float,
        valid_objf: float,
        best_valid_objf: float,
        best_epoch: int,
        local_rank: int = 0
):
    if local_rank is not None and local_rank != 0:
        return

    with open(filename, 'w') as f:
        f.write('model_path: {}\n'.format(model_path))
        f.write('epoch: {}\n'.format(current_epoch))
        f.write('learning rate: {}\n'.format(learning_rate))
        f.write('objf: {}\n'.format(objf))
        f.write('best objf: {}\n'.format(best_objf))
        f.write('valid objf: {}\n'.format(valid_objf))
        f.write('best valid objf: {}\n'.format(best_valid_objf))
        f.write('best epoch: {}\n'.format(best_epoch))

    logging.info('write training info to {}'.format(filename))


def get_phone_symbols(symbol_table: k2.SymbolTable,
                      pattern: str = r'^#\d+$') -> List[int]:
    '''Return a list of phone IDs containing no disambiguation symbols.

    Caution:
      0 is not a phone ID so it is excluded from the return value.

    Args:
      symbol_table:
        A symbol table in k2.
      pattern:
        Symbols containing this pattern are disambiguation symbols.
    Returns:
      Return a list of symbol IDs excluding those from disambiguation symbols.
    '''
    regex = re.compile(pattern)
    symbols = symbol_table.symbols
    ans = []
    for s in symbols:
        if not regex.match(s):
            ans.append(symbol_table[s])
    if 0 in ans:
        ans.remove(0)
    ans.sort()
    return ans


def cut_id_dumper(dataloader, path: Path):
    """
    Debugging utility. Writes processed cut IDs to a file.
    Expects ``return_cuts=True`` to be passed to the Dataset class.

    Example::

        >>> for batch in cut_id_dumper(dataloader):
        ...     pass
    """
    if not dataloader.dataset.return_cuts:
        return dataloader  # do nothing, "return_cuts=True" was not set
    with path.open('w') as f:
        for batch in dataloader:
            for cut in batch['supervisions']['cut']:
                print(cut.id, file=f)
            yield batch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def describe(model: torch.nn.Module):
    logging.info('=' * 80)
    logging.info('Model parameters summary:')
    logging.info('=' * 80)
    total = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        logging.info(f'* {name}: {num_params:>{80 - len(name) - 4}}')
    logging.info('=' * 80)
    logging.info(f'Total: {total}')
    logging.info('=' * 80)


def get_texts(best_paths: k2.Fsa, indices: Optional[torch.Tensor] = None) -> List[List[int]]:
    '''Extract the texts from the best-path FSAs, in the original order (before
       the permutation given by `indices`).
       Args:
           best_paths:  a k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
                    containing multiple FSAs, which is expected to be the result
                    of k2.shortest_path (otherwise the returned values won't
                    be meaningful).  Must have the 'aux_labels' attribute, as
                  a ragged tensor.
           indices: possibly a torch.Tensor giving the permutation that we used
                    on the supervisions of this minibatch to put them in decreasing
                    order of num-frames.  We'll apply the inverse permutation.
                    Doesn't have to be on the same device as `best_paths`
      Return:
          Returns a list of lists of int, containing the label sequences we
          decoded.
    '''
    # remove any 0's or -1's (there should be no 0's left but may be -1's.)
    aux_labels = k2.ragged.remove_values_leq(best_paths.aux_labels, 0)
    aux_shape = k2.ragged.compose_ragged_shapes(best_paths.arcs.shape(),
                                                aux_labels.shape())
    # remove the states and arcs axes.
    aux_shape = k2.ragged.remove_axis(aux_shape, 1)
    aux_shape = k2.ragged.remove_axis(aux_shape, 1)
    aux_labels = k2.RaggedInt(aux_shape, aux_labels.values())
    assert (aux_labels.num_axes() == 2)
    aux_labels, _ = k2.ragged.index(aux_labels,
                                    invert_permutation(indices).to(dtype=torch.int32,
                                                                   device=best_paths.device))
    return k2.ragged.to_list(aux_labels)


def invert_permutation(indices: torch.Tensor) -> torch.Tensor:
    ans = torch.zeros(indices.shape, device=indices.device, dtype=torch.long)
    ans[indices] = torch.arange(0, indices.shape[0], device=indices.device)
    return ans


def find_first_disambig_symbol(symbols: k2.SymbolTable) -> int:
    return min(v for k, v in symbols._sym2id.items() if k.startswith('#'))


def store_transcripts(path: Pathlike, texts: Iterable[Tuple[str, str]]):
    with open(path, 'w') as f:
        for ref, hyp in texts:
            print(f'ref={ref}', file=f)
            print(f'hyp={hyp}', file=f)

def write_error_stats(f: TextIO, test_set_name: str, results: List[Tuple[str,str]]) -> None:
    subs: Dict[Tuple[str,str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0,0,0,0,0])
    num_corr = 0
    ERR = '*'
    for ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        for ref_word,hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word,hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for r,_ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = '%.2f' % (100.0 * tot_errs / ref_len)

    logging.info(
        f'[{test_set_name}] %WER {tot_errs / ref_len:.2%} '
        f'[{tot_errs} / {ref_len}, {ins_errs} ins, {del_errs} del, {sub_errs} sub ]'
    )

    print(f"%WER = {tot_err_rate}", file=f)
    print(f"Errors: {ins_errs} insertions, {del_errs} deletions, {sub_errs} substitutions, over {ref_len} reference words ({num_corr} correct)",
          file=f)
    print("Search below for sections starting with PER-UTT DETAILS:, SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
          file=f)

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [ [[x],[y]] for x,y in ali ]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i+1][0] != ali[i+1][1]:
                    ali[i+1][0] = ali[i][0] + ali[i+1][0]
                    ali[i+1][1] = ali[i][1] + ali[i+1][1]
                    ali[i] = [[],[]]
            ali = [ [list(filter(lambda a: a != ERR, x)),
                     list(filter(lambda a: a != ERR, y))]
                     for x,y in ali ]
            ali = list(filter(lambda x: x != [[],[]], ali))
            ali = [ [ERR if x == [] else ' '.join(x),
                     ERR if y == [] else ' '.join(y)]
                    for x,y in ali ]

        print(' '.join((ref_word if ref_word == hyp_word else f'({ref_word}->{hyp_word})'
                        for ref_word,hyp_word in ali)), file=f)


    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count,(ref,hyp) in sorted([(v,k) for k,v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count,ref in sorted([(v,k) for k,v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count,hyp in sorted([(v,k) for k,v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _,word,counts in sorted([(sum(v[1:]),k,v) for k,v in words.items()], reverse=True):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
