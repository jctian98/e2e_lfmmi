from typing import Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler


def l1_norm(x):
    return torch.sum(torch.abs(x))


def l2_norm(x):
    return torch.sum(torch.pow(x, 2))


def linf_norm(x):
    return torch.max(torch.abs(x))


def measure_weight_norms(model: nn.Module, norm: str = 'l2') -> Dict[str, float]:
    """
    Compute the norms of the model's parameters.

    :param model: a torch.nn.Module instance
    :param norm: how to compute the norm. Available values: 'l1', 'l2', 'linf'
    :return: a dict mapping from parameter's name to its norm.
    """
    with torch.no_grad():
        norms = {}
        for name, param in model.named_parameters():
            if norm == 'l1':
                val = l1_norm(param)
            elif norm == 'l2':
                val = l2_norm(param)
            elif norm == 'linf':
                val = linf_norm(param)
            else:
                raise ValueError(f"Unknown norm type: {norm}")
            norms[name] = val.item()
        return norms


def measure_semiorthogonality(model: nn.Module) -> Dict[str, float]:
    """
    Compute the semi-orthogonality objective function proposed by:

        "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks",
        Daniel Povey, Gaofeng Cheng, Yiming Wang, Ke Li, Hainan Xu, Mahsa Yarmohamadi,
        Sanjeev Khudanpur, Interspeech 2018
    """
    with torch.no_grad():
        scores = {}
        for name, m in model.named_modules():
            if hasattr(m, 'constrain_orthonormal'):
                weight = m.state_dict()['conv.weight']
                dim = weight.shape[0]
                w = weight.reshape(dim, -1)
                P = torch.mm(w, w.t())
                scale = torch.trace(torch.mm(P, P.t()) / torch.trace(P))
                I = torch.eye(dim, dtype=P.dtype, device=P.device)
                Q = P - scale * I
                score = torch.trace(torch.mm(Q, Q.t()))
                scores[name] = score.item()
        return scores


def measure_gradient_norms(model: nn.Module, norm: str = 'l1') -> Dict[str, float]:
    """
    Compute the norms of the gradients for each of model's parameters.

    :param model: a torch.nn.Module instance
    :param norm: how to compute the norm. Available values: 'l1', 'l2', 'linf'
    :return: a dict mapping from parameter's name to its gradient's norm.
    """
    with torch.no_grad():
        norms = {}
        for name, param in model.named_parameters():
            if norm == 'l1':
                val = l1_norm(param)
            elif norm == 'l2':
                val = l2_norm(param)
            elif norm == 'linf':
                val = linf_norm(param)
            else:
                raise ValueError(f"Unknown norm type: {norm}")
            norms[name] = val.item()
        return norms


def optim_step_and_measure_param_change(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """
    Perform model weight update and measure the "relative change in parameters per minibatch."
    It is understood as a ratio between the L2 norm of the difference between original and updates parameters,
    and the L2 norm of the original parameter. It is given by the formula:

        .. math::
            \begin{aligned}
                \delta = \frac{\Vert\theta - \theta_{new}\Vert^2}{\Vert\theta\Vert^2}
            \end{aligned}
    """
    param_copy = {n: p.detach().clone() for n, p in model.named_parameters()}
    if scaler:
        scaler.step(optimizer)
    else:
        optimizer.step()
    relative_change = {}
    with torch.no_grad():
        for n, p_new in model.named_parameters():
            p_orig = param_copy[n]
            delta = l2_norm(p_orig - p_new) / l2_norm(p_orig)
            relative_change[n] = delta.item()
    return relative_change
