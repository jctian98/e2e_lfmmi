import math
import torch
from torch import Tensor
from typing import Dict, List, Tuple


def encode_supervisions(supervisions: Dict[str, Tensor]) -> Tuple[Tensor, List[str]]:
    """
    Encodes Lhotse's ``batch["supervisions"]`` dict into a pair of torch Tensor,
    and a list of transcription strings.

    The supervision tensor has shape ``(batch_size, 3)``.
    Its second dimension contains information about sequence index [0],
    start frames [1] and num frames [2].

    The batch items might become re-ordered during this operation -- the returned tensor
    and list of strings are guaranteed to be consistent with each other.

    This mimics subsampling by a factor of 4 with Conv1D layer with no padding.
    """
    supervision_segments = torch.stack(
        (supervisions['sequence_idx'],
         (((supervisions['start_frame'] - 1) // 2 - 1) // 2),
         (((supervisions['num_frames'] - 1) // 2 - 1) // 2)),
        1
    ).to(torch.int32)
    supervision_segments = torch.clamp(supervision_segments, min=0)
    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]
    texts = supervisions['text']
    texts = [texts[idx] for idx in indices]
    return supervision_segments, texts


def get_tot_objf_and_num_frames(
        tot_scores: Tensor,
        frames_per_seq: Tensor
    ) -> Tuple[torch.Tensor, int, int]:
    """Figures out the total score(log-prob) over all successful supervision segments
    (i.e. those for which the total score wasn't -infinity), and the corresponding
    number of frames of neural net output
         Args:
            tot_scores: a Torch tensor of shape (num_segments,) containing total scores
                       from forward-backward
            frames_per_seq: a Torch tensor of shape (num_segments,) containing the number of
                           frames for each segment
        Returns:
             Returns a tuple of 3 scalar tensors:  (tot_score, ok_frames, all_frames)
        where ok_frames is the frames for successful (finite) segments, and
       all_frames is the frames for all segments (finite or not).
    """
    mask = torch.ne(tot_scores, -math.inf)
    # finite_indexes is a tensor containing successful segment indexes, e.g.
    # [ 0 1 3 4 5 ]
    finite_indexes = torch.nonzero(mask).squeeze(1)
    ok_frames = frames_per_seq[finite_indexes].sum()
    all_frames = frames_per_seq.sum()
    return tot_scores[finite_indexes].sum(), ok_frames, all_frames
