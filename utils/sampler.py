import torch
import torch.utils.data as data
import random

# We cannot make the data loading totally random due to the slow ceph 
# So we use this sampler to ensure that the data reading will be contrained
# in limited number of arks
class BufferSampler(object):
    def __init__(self, length, utts_per_ark, batch_size, buf_size, seed=0, prefetch_ratio=0.3):
        """
        length: number of minibatches
        utts_per_ark: the number of utterances in each ark except the last one
        batch_size: the batch size used in training
        buf_size: the number of arks that you want to put in the buffer
        prefetch_ratio: when the remained number of minibatches is below this ratio, 
                        we start to featch the arks in next group
                        0.5 means we begin to read the next group of arks when half of
                        this group is consumed
        """
    
        self.batch_per_ark = int(utts_per_ark / batch_size)
        self.buf_size = buf_size
        self.prefetch_ratio = prefetch_ratio
        self.num_batches = length
        self.seed = seed
        
        # seed2 is a bias on seed. It never work independently
        # it is different on different GPU rank
        try:
            import torch.distributed as dist
            self.seed2 = dist.get_rank()
        except:
            print("Sampler: you are not using DDP training paradigm.")
            print("Sampler: So the rank bias of random seed is set to 0", flush=True)
            self.seed2 = 0

    def __iter__(self):
        self.reset()
        print("A new iterator in sampler is built")
        # make 0, ..., length - 1 in indices
        assert sum(self.indices) == self.num_batches * (self.num_batches - 1) / 2
        return iter(self.indices)
 
    def __len__(self):
        return self.num_batches
    
    """
    This is the core function of this sampler
    The output indices have features below:
    (1) All arks are divided into several groups. Each group consists
        of at most `buf_size` arks
    (2) The indices are from the same group until all data in this group
        is consumed. This is to avoid buffering too many arks.
    (3) For DDP training, the grouping results are identical. This is to 
        ensure that the length distribution in this group is similar 
        across the different ranks. This is controlled by self.seed.
    (4) Within the group, the order of indices cannot be identical 
        across the ranks, or the global mini-batch will be identical
        in each epochs. In this case, we ensure that for any valid
        t, the t-th minibatch in this group across the different 
        ranks are from the same ark-id but not necessarily the same.
        This provides more variation in training data. This is controlled
        by `self.seed2`
    """
    def _get_indices(self):
        num_arks = int(self.num_batches // self.batch_per_ark) + \
                     int(self.num_batches % self.batch_per_ark != 0)

        # group arks
        ark_ids = list(range(num_arks))
        random.shuffle(ark_ids)
        start = 0
        groups = []
        while start < num_arks:
            end = min(start + self.buf_size, num_arks)
            group = ark_ids[start: end]
            groups.append(group)
            start += self.buf_size
 
        def process_group(group, seed_bias):
            eg_indices = [] # global idx of the mini-batches
            ark_indices = [] # ark idx of the mini-batches
            for i, arkid in enumerate(group):
                start = arkid * self.batch_per_ark
                end = min((arkid+1) * self.batch_per_ark, self.num_batches)
 
                eg_indice = list(range(start, end))
                eg_indices.append(eg_indice)
 
                ark_indice = [i] * (end - start)
                ark_indices.append(ark_indice)

            ark_indices = self._splice_list(ark_indices)

            # the ark_indices is with self.seed
            # as we need it identical on different GPU ranks
            random.shuffle(ark_indices)

            # eg_indices is with self.seed + self.seed2
            # we need it different on different GPU ranks 
            random.seed(self.seed + self.seed2)
            for e in eg_indices:
                random.shuffle(e)
           
            # we need recover the seed so the next time
            # we shuffle ark_indices will still have
            # the same results across the GPUs.
            # we do not use `self.seed` only as it 
            # always return to the same start point
            random.seed(self.seed + seed_bias + 888) 
            
            # combine finally
            group_indice = []
            for i in ark_indices:
                batch_idx = eg_indices[i].pop()
                group_indice.append(batch_idx)
            return group_indice

        group_indices = [process_group(g, b) for b, g in enumerate(groups)]
        return self._splice_list(group_indices)
   
    # Using these indices leads to identical global batches in 
    # each epoch 
    def _get_indices_deprecated(self):
        num_arks = int(self.num_batches // self.batch_per_ark) + \
                     int(self.num_batches % self.batch_per_ark != 0)

        ark_ids = list(range(num_arks))
        random.shuffle(ark_ids)
        ark_indices = [(idx * self.batch_per_ark, 
                        min((idx+1) * self.batch_per_ark, self.num_batches))
                        for idx in ark_ids]
        ark_indices = [list(range(*idx)) for idx in ark_indices]

        # grouping ark indices and shuffle within the group
        start = 0 
        group_indices = []
        while start < num_arks:
            end = min(start + self.buf_size, num_arks)
            
            group_indice = ark_indices[start: end]
            group_indice = self._splice_list(group_indice)
            random.shuffle(group_indice)
            group_indices.append(group_indice)
            start += self.buf_size

        group_indices = self._splice_list(group_indices)
        
        return group_indices


    def reset(self, seed=None):
        # change the seed and reset the indices
        # It is important to use the seed in DDP training
        # as the result of sampler is identical on each GPU.
        # Since the index of minibatch is proportional to the
        # length of utternace, this will help us to balance 
        # the load of each GPU 
        seed = seed if seed is not None else self.seed + 1
        self.seed = seed
        random.seed(seed)
        self.indices = self._get_indices()

    def _splice_list(self, lsts):
        out = []
        for l in lsts:
            out += l
        return out

    # this provides the prefetch factor of dataloader
    # no matter how much mini-batches to preload, all
    # arks in the next group will be loaded. So a small
    # ratio is enough and will save memory
    # just make sure 0.3 group of data will not run out
    # before the next group is loaded
    def get_prefetch_factor(self):
        return int(self.buf_size * self.batch_per_ark * self.prefetch_ratio)

class testdataset:
    def __init__(self, length):
        self.l = length

    def __len__(self):
        return self.l

if __name__ == '__main__':
    # 26 batches (52 utts), 4 batches in each ark, max 3 arks in buf, batch_size = 2
    num_minibatches = 26
    sampler = BufferSampler(num_minibatches, utts_per_ark=4, batch_size=2, buf_size=3)
    out = ""
    for i in iter(sampler):
        out += f"{i}\t"

