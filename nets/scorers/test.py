        
        num, den = self.graph_compiler.compile(texts, self.P, replicate_den=True)
        T = x.size()[1]
        scores = []
        for t in range(T, 0, -1):
            supervision = torch.Tensor([[0, 0, t]]).to(torch.int32) # [idx, start, length]
            dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
            lats = k2.intersect_dense(den, dense_fsa_vec, output_beam=10.0)
            frame_score = lats.get_tot_scores(log_semiring=True, use_double_scores=True)
            scores.append(frame_score)
        tot_scores = torch.cat(scores).unsqueeze(0)


