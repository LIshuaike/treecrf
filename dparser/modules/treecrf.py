import torch
import torch.nn as nn
import torch.autograd as autograd

from dparser.utils.utils import stripe


class CRFDependency(nn.Module):
    r"""
    First-order TreeCRF for calculating partition functions and marginals in :math:`O(n^3)` for projective dependency trees.

    References:
        - Yu Zhang, Zhenghua Li and Min Zhang. 2020.
          `Efficient Second-Order TreeCRF for Neural Dependency Parsing`_.

    .. _Efficient Second-Order TreeCRF for Neural Dependency Parsing:
        https://www.aclweb.org/anthology/2020.acl-main.302/
    """
    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False, partial=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs.
                This should be provided for loss calculation.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        training = scores.requires_grad
        batch_size, seq_len, _ = scores.shape
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        probs = scores
        if mbr:
            probs, = autograd.grad(logZ, scores, retain_graph=training)

        if target is None:
            return probs
        # the second inside process is needed if use partial annotation
        if partial:
            score = self.inside(scores, mask, target)
        else:
            score = scores.gather(
                -1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
        loss = (logZ - score) / mask.sum()

        return loss, probs

    def inside(self, scores, mask, cands=None):
        # the end position of each sentence in a batch
        lens = mask.sum(1)
        batch_size, seq_len, _ = scores.shape
        # [seq_len, seq_len, batch_size]
        scores = scores.permute(2, 1, 0)
        s_i = torch.full_like(scores, float('-inf'))
        s_c = torch.full_like(scores, float('-inf'))
        s_c.diagonal().fill_(0)

        # set the scores of arcs excluded by cands to -inf
        if cands is not None:
            mask = mask.index_fill(1, lens.new_tensor(0), 1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
            cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
            cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
            cands = cands.permute(2, 1, 0) & mask
            scores = scores.masked_fill(~cands, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w

            # ilr = C(i->r) + C(j->r+1)
            # [n, w, batch_size]
            ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
            if ilr.requires_grad:
                ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            il = ir = ilr.permute(2, 0, 1).logsumexp(-1)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i
            # with I(j->i) of n spans
            s_i.diagonal(-w).copy_(il + scores.diagonal(-w))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i
            # with I(i->j) of n spans
            s_i.diagonal(w).copy_(ir + scores.diagonal(w))

            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(-w) .copy_(cl.permute(2, 0, 1).logsumexp(-1))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
            # disable multi words to modify the root
            s_c[0, w][lens.ne(w)] = float('-inf')

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()
