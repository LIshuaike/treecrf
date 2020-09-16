class Metric():
    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __eq__(self, other):
        return self.score == other

    def __ge__(self, other):
        return self.score > other

    def __gt__(self, other):
        return self.score > other

    def __ne__(self, other):
        return self.score != other

    @property
    def score(self):
        raise AttributeError


class AttachmentMethod(Metric):
    def __init__(self, eps=1e-6):
        super(AttachmentMethod, self).__init__()
        self.eps = eps
        self.total = 0
        self.corrent_arcs = 0
        self.corrent_rels = 0

    def __call__(self, pred_arcs, pred_rels, gold_arcs, gold_rels, mask):
        arc_mask = pred_arcs.eq(gold_arcs) & mask
        rel_mask = pred_rels.eq(gold_rels) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.total += len(arc_mask_seq)
        self.corrent_arcs += arc_mask_seq.sum().item()
        self.corrent_rels += rel_mask_seq.sum().item()

    def __repr__(self):
        return f'UAS:{self.uas:.2%} LAS:{self.las:.2%}'

    @property
    def score(self):
        return self.las

    @property
    def uas(self):
        return self.corrent_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.corrent_rels / (self.total + self.eps)