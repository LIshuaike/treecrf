from collections import namedtuple

Sentence = namedtuple('Sentencee', [
    'ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD',
    'PDEPREL'
],
                      defaults=[None] * 10)


class Corpus():
    root = '<ROOT>'

    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __repr__(self):
        return '\n'.join('\n'.join('\t'.join(map(str, i))
                                   for i in zip(*(f for f in sentence if f))) +
                         '\n' for sentence in self)

    @property
    def words(self):
        return [[self.root] + list(sentence.FORM) for sentence in self]

    @property
    def tags(self):
        return [[self.root] + list(sentence.CPOS) for sentence in self]

    @property
    def heads(self):
        return [[0] + list(map(int, sentence.HEAD)) for sentence in self]

    @property
    def rels(self):
        return [[self.root] + list(sentence.DEPREL) for sentence in self]

    @classmethod
    def load(cls, fp, proj=False, columns=range(10)):
        sentences, columns = [], []
        with open(fp, encoding='utf=8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    if columns:
                        sentences.append(Sentence(*columns))
                    columns = []
                else:
                    for i, column in enumerate(line.split('\t')):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)
            if columns:
                sentences.append(Sentence(*columns))
        # print(f'before discard non-projective :{len(sentences)}')
        if proj:
            sentences = [
                s for s in sentences
                if cls.isprojective(list(map(int, s.HEAD)))
            ]
            # print(f'after discard non-projective :{len(sentences)}')
        corpus = cls(sentences)

        return corpus

    @classmethod
    def isprojective(cls, sequence):
        r"""
        Checks if a dependency tree is projective.
        This also works for partial annotation.

        Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
        which are hard to detect in the scenario of partial annotation.

        Args:
            sequence (list[int]):
                A list of head indices.

        Returns:
            ``True`` if the tree is projective, ``False`` otherwise.

        Examples:
            >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
            False
            >>> CoNLL.isprojective([3, -1, 2])
            False
        """

        pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
        for i, (hi, di) in enumerate(pairs):
            for hj, dj in pairs[i + 1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    return False
                if lj <= hi <= rj and hj == di:
                    return False
                if (li < lj < ri
                        or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                    return False
        return True

    # def isProj(self):
    #     n = len(self.words)
    #     words = self.words
    #     if self.start > 1: return False
    #     if self.start == 0: words = [None] + words
    #     for i in range(1, n):
    #         hi = words[i].head
    #         for j in range(i + 1, hi):
    #             hj = words[j].head
    #             if (hj - hi) * (hj - i) > 0:
    #                 return False
    #     return True

    @classmethod
    def istree(cls, sequence, proj=False, multiroot=False):
        r"""
        Checks if the arcs form an valid dependency tree.

        Args:
            sequence (list[int]):
                A list of head indices.
            proj (bool):
                If ``True``, requires the tree to be projective. Default: ``False``.
            multiroot (bool):
                If ``False``, requires the tree to contain only a single root. Default: ``True``.

        Returns:
            ``True`` if the arcs form an valid tree, ``False`` otherwise.

        Examples:
            >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
            True
            >>> CoNLL.istree([3, 0, 0, 3], proj=True)
            False
        """

        from dparser.utils.utils import tarjan
        if proj and not cls.isprojective(sequence):
            return False
        n_roots = sum(head == 0 for head in sequence)
        if n_roots == 0:
            return False
        if not multiroot and n_roots > 1:
            return False
        if any(i == head for i, head in enumerate(sequence, 1)):
            return False
        return next(tarjan(sequence), None) is None

    def save(self, fp):
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(f"{self}\n")


print(Corpus.load('del_sentence', True))