from dparser.modules import CHAR_LSTM, MLP, Biaffine, BiLSTM, IndependentDropout, SharedDropout, ScalarMix, CRFDependency
from dparser.utils.decoder import mst, eisner
from dparser.utils.corpus import Corpus

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class CRFParser(nn.Module):
    def __init__(self, config, embed):
        super(CRFParser, self).__init__()

        self.config = config
        # input layer(embedding layer and char LSTM layer)
        self.pretrained = nn.Embedding.from_pretrained(embed)
        self.word_embed = nn.Embedding(num_embeddings=config.n_words,
                                       embedding_dim=config.n_embed)
        self.char_lstm = CHAR_LSTM(num_embeddings=config.n_chars,
                                   embedding_dim=config.n_char_embed,
                                   output_dim=config.n_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        # LSTM layer
        self.lstm = BiLSTM(input_size=config.n_embed * 2,
                           hidden_size=config.n_lstm_hidden,
                           num_layers=config.n_lstm_layers,
                           dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        # MLP layer
        self.mlp_arc_h = MLP(in_features=config.n_lstm_hidden * 2,
                             out_features=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_arc_d = MLP(in_features=config.n_lstm_hidden * 2,
                             out_features=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(in_features=config.n_lstm_hidden * 2,
                             out_features=config.n_mlp_rel,
                             dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(in_features=config.n_lstm_hidden * 2,
                             out_features=config.n_mlp_rel,
                             dropout=config.mlp_dropout)

        # biaffine layer
        self.attn_arc = Biaffine(n_in=config.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.attn_rel = Biaffine(n_in=config.n_mlp_rel,
                                 n_out=config.n_rels,
                                 bias_x=True,
                                 bias_y=True)
        # decoder layer
        self.crf = CRFDependency()

        self.pad_index = config.pad_index
        self.unk_index = config.unk_index
        self.criterion = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.word_embed.weight)

    def forward(self, words, chars):
        batch_size, seq_len = words.shape
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.word_embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        #get outputs from embedding layer
        word_embed = self.pretrained(words) + self.word_embed(ext_words)
        char_embed = self.char_lstm(chars[mask])
        char_embed = pad_sequence(torch.split(char_embed, lens.tolist()), True)
        embeds = self.embed_dropout(word_embed, char_embed)
        embed = torch.cat(embeds, dim=-1)

        # lstm
        # sorted_lens, indices = torch.sort(lens, descending=True)
        # inverse_indics = indices.argsort()
        # x = pack_padded_sequence(embed[indices], sorted_lens, True)
        # x = self.lstm(x)[-1]
        # x, _ = pad_packed_sequence(x, True)
        # x = self.lstm_dropout(x)[inverse_indics]
        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)
        # apply MLP to the BiLSTM output states

        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        #[batch_size, seq_len, seq_len]
        s_arc = self.attn_arc(arc_d, arc_h)
        #[batch_size, seq_len, seq_len, n_rels]
        s_rel = self.attn_rel(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    @classmethod
    def load(cls, fp):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(fp, map_location=device)
        parser = cls(state['config'], state['embed'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)
        return parser

    def save(self, fp):
        state = {
            'config': self.config,
            'embed': self.pretrained.weight,
            'state_dict': self.state_dict()
        }
        torch.save(state, fp)

    def get_loss(self, s_arc, s_rel, arcs, rels, mask, mbr=True,
                 partial=False):
        r"""
        Args:
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
        """
        batch_size, seq_len = mask.shape
        arc_loss, arc_probs = self.crf(s_arc, mask, arcs, mbr, partial)
        # -1 denotes un-annoted arcs
        if partial:
            mask = mask & arcs.ge(0)

        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss

        return loss, arc_probs

    def decode(self, s_arc, s_rel, mask, tree=True, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        lens = mask.sum(1)
        pred_arcs = s_arc.argmax(-1)
        bad = [
            not Corpus.istree(seq[1:i + 1], proj)
            for i, seq in zip(lens.tolist(), pred_arcs.tolist())
        ]
        if tree and any(bad):
            decode_alg = eisner if proj else mst
            pred_arcs[bad] = decode_alg(s_arc[bad], mask[bad])

        pred_rels = s_rel.argmax(-1).gather(
            -1, pred_arcs.unsqueeze(-1)).squeeze(-1)
        return pred_arcs, pred_rels
