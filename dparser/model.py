from dparser.metrics import AttachmentMethod

import torch
import torch.nn as nn


class Model():
    def __init__(self, config, vocab, parser):

        self.config = config
        self.vocab = vocab
        self.parser = parser

    # def train(self, loader, partial=False):
    #     self.parser.train()

    #     metric = AttachmentMethod()
    #     with torch.autograd.detect_anomaly():
    #         for i, (words, chars, tags, arcs, rels) in enumerate(loader):
    #             # [batch_size, seq_len]
    #             mask = words.ne(self.vocab.pad_index)
    #             # ignore the first token of each sentence
    #             mask[:, 0] = 0
    #             s_arc, s_rel = self.parser(words, chars)

    #             loss, s_arc = self.parser.get_loss(s_arc,
    #                                                s_rel,
    #                                                arcs,
    #                                                rels,
    #                                                mask,
    #                                                mbr=True,
    #                                                partial=partial)

    #             # loss.backward()
    #             # nn.utils.clip_grad_norm_(self.parser.parameters(),
    #             #                         self.config.clip)
    #             # self.optimizer.step()
    #             # self.scheduler.step()
    #             loss = loss / self.config.update_steps
    #             loss.backward()

    #             if (i + 1) % self.config.update_steps == 0:
    #                 nn.utils.clip_grad_norm_(self.parser.parameters(),
    #                                          self.config.clip)
    #                 self.optimizer.step()
    #                 self.scheduler.step()
    #                 self.optimizer.zero_grad()

    #             pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel, mask)
    #             if self.config.partial:
    #                 mask &= arcs.ge(0)
    #             if not self.config.punct:
    #                 puncts = words.new_tensor(self.vocab.puncts)
    #                 mask &= words.unsqueeze(-1).ne(puncts).all(-1)
    #             metric(pred_arcs, pred_rels, arcs, rels, mask)
    #     return metric

    def train(self, loader, partial=False):
        self.parser.train()

        metric = AttachmentMethod()
        for i, (words, chars, tags, arcs, rels) in enumerate(loader):
            # [batch_size, seq_len]
            # print(self.vocab.id2word(words))
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.parser(words, chars)

            loss, s_arc = self.parser.get_loss(s_arc,
                                               s_rel,
                                               arcs,
                                               rels,
                                               mask,
                                               mbr=True,
                                               partial=partial)

            loss = loss / self.config.update_steps
            loss.backward()

            if (i + 1) % self.config.update_steps == 0:
                nn.utils.clip_grad_norm_(self.parser.parameters(),
                                         self.config.clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel, mask)
            if self.config.partial:
                mask &= arcs.ge(0)
            if not self.config.punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)
            metric(pred_arcs, pred_rels, arcs, rels, mask)
        return metric

    @torch.no_grad()
    def evaluate(self, loader, partial=True):
        self.parser.eval()

        loss, metirc = 0, AttachmentMethod()

        for words, chars, tags, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            mask[:, 0] = 0

            s_arc, s_rel = self.parser(words, chars)
            loss, s_arc = self.parser.get_loss(s_arc,
                                               s_rel,
                                               arcs,
                                               rels,
                                               mask,
                                               mbr=True,
                                               partial=partial)
            pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel, mask)

            if self.config.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if specified
            if not self.config.punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)

            loss += loss.item()

            metirc(pred_arcs, pred_rels, arcs, rels, mask)
        loss /= len(loader)

        return loss, metirc

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()

        all_arcs, all_rels = [], [], []
        for words, chars, tags in loader:
            mask = words.ne(self.vocab.pad_index)
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_arc, s_rel = self.parser(words, chars)
            pred_arcs, pred_rels = self.parser.decode(s_arc, s_rel, mask)

            all_arcs.extend(torch.split(pred_arcs[mask], lens))
            all_rels.extend(torch.split(pred_rels[mask], lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_arcs, all_rels
