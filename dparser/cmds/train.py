import os
from datetime import datetime, timedelta
from dparser import Model, CRFParser
from dparser.metrics import AttachmentMethod
from dparser.utils import Corpus, Embedding, Vocab
from dparser.utils.data import TextDataset, batchify

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np
np.random.seed(0)


class Train():
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(name, help='Train a model')
        subparser.add_argument('--buckets',
                               default=64,
                               type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--nopunct',
                               dest='punct',
                               action='store_false',
                               help='whether to exclude punctuation')
        subparser.add_argument('--ftrain',
                               default='data/ctb7/train.conll',
                               help='path to dev file')
        subparser.add_argument('--fdev',
                               default='data/ctb7/dev.conll',
                               help='path to dev file')
        subparser.add_argument('--ftest',
                               default='data/ctb7/test.conll',
                               help='path to test file')
        subparser.add_argument('--fembed',
                               default='data/giga.100.txt',
                               help='path to pretrained embedding file')
        subparser.add_argument('--patience',
                               default=100,
                               type=int,
                               help='patience to stop')
        subparser.add_argument('--mbr',
                               action='store_true',
                               help='whether to use MBR decoding')
        subparser.add_argument('--tree',
                               action='store_true',
                               help='whether to ensure well-formedness')
        subparser.add_argument('--proj',
                               action='store_true',
                               help='whether to projectivize the data')
        subparser.add_argument('--partial',
                               action='store_true',
                               help='whether partial annotation is included')
        subparser.add_argument('--unk',
                               default=None,
                               help='unk token in pretrained embeddings')
        return subparser

    def __call__(self, config):

        train = Corpus.load(config.ftrain, config.proj)
        dev = Corpus.load(config.fdev, config.proj)
        test = Corpus.load(config.ftest, config.proj)
        if config.preprocess or not os.path.exists(config.vocab):
            print('Preprocess the data')
            vocab = Vocab.from_corpus(corpus=train, min_freq=2)
            vocab.load_embedding(Embedding.load(config.fembed, config.unk))
            torch.save(vocab, config.vocab)
        else:
            print('load vocabulary')
            vocab = torch.load(config.vocab)

        config.update({
            'n_words': vocab.n_init,
            'n_chars': vocab.n_chars,
            'n_tags': vocab.n_tags,
            'n_rels': vocab.n_rels,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        print(vocab)

        print('Load the dataset')
        ds_train = TextDataset(vocab.numericalize(train), config.buckets)
        ds_dev = TextDataset(vocab.numericalize(dev), config.buckets)
        ds_test = TextDataset(vocab.numericalize(test), config.buckets)

        # set data loaders
        dl_train = batchify(ds_train, config.batch_size)
        dl_dev = batchify(ds_dev, config.batch_size)
        dl_test = batchify(ds_test, config.batch_size)
        print(f"{'train:':6} {len(ds_train):5} sentences in total, "
              f"{len(dl_train):3} batches provided")
        print(f"{'dev:':6} {len(ds_dev):5} sentences in total, "
              f"{len(dl_dev):3} batches provided")
        print(f"{'test:':6} {len(ds_test):5} sentences in total, "
              f"{len(dl_test):3} batches provided")

        print("Create model")
        parser = CRFParser(config, vocab.embedding)
        parser.to(config.device)
        print(f'{parser}\n')

        model = Model(config, vocab, parser)

        total_time = timedelta()
        best_e, best_metric = 1, AttachmentMethod()
        model.optimizer = Adam(model.parser.parameters(), config.lr,
                               (config.mu, config.nu), config.epsilon)
        model.scheduler = ExponentialLR(model.optimizer,
                                        config.decay**(1 / config.decay_steps))

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            print(f"Epoch {epoch} / {config.epochs}:")
            metric_p = model.train(dl_train)
            print(f"{'train:':6} {metric_p}")
            loss, dev_metric_p = model.evaluate(dl_dev)
            print(f"{'dev:':6} Loss: {loss:.4f}  {dev_metric_p}")
            loss, metric_p = model.evaluate(dl_test)
            print(f"{'test:':6} Loss: {loss:.4f} {metric_p}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric_p > best_metric and epoch > 10:
                best_e, best_metric = epoch, dev_metric_p
                model.parser.save(config.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break
        model.parser = CRFParser.load(config.model)
        loss, metirc_p = model.evaluate(dl_test)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric_p.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")