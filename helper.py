import pennylane as qml

import random
import os
import argparse
import time
from io import open
import numpy as np

from typing import List, Tuple, Dict
from collections import defaultdict

import nltk
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def nll_loss(scores, targets):
    """
    :param scores: shape (batch_size, time_steps, vocab_size)
    :param targets: shape (batch_size, time_steps)
    :return: cross entropy loss, takes input (N, C) and (N) where C=num classes (words in vocab)
    """
    # targets are shape (batch_size, time_steps)
    batch_size = targets.size(0)
    # scores: (batch_size, time_steps, vocab_size)(batch_size * time_steps, vocab_size)
    scores = scores.reshape(-1, scores.size(2))
    # targets: (batch_size, time_steps) -> (batch_size*time_steps)
    targets = targets.reshape(-1)

    return F.cross_entropy(input=scores, target=targets) * batch_size


def perplexity(data:DataLoader, model:nn.Module, batch_size:int) -> float:
    """
    :param data:
    :param model:
    :param batch_size:
    :return:
    """
    model.eval()
    with torch.no_grad():
        losses = []
        states = model.init_state(batch_size)
        #print('states', states[0][0].shape)
        for (x, y) in data:
            # scores, states = model(x, states)
            scores, states = model.forward(x=x, states=states)
            loss = nll_loss(scores, y)
            losses.append(loss.data.item() / batch_size)

    perplexity = np.exp(np.mean(losses))

    return perplexity

class Sequence_Dataset(Dataset):
    def __init__(self, x:torch.LongTensor, y:torch.LongTensor):
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

def _load_text_data(path: str) -> str:
    """
    read text file
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        text = f.read()
    return text


def _tokenize(text: str) -> List[str]:
    """
    Remove unwanted tokens like <s> and \n
    :param text: tokens separated by ' '
    :return: list of tokens
    """
    text = text.replace('<s>', '')
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    text = text.lower()
    words = text.strip().split(' ')
    return words

def _init_corpora(path:str, topic:str, freq_threshold:int,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    :param path:
    :param topic:
    :return:
    """
    # read the text
    train = _load_text_data(os.path.join(path, f"{topic}.train.txt"))
    valid = _load_text_data(os.path.join(path, f"{topic}.valid.txt"))
    test = _load_text_data(os.path.join(path, f"{topic}.test.txt"))

    if topic == 'wiki':
        # remove wiki tags
        train = ' '.join(train.replace('=', ' ').split())
        valid = ' '.join(valid.replace('=', ' ').split())
        test = ' '.join(test.replace('=', ' ').split())

    # split into word/token
    train = _tokenize(text=train)
    valid = _tokenize(text=valid)
    test = _tokenize(text=test)

    # apply frequency threshold to training set
    freq_dist = nltk.FreqDist(train)
    train = ['<unk>' if freq_dist[t] < freq_threshold else t for t in train]

    # create vocabulary: set of words in train and word to index mapping
    vocab = sorted(set(train))
    word2index = {word: index+1 for index, word in enumerate(vocab)}
    word2index['<pad>'] = 0

    # convert each word to a list of integers. if word is not in vocab, we use unk
    train = [word2index[word] for word in train]
    valid = [word2index[word] if word in word2index else word2index['<unk>'] for word in valid]
    test = [word2index[word] if word in word2index else word2index['<unk>'] for word in test]

    # return (n, ) arrays for train, valid, test, and the word2index dict
    return np.array(train), np.array(valid), np.array(test), word2index

def _generate_io_sequences(sequence: np.ndarray, time_steps: int) -> Tuple:
    """
    :param sequence: sequence of integer representation of words
    :param time_steps: number of time steps in LSTM cell
    :return: Tuple of torch tensors of shape (n, time_steps)
    """
    sequence = torch.LongTensor(sequence)

    # from seq we generate 2 copies.
    inputs, targets = sequence, sequence[1:]

    # split seq into seq of of size time_steps
    inputs = torch.split(tensor=inputs, split_size_or_sections=time_steps)
    targets = torch.split(tensor=targets, split_size_or_sections=time_steps)

    # recall: word2index['<pad>'] = 0
    inputs_padded = pad_sequence(sequences=inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(sequences=targets, batch_first=True, padding_value=0)

    return (inputs_padded, targets_padded)


def _build_dataloader(data:np.ndarray, time_steps:int, batch_size:int) -> DataLoader:
    """
    :param data: input list of integers
    :param batch_size: hyper parameter, for mini-batch size
    :param time_steps: hyper parameter for sequence length for bptt
    :return: DataLoader for SGD
    """
    # given int list, generate input and output sequences of length = time_steps
    inputs, targets = _generate_io_sequences(sequence=data, time_steps=time_steps)
    
    # cut off any data that will create incomplete batches
    num_batches = len(inputs) // batch_size
    inputs = inputs[:num_batches*batch_size]
    targets = targets[:num_batches*batch_size]
    
    # create Dataset object and from it create data loader
    dataset = Sequence_Dataset(x=inputs, y=targets)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def init_datasets(topic:str, freq_threshold:int, time_steps:int, batch_size:int, path:str) -> Dict:
    """
    :param path: path to data files: [topic].train.txt
    :param topic: [topic].train.txt, where topic can be wikitext, or nyt_covid
    :param freq_threshold: hyperparameter words in training set with freq < threshold are replaced by '<unk>'
    :param time_steps: hyperparameter number of time steps and therefore seq_length for bptt
    :param batch_size: hyperparameter batch size
    :return: datasets dict
    """
    train, valid, test, word2index = _init_corpora(path=path, topic=topic, freq_threshold=freq_threshold)
    train_loader = _build_dataloader(data=train, time_steps=time_steps, batch_size=batch_size)
    valid_loader = _build_dataloader(data=valid, time_steps=time_steps, batch_size=batch_size)
    test_loader = _build_dataloader(data=test, time_steps=time_steps, batch_size=batch_size)
    datasets = {
        'data_loaders': (train_loader, valid_loader, test_loader),
        'word2index': word2index,
        'vocab_size': len(word2index)
    }
    return datasets


def train(data: Tuple[DataLoader, DataLoader, DataLoader], model: nn.Module, epochs: int, learning_rate: float,
          learning_rate_decay: float, max_grad: float) -> Tuple[nn.Module, Dict]:
    """
    model training loop
    :param data: tup of DataLoaders train, valid, test
    :param model: LSTM_Model
    :param epochs: number of epochs
    :param learning_rate: initial learning rate
    :param learning_rate_decay: learning rate decay factor
    :param max_grad: gradient clipped at this value
    :return: trained model and perplexity scores (per epoch for valid and final score for test)
    """
    train_loader, valid_loader, test_loader = data
    start_time = time.time()

    perplexity_scores = {
        'valid': [],
        'test': 0
    }

    total_words = 0
    print("Starting training.\n")
    batch_size = train_loader.batch_size

    for epoch in range(epochs):
        states = model.init_state(batch_size)

        if epoch + 1 > 5:
            learning_rate = learning_rate / learning_rate_decay

        for i, (x, y) in enumerate(train_loader):
            total_words += x.numel()
            model.zero_grad()
            states = model.detach_states(states)
            scores, states = model.forward(x, states)
            loss = nll_loss(scores=scores, targets=y)
            loss.backward()

            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                for param in model.parameters():
                    param -= learning_rate * param.grad

            if i % (len(train_loader) // 10) == 0:
                end_time = time.time()
                print("batch no = {:d} / {:d}, ".format(i, len(train_loader)) +
                      "train loss = {:.3f}, ".format(loss.item() / batch_size) +
                      "wps = {:d}, ".format(round(total_words / (end_time - start_time))) +
                      "lr = {:.3f}, ".format(learning_rate) +
                      "since beginning = {:d} mins, ".format(round((end_time - start_time) / 60)))

        model.eval()
        valid_perplexity = perplexity(data=valid_loader, model=model, batch_size=batch_size)
        perplexity_scores['valid'].append(valid_perplexity)
        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch + 1, valid_perplexity))
        print("*************************************************\n")

    test_perp = perplexity(data=test_loader, model=model, batch_size=batch_size)
    perplexity_scores['test'] = test_perp
    print("Test set perplexity : {:.3f}".format(test_perp))

    print("Training is over.")

    return model, perplexity_scores