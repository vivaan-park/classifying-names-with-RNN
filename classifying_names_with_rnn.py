import glob
import os
import string
import random
from unidecode import unidecode

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.output_softmax = nn.LogSoftmax(dim=1)

    def forward(self, tensor, hidden):
        hidden = F.tanh(self.input_to_hidden(tensor) + self.hidden_to_hidden(hidden))
        output = self.hidden_to_output(hidden)
        output = self.output_softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def find_files(path):
    return glob.glob(path)


def read_line(file_path):
    with open(file_path, 'r') as file:
        return [text.strip() for text in file]


def unicode_to_ascii(unicode_str):
    return unidecode(unicode_str)


def all_letters():
    return string.ascii_letters + ' .,;'


def letter_to_index(letter):
    return all_letters().find(letter)


def letter_to_tensor(letter):
    tensor = torch.zeros(1, len(all_letters()))
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters()))
    for l, letter in enumerate(line):
        tensor[l][0][letter_to_index(letter)] = 1
    return tensor


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example(all_categories, category_lines):
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def evaluate(rnn, line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def main():
    device = {
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    }

    data = 'data/names/*.txt'

    all_categories = []
    category_lines = {}

    for file_path in find_files(data):
        category = os.path.splitext(os.path.basename(file_path))[0]
        all_categories.append(category)

        lines = read_line(file_path)
        category_lines[category] = lines

    criterion = nn.NLLLoss()
    learning_rate = 0.005

    n_hidden = 128
    rnn = RNN(len(all_letters()), n_hidden, len(all_categories))

    for i in range(10):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_lines)
        print('category =', category, '/ line =', line)

    hidden = rnn.init_hidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for param in rnn.parameters():
        param.data.add_(param.grad.data, alpha=learning_rate)

    confusion = torch.zeros(len(all_categories), len(all_categories))
    n_confusion = 10000

    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_lines)
        output = evaluate(rnn, line_tensor)
        guess, guess_i = category_from_output(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    for i in range(len(all_categories)):
        confusion[i] = confusion[i] / confusion[i].sum()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


if __name__ == '__main__':
    main()


