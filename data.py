import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_line(line):
    # turn to lower case
    line = line.lower()
    # remove leading/trailing white space
    line = line.strip()
    # remove punctuation
    line = re.sub(r"[^\w ]", ' ', line)
    # remove digits
    line = re.sub(r"\d+", ' ', line)
    return line

class Vocabulary:
    def __init__(self):
        self.word2index = {'SOS': 0, 'EOS': 1}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2
        self.word_counts = {}
        self.vocab_ = set()

    def add_sentence(self, line):
        for word in line.split(' '):
            if word == ' ' or word == '':
                continue
            self.add_word(word)
    
    def add_word(self, word):
        if word in self.word2index.keys():
            self.word_counts[word] += 1
        else:
            self.vocab_.add(word)
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_counts[word] = 1
            self.n_words += 1

def read_file(input, max_rows=None):
    count = 0
    vocab = Vocabulary()
    lines = []
    with open(input, 'r') as infile:
        while True:
            if max_rows is not None:
                if count > max_rows:
                    break
                count += 1
            line = infile.readline()
            if not line:
                print('EOF reached')
                break
            line = normalize_line(line)
            vocab.add_sentence(line)
            lines.append(line)
    return lines, vocab

def build_pairs_and_vocab(input, target, max_rows=None):
    l1_lines, l1_vocab = read_file(input, max_rows)
    l2_lines, l2_vocab = read_file(target, max_rows)

    paired = list(zip(l1_lines, l2_lines))

    print(f'Number of sentences: {len(paired)}')
    print(f'Number of input words: {l1_vocab.n_words}')
    print(f'Number of target words: {l2_vocab.n_words}')
    return paired, l1_vocab, l2_vocab

def index_from_sentence(sentence, vocab):
    return [vocab.word2index[word] for word in sentence.split(' ')]

def tensorize_sentence(pair, input_vocab, target_vocab, eos=1):
    # turn words into list of indices
    input_index = index_from_sentence(pair[0], input_vocab)
    target_index = index_from_sentence(pair[1], target_vocab)
    # add EOS token to sentence
    input_index.append(eos)
    target_index.append(eos)
    # turn lists into tensors
    input_tensor = torch.tensor(input_index, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_index, dtype=torch.long, device=device)
    return (input_tensor, target_tensor)


