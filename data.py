import re
import torch

from vocabulary import Vocabulary

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
    """
    Read input and target language files and return a list
    of paired sentences, with text normalised

    Parameters
    ----------
    input: str
        Filepath to input language data file
    target: str
        Filepath to target language data file
    max_rows: int, default=None
        Max number of lines/sentences to read
    
    Returns
    -------
    paired: List[List[str]]
        All sentences, paired between languages
    input_vocab: Vocabulary
        Vocabulary object for input language
    target_vocab: Vocabulary
        Vocabulary object for output language
    """
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
    """
    Turn a pair of sentences into pytorch tensors, ready for use

    Parameters
    ----------
    pair: List[str]
        Input-target pair of sentences to tensorize
    input_vocab: Vocabulary
        Vocabulary object for input language
    target_vocab: Vocabulary
        Vocabulary object for target language
    eos: Union[int, str], default=1
        EOS token to append to sentences
    
    Returns
    -------
    tensors: List[torch.tensor]
        Pair of tensors corresponding to original pair of sentences
    """
    tensors = []
    vocabs = (input_vocab, target_vocab)
    for i, sentence in enumerate(pair):
        # turn words into list of indices
        index = index_from_sentence(sentence, vocabs[i])
        # add EOS token to sentence
        index.append(eos)
        # turn lists into tensors
        tensor = torch.tensor(index, dtype=torch.long, device=device)
        tensors.append(tensor)
    return tensors


