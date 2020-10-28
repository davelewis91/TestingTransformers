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