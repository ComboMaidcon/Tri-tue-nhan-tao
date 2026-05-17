import re
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def build(self, sentences, freq_threshold):
        counter = Counter()
        for sentence in sentences:
            tokens = re.findall(r"\w+", sentence.lower())
            counter.update(tokens)
        idx = len(self.stoi)
        for word, freq in sorted(counter.items()):
            if freq >= freq_threshold:
                self.stoi[word] = idx
                self.itos[idx]  = word
                idx += 1
        print(f" Vocab size: {len(self.stoi)} từ (threshold={freq_threshold})")

    def numericalize(self, text):
        tokens = re.findall(r"\w+", text.lower())
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in tokens]
