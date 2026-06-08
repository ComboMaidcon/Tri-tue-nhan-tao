import re
from collections import Counter

try:
    import nltk
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    _nltk_available = True
except ModuleNotFoundError:
    nltk = None
    _nltk_available = False


# okenizer
def tokenize(text: str) -> list[str]:
    text = text.lower()
    if _nltk_available:
        return nltk.word_tokenize(text)
    return re.findall(r"\w+", text)


# Vocabulary
class Vocabulary:
    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.word2idx: dict[str, int] = {
            "<PAD>": self.PAD,
            "<SOS>": self.SOS,
            "<EOS>": self.EOS,
            "<UNK>": self.UNK,
        }
        self.idx2word: dict[int, str] = {v: k for k, v in self.word2idx.items()}

    # Build
    def build(self, captions: list[str]) -> None:
        counter: Counter = Counter()
        for cap in captions:
            counter.update(tokenize(cap))

        idx = len(self.word2idx)
        for word, freq in counter.most_common():     # most_common → thứ tự ổn định
            if freq >= self.freq_threshold:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx]  = word
                    idx += 1

        print(f"✓ Vocab size: {len(self.word2idx):,} từ "
              f"(freq_threshold={self.freq_threshold})")

    # Encode / Decode
    def encode(self, caption: str) -> list[int]:
        tokens = tokenize(caption)
        return (
            [self.SOS]
            + [self.word2idx.get(t, self.UNK) for t in tokens]
            + [self.EOS]
        )

    def decode(self, indices: list[int]) -> str:
        words = []
        for i in indices:
            if i == self.EOS:
                break
            if i not in (self.PAD, self.SOS):
                words.append(self.idx2word.get(i, "<UNK>"))
        return " ".join(words)

    # Dunder
    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __repr__(self) -> str:
        return (
            f"Vocabulary(size={len(self)}, "
            f"freq_threshold={self.freq_threshold})"
        )
