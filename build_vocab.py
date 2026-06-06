import pickle
import json
import csv
import re
from collections import Counter
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import nltk
    _nltk_available = True
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
except ModuleNotFoundError:
    nltk = None
    _nltk_available = False


def tokenize(text: str) -> list[str]:
    text = text.lower()
    if _nltk_available:
        return nltk.word_tokenize(text)
    return re.findall(r"\w+", text)

class Vocabulary:
    """
    Vocab xây từ train captions.
    Special tokens: <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3
    """
    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build(self, captions: list):
        """Build vocab từ captions list"""
        counter = Counter()
        for cap in captions:
            counter.update(tokenize(cap))

        idx = len(self.word2idx)
        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        print(f'✓ Vocab size: {len(self.word2idx):,} từ (freq_threshold={self.freq_threshold})')

    def encode(self, caption: str) -> list:
        """Caption → [SOS] + token indices + [EOS]"""
        tokens = tokenize(caption)
        return ([self.SOS]
                + [self.word2idx.get(t, self.UNK) for t in tokens]
                + [self.EOS])

    def decode(self, indices: list) -> str:
        """Indices → caption (remove special tokens)"""
        words = []
        for i in indices:
            if i == self.EOS:
                break
            if i not in (self.PAD, self.SOS):
                words.append(self.idx2word.get(i, '<UNK>'))
        return ' '.join(words)

    def __len__(self):
        return len(self.word2idx)


CAPTIONS_FILE = "/home/codespace/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/results.csv"
FREQ_THRESHOLD = 5  # Unified với CNN+LSTM
OUTPUT_DIR = Path(".")
VOCAB_PKL = OUTPUT_DIR / "vocab.pkl"
VOCAB_JSON = OUTPUT_DIR / "vocab.json"

# Demo captions (dùng khi không có dataset)
DEMO_CAPTIONS = [
    "a dog is running in the park",
    "a cat is sitting on the couch",
    "a boy is playing with a ball",
    "a girl is reading a book",
    "a dog and cat are playing together",
    "a man is cooking in the kitchen",
    "a woman is walking in the park",
    "children are playing in the park",
    "a person is riding a bicycle",
    "a dog is sleeping on the bed",
] * 50  # Repeat để có đủ tần suất


def read_captions_with_csv(file_path: Path) -> list:
    captions = []
    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter='|')
        header = next(reader, None)
        if header is None:
            return captions

        header = [col.strip() for col in header]
        cap_col = [i for i, c in enumerate(header)
                   if 'comment' in c.lower() or 'caption' in c.lower()]
        if not cap_col:
            raise ValueError("Không tìm được cột caption trong CSV.")
        cap_idx = cap_col[-1]

        for row in reader:
            if cap_idx < len(row):
                caption = row[cap_idx].strip()
                if caption:
                    captions.append(caption)
    return captions


if __name__ == "__main__":
    print("=" * 60)
    print("BUILDING VOCABULARY")
    print("=" * 60)
    
    # Thử load từ file thực nếu có
    captions = None
    if Path(CAPTIONS_FILE).exists():
        print("\nĐọc captions từ:", CAPTIONS_FILE)
        if pd is not None:
            df = pd.read_csv(CAPTIONS_FILE, sep="|")
            df.columns = df.columns.str.strip()

            print("Các cột:", list(df.columns))

            cap_col = [c for c in df.columns
                       if "comment" in c.lower() or "caption" in c.lower()]
            if not cap_col:
                raise ValueError("Không tìm được cột caption!")
            cap_col = cap_col[-1]
            print(f"✓ Dùng cột: '{cap_col}'")

            captions = df[cap_col].dropna().astype(str).tolist()
        else:
            print("pandas không có, dùng csv reader thay thế")
            captions = read_captions_with_csv(Path(CAPTIONS_FILE))

        print(f"✓ Số lượng captions: {len(captions):,}")
    else:
        print("\n📖 File dataset không tìm thấy, dùng demo captions")
        captions = DEMO_CAPTIONS
        print(f"✓ Demo captions: {len(captions):,}")

    # Build vocab
    vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
    vocab.build(captions)

    # Lưu vocab dạng pickle
    with open(VOCAB_PKL, "wb") as f:
        pickle.dump(vocab, f)
    print(f"\n✓ Lưu vocab object → {VOCAB_PKL}")

    # Lưu vocab dạng JSON (readable + compatible)
    with open(VOCAB_JSON, "w") as f:
        json.dump(vocab.word2idx, f, indent=2)
    print(f"✓ Lưu word2idx → {VOCAB_JSON}")

    # Demo
    print(f"\nThống kê:")
    print(f"   Total tokens: {len(vocab):,}")
    print(f"   Freq threshold: {vocab.freq_threshold}")
    print(f"\nDemo encode/decode:")
    sample_caption = "a dog is running in the park"
    encoded = vocab.encode(sample_caption)
    decoded = vocab.decode(encoded)
    print(f"   Input:  '{sample_caption}'")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: '{decoded}'")
