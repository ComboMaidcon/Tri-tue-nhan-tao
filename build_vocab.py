import argparse
import csv
import json
import pickle
from pathlib import Path

try:
    import pandas as pd
    _pd_available = True
except ModuleNotFoundError:
    pd = None
    _pd_available = False

from vocabulary import Vocabulary


# Đường dẫn mặc định
DEFAULT_CAPTIONS = (
    "/home/codespace/.cache/kagglehub/datasets/"
    "hsankesara/flickr-image-dataset/versions/1/"
    "flickr30k_images/results.csv"
)
DEFAULT_FREQ      = 5
DEFAULT_OUTPUT    = Path(".")
VOCAB_PKL_NAME    = "vocab.pkl"
VOCAB_JSON_NAME   = "vocab.json"


# Đọc captions
def _read_with_pandas(file_path: Path) -> list[str]:
    df = pd.read_csv(file_path, sep="|")
    df.columns = df.columns.str.strip()
    cap_col = [c for c in df.columns if "comment" in c.lower() or "caption" in c.lower()]
    if not cap_col:
        raise ValueError("Không tìm được cột caption trong CSV.")
    col = cap_col[-1]
    print(f"Dùng cột: '{col}'")
    return df[col].dropna().astype(str).str.strip().tolist()


def _read_with_csv(file_path: Path) -> list[str]:
    captions = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader, None)
        if header is None:
            return captions
        header = [c.strip() for c in header]
        cap_idx_list = [i for i, c in enumerate(header)
                        if "comment" in c.lower() or "caption" in c.lower()]
        if not cap_idx_list:
            raise ValueError("Không tìm được cột caption trong CSV.")
        cap_idx = cap_idx_list[-1]
        for row in reader:
            if cap_idx < len(row):
                caption = row[cap_idx].strip()
                if caption:
                    captions.append(caption)
    return captions


def load_captions(file_path: Path) -> list[str]:
    if _pd_available:
        return _read_with_pandas(file_path)
    print("pandas không khả dụng, dùng csv reader thay thế.")
    return _read_with_csv(file_path)


# Build & lưu vocab 
def build_and_save(
    captions: list[str],
    freq_threshold: int,
    output_dir: Path,
) -> Vocabulary:
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build(captions)

    pkl_path  = output_dir / VOCAB_PKL_NAME
    json_path = output_dir / VOCAB_JSON_NAME

    # Lưu dạng pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Lưu vocab object → {pkl_path}")

    # Lưu dạng JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vocab.word2idx, f, ensure_ascii=False, indent=2)
    print(f"Lưu word2idx    → {json_path}")

    return vocab


# Load vocab từ file
def load_vocab(pkl_path: Path) -> Vocabulary:
    """Load Vocabulary đã được lưu từ file pickle."""
    with open(pkl_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded vocab: {len(vocab):,} từ  ←  {pkl_path}")
    return vocab


def load_or_build_vocab(
    captions: list[str],
    freq_threshold: int = DEFAULT_FREQ,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Vocabulary:
    pkl_path = output_dir / VOCAB_PKL_NAME
    if pkl_path.exists():
        print("Load vocab từ file pickle...")
        return load_vocab(pkl_path)
    print("Xây dựng vocab từ captions...")
    return build_and_save(captions, freq_threshold, output_dir)


# CLI
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Vocabulary từ Flickr30k captions."
    )
    parser.add_argument(
        "--captions", type=str, default=DEFAULT_CAPTIONS,
        help="Đường dẫn tới file results.csv của Flickr30k.",
    )
    parser.add_argument(
        "--freq", type=int, default=DEFAULT_FREQ,
        help=f"Ngưỡng tần suất tối thiểu (mặc định: {DEFAULT_FREQ}).",
    )
    parser.add_argument(
        "--out", type=str, default=str(DEFAULT_OUTPUT),
        help="Thư mục xuất vocab.pkl và vocab.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    captions_path = Path(args.captions)
    output_dir    = Path(args.out)

    print("=" * 60)
    print("BUILDING VOCABULARY")
    print("=" * 60)

    if not captions_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file captions: {captions_path}")

    print(f"\nĐọc captions từ: {captions_path}")
    captions = load_captions(captions_path)
    print(f"Số lượng captions: {len(captions):,}")

    vocab = build_and_save(captions, args.freq, output_dir)

    # Thống kê
    print(f"\nThống kê:")
    print(f"   Total tokens  : {len(vocab):,}")
    print(f"   Freq threshold: {vocab.freq_threshold}")

    # Demo encode / decode
    sample = "a dog is running in the park"
    encoded = vocab.encode(sample)
    decoded = vocab.decode(encoded)
    print(f"\nDemo encode/decode:")
    print(f"   Input  : '{sample}'")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: '{decoded}'")


if __name__ == "__main__":
    main()
