import pickle
import pandas as pd
from vocabulary import Vocabularyg

CAPTIONS_FILE  = "/home/codespace/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/results.csv" #thay duong dan
FREQ_THRESHOLD = 3
OUTPUT_PATH    = "vocab.pkl"


if __name__ == "__main__":
    print(" Đọc captions từ:", CAPTIONS_FILE)
    df = pd.read_csv(CAPTIONS_FILE, sep="|")
    df.columns = df.columns.str.strip()

    print("Các cột trong file:", list(df.columns))

    # Tự động nhận diện cột caption
    cap_col = [c for c in df.columns
               if "comment" in c.lower() or "caption" in c.lower()]
    if not cap_col:
        raise ValueError("Không tìm được cột caption. Kiểm tra lại file CSV.")
    cap_col = cap_col[-1]
    print(f"Dùng cột caption: '{cap_col}'")

    captions = df[cap_col].dropna().astype(str).tolist()
    print(f"Tổng số captions: {len(captions)}")

    vocab = Vocabulary()
    vocab.build(captions, FREQ_THRESHOLD)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Đã lưu → {OUTPUT_PATH}")
    print(f"   Ví dụ: 'dog' → {vocab.stoi.get('dog', 'N/A')}")
    print(f"   Ví dụ: 'man' → {vocab.stoi.get('man', 'N/A')}")
