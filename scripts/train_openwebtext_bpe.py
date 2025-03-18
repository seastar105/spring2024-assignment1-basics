import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import time

from cs336_basics.bbpe import Tokenizer

if __name__ == "__main__":
    start = time.perf_counter()
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["|endoftext|"]
    num_proc = 16
    vocab, merges = Tokenizer.train(input_path, vocab_size, special_tokens, progress=True, num_proc=num_proc)
    end = time.perf_counter()

    elapsed_time = end - start
    print(f"Training on OpenWebText took {elapsed_time:.2f}s")

    Path("tokenizers").mkdir(exist_ok=True)
    vocab_path = "tokenizers/open_webtext_vocab.json"
    merges_path = "tokenizers/open_webtext_merges.txt"
    vocab = Tokenizer.export_vocab(vocab)
    merges = Tokenizer.export_merges(merges)

    max_token = ""
    for token in vocab.values():
        if len(token) > len(max_token):
            max_token = token
    print(f"Max token length: {len(max_token)}, {max_token}")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            merge = " ".join(merge)
            print(merge, file=f)
    print(f"Vocab saved to {vocab_path}")
    print(f"Merges saved to {merges_path}")
