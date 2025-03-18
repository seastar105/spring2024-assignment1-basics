import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import time

from cs336_basics.bbpe import BBPE

if __name__ == "__main__":
    start = time.perf_counter()
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["|endoftext|"]
    num_proc = 16
    vocab, merges = BBPE.train(input_path, vocab_size, special_tokens, progress=True, num_proc=num_proc)
    end = time.perf_counter()

    elapsed_time = end - start
    print(f"Training on TinyStories took {elapsed_time:.2f}s")

    Path("tokenizers").mkdir(exist_ok=True)
    vocab_path = "tokenizers/tiny_stories_vocab.json"
    merges_path = "tokenizers/tiny_stories_merges.txt"
    vocab = BBPE.export_vocab(vocab)
    merges = BBPE.export_merges(merges)

    max_token = ""
    for token in vocab.values():
        if len(token) > len(max_token):
            max_token = token
    print(f"Max token length: {len(max_token)}, {max_token}")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            merge = " ".join(merge)
            print(merge, file=f)
    print(f"Vocab saved to {vocab_path}")
    print(f"Merges saved to {merges_path}")
