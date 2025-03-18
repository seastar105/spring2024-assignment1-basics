import time

import numpy as np

from cs336_basics.bbpe import Tokenizer

DOC_SEPARATOR = "<|endoftext|>"
DATA_PATH = {
    "open_webtext": {
        "train": "data/owt_train.txt",
        "valid": "data/owt_valid.txt",
    },
    "tiny_stories": {
        "train": "data/TinyStoriesV2-GPT4-train.txt",
        "valid": "data/TinyStoriesV2-GPT4-valid.txt",
    },
}


def sample_open_webtext():
    READ_SIZE = 100 * 1024 * 1024
    txt_path = DATA_PATH["open_webtext"]["train"]
    with open(txt_path) as f:
        corpus = f.read(READ_SIZE)
    docs = corpus.split(DOC_SEPARATOR)
    docs = np.random.choice(docs, 100, replace=False)
    docs = [doc.strip() + DOC_SEPARATOR for doc in docs]
    return docs


def sample_tiny_stories():
    READ_SIZE = 100 * 1024 * 1024
    txt_path = DATA_PATH["tiny_stories"]["train"]
    with open(txt_path) as f:
        corpus = f.read(READ_SIZE)
    docs = corpus.split(DOC_SEPARATOR)
    docs = np.random.choice(docs, 100, replace=False)
    docs = [doc.strip() + DOC_SEPARATOR for doc in docs]
    return docs


def compression_exp():
    owt_tokenizer = Tokenizer.from_pretrained("open_webtext")
    tiny_tokenizer = Tokenizer.from_pretrained("tiny_stories")

    bytes_per_token = 2  # Assume 2 bytes per token
    owt_docs = sample_open_webtext()
    owt_bytes = sum([len(doc.encode("utf-8")) for doc in owt_docs])

    owt_tokens = owt_tokenizer.encode("".join(owt_docs))
    owt_token_bytes = len(owt_tokens) * bytes_per_token
    owt_owt_ratio = owt_token_bytes / owt_bytes

    tiny_docs = sample_tiny_stories()
    tiny_bytes = sum([len(doc.encode("utf-8")) for doc in tiny_docs])

    tiny_tokens = owt_tokenizer.encode("".join(tiny_docs))
    tiny_token_bytes = len(tiny_tokens) * bytes_per_token
    tiny_owt_ratio = tiny_token_bytes / tiny_bytes

    print(f"OpenWebText Tokenizer compression ratio on OpenWebText: {owt_owt_ratio * 100:.2f}")
    print(f"OpenWebText Tokenizer compression ratio on TinyStories: {tiny_owt_ratio * 100:.2f}")

    owt_tokens = tiny_tokenizer.encode("".join(owt_docs))
    owt_token_bytes = len(owt_tokens) * bytes_per_token
    owt_tiny_ratio = owt_token_bytes / owt_bytes

    tiny_tokens = tiny_tokenizer.encode("".join(tiny_docs))
    tiny_token_bytes = len(tiny_tokens) * bytes_per_token
    tiny_tiny_ratio = tiny_token_bytes / tiny_bytes

    print(f"TinyStories Tokenizer compression ratio on OpenWebText: {owt_tiny_ratio * 100:.2f}")
    print(f"TinyStories Tokenizer compression ratio on TinyStories: {tiny_tiny_ratio * 100:.2f}")


def get_throuput():
    owt_tokenizer = Tokenizer.from_pretrained("open_webtext")
    tiny_tokenizer = Tokenizer.from_pretrained("tiny_stories")

    n_bytes = 2**30

    with open(DATA_PATH["open_webtext"]["train"], "rb") as f:
        owt_bytes = f.read(n_bytes)
    owt_str = owt_bytes.decode("utf-8")

    def chunk_iterator(s, chunk_size):
        offset = 0
        while offset < len(s):
            chunk = s[offset : offset + chunk_size]
            offset += chunk_size
            yield chunk

    start = time.perf_counter()
    owt_tokens = [token for token in owt_tokenizer.encode_iterable(chunk_iterator(owt_str, 4 * 1024), num_proc=16)]
    print(len(owt_tokens))
    end = time.perf_counter()
    print(f"OpenWebText Tokenizer throughput: {n_bytes / (end - start) / 1024 / 1024:.2f} MB/s")

    with open(DATA_PATH["tiny_stories"]["train"], "rb") as f:
        tiny_bytes = f.read(n_bytes)
    tiny_str = tiny_bytes.decode("utf-8")

    start = time.perf_counter()
    tiny_tokens = [token for token in tiny_tokenizer.encode_iterable(chunk_iterator(tiny_str, 4 * 1024), num_proc=16)]
    print(len(tiny_tokens))
    end = time.perf_counter()
    print(f"TinyStories Tokenizer throughput: {n_bytes / (end - start) / 1024 / 1024:.2f} MB/s")


if __name__ == "__main__":
    compression_exp()
    get_throuput()
