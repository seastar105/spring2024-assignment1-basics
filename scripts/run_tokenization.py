import gc
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
NUM_PROC = 16


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


def get_throughput():
    owt_tokenizer = Tokenizer.from_pretrained("open_webtext")
    tiny_tokenizer = Tokenizer.from_pretrained("tiny_stories")

    n_bytes = 2**27  # 128 MB

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
    owt_tokens = [
        token for token in owt_tokenizer.encode_iterable(chunk_iterator(owt_str, 32 * 1024), num_proc=NUM_PROC)
    ]
    print(len(owt_tokens))
    end = time.perf_counter()
    print(f"OpenWebText Tokenizer throughput: {n_bytes / (end - start) / 1024 / 1024:.2f} MB/s")

    with open(DATA_PATH["tiny_stories"]["train"], "rb") as f:
        tiny_bytes = f.read(n_bytes)
    tiny_str = tiny_bytes.decode("utf-8")

    start = time.perf_counter()
    tiny_tokens = [
        token for token in tiny_tokenizer.encode_iterable(chunk_iterator(tiny_str, 32 * 1024), num_proc=NUM_PROC)
    ]
    print(len(tiny_tokens))
    end = time.perf_counter()
    print(f"TinyStories Tokenizer throughput: {n_bytes / (end - start) / 1024 / 1024:.2f} MB/s")


def chunk_read(path, chunk_size):
    SKIP_CHUNK = 0
    tot = 0
    with open(path) as f:
        while True:
            chunk = f.read(chunk_size)
            tot += len(chunk)
            if tot <= SKIP_CHUNK * chunk_size:
                continue
            if not chunk:
                break
            yield chunk


def tokenize_tiny_stories():
    tiny_tokenizer = Tokenizer.from_pretrained("tiny_stories")

    input_path = DATA_PATH["tiny_stories"]["train"]
    tokens = []
    for token in tiny_tokenizer.encode_iterable(chunk_read(input_path, 256 * 1024), num_proc=NUM_PROC, progress=True):
        tokens.append(token)
    tokens = np.array(tokens).astype(np.uint16)
    np.save("data/tiny_stories_train_tokens.npy", tokens)

    input_path = DATA_PATH["tiny_stories"]["valid"]
    tokens = []
    for token in tiny_tokenizer.encode_iterable(chunk_read(input_path, 256 * 1024), num_proc=NUM_PROC, progress=True):
        tokens.append(token)
    tokens = np.array(tokens).astype(np.uint16)
    np.save("data/tiny_stories_valid_tokens.npy", tokens)


def tokenize_open_webtext():
    owt_tokenizer = Tokenizer.from_pretrained("open_webtext")

    input_path = DATA_PATH["open_webtext"]["train"]
    np_tokens = np.array([]).astype(np.uint16)
    tokens = []
    for token in owt_tokenizer.encode_iterable(
        chunk_read(input_path, 10 * 1024 * 1024), num_proc=NUM_PROC, progress=True
    ):
        tokens.append(token)
        if len(tokens) >= 10**8:
            tokens = np.array(tokens).astype(np.uint16)
            np_tokens = np.concatenate([np_tokens, tokens])
            tokens = []
            gc.collect()
    if len(tokens) > 0:
        tokens = np.array(tokens).astype(np.uint16)
        np_tokens = np.concatenate([np_tokens, tokens])
    np.save("data/open_webtext_train_tokens.npy", np_tokens)

    input_path = DATA_PATH["open_webtext"]["valid"]
    tokens = []
    for token in owt_tokenizer.encode_iterable(
        chunk_read(input_path, 10 * 1024 * 1024), num_proc=NUM_PROC, progress=True
    ):
        tokens.append(token)
    tokens = np.array(tokens).astype(np.uint16)
    np.save("data/open_webtext_valid_tokens.npy", tokens)


if __name__ == "__main__":
    compression_exp()
    get_throughput()

    tokenize_tiny_stories()
    tokenize_open_webtext()
