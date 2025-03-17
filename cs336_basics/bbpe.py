import logging
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

import regex as re
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.error(f"{name} took {end - start:.2f}s")


class BBPE:
    def __init__(self, special_tokens: List[str] = []):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)

        self.special_tokens = special_tokens
        self.vocab = dict()
        for i, token in enumerate(self.special_tokens):
            self.vocab[i] = token.encode("utf-8")
        for i in range(256):
            self.vocab[256 + i] = bytes([i])
        self.merges: List[Tuple[bytes, bytes]] = []
        self.special_tokens = special_tokens

    def train(self, input_path: str, vocab_size: int, special_tokens: List[str], progress: bool = False):
        with timer("Reading corpus"):
            with open(input_path) as f:
                corpus = f.read()

        with timer("Pre-tokenizing"):
            chunks = self.pre_tokenize(corpus)

        with timer("Training BPE"):
            vocab = dict()
            for i in range(256):
                vocab[i] = bytes([i])
            merges = []
            num_merges = vocab_size - len(vocab) - len(special_tokens)

            iterator = tqdm(range(num_merges)) if progress else range(num_merges)

            # compute all pair occurences
            occurences = defaultdict(int)
            for chunk in chunks:
                for p1, p2 in zip(chunk, chunk[1:]):
                    occurences[(p1, p2)] += 1

            for merge_idx in iterator:
                # choose most frequence pair, tie breaking rule is lexicographical order
                max_pair = max(occurences, key=lambda x: (occurences[x], vocab[x[0]], vocab[x[1]]))

                # add to vocab and merge
                idx = 256 + merge_idx
                merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
                vocab[idx] = vocab[max_pair[0]] + vocab[max_pair[1]]

                new_chunks = []
                for chunk in chunks:
                    new_chunk = []
                    dec = []
                    inc = []
                    i = 0
                    while i < len(chunk):
                        if i < len(chunk) - 1 and chunk[i] == max_pair[0] and chunk[i + 1] == max_pair[1]:
                            new_chunk.append(idx)
                            # update occurences
                            dec.append(max_pair)
                            if i > 0:
                                inc.append((chunk[i - 1], idx))
                                dec.append((chunk[i - 1], chunk[i]))
                            if i < len(chunk) - 2:
                                inc.append((idx, chunk[i + 2]))
                                dec.append((chunk[i + 1], chunk[i + 2]))
                            i += 2
                        else:
                            new_chunk.append(chunk[i])
                            i += 1

                    for pair in inc:
                        occurences[pair] += 1
                    for pair in dec:
                        occurences[pair] -= 1
                    new_chunks.append(new_chunk)
                chunks = new_chunks
        # add special tokens, starting from 0
        new_vocab = dict()
        for i, token in enumerate(special_tokens):
            new_vocab[i] = token.encode("utf-8")

        for i, token in vocab.items():
            new_vocab[i + len(special_tokens)] = token

        self.vocab = new_vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def pre_tokenize(self, corpus: str) -> List[List[int]]:
        # is corpus a string? or list of strings?
        chunks = re.findall(self.compiled_pattern, corpus)
        chunks = [list(chunk.encode("utf-8")) for chunk in chunks]
        return chunks
