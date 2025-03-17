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
    level=logging.INFO,
)
logger.setLevel(logging.INFO)


@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{name} took {end - start:.2f}s")


class Node:
    """Node class for doubly linked list"""

    def __init__(self, value: int, prev: "Node" = None, next: "Node" = None):
        self.value = value
        self.prev = prev
        self.next = next

    def __repr__(self):
        if self.prev:
            prev = self.prev.value
        else:
            prev = "empty"
        if self.next:
            next = self.next.value
        else:
            next = "empty"
        return f"Node(prev={prev}, value={self.value}, next={next})"

    def delete(self):
        # delete this node, update prev and next
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev

        self.prev = None
        self.next = None


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

            # construct initial index and occurences from chunks
            # index key is pair, value is first node of pair
            # NOTE: invalid node will be NOT deleted for simplicity, this logic will be updated if memory is a concern
            occurences = defaultdict(int)
            index = defaultdict(list)

            for chunk in chunks:
                prev_node = Node(chunk[0])
                for value in chunk[1:]:
                    cur_node = Node(value, prev=prev_node)
                    prev_node.next = cur_node
                    pair = (prev_node.value, cur_node.value)
                    occurences[pair] += 1
                    index[pair].append(prev_node)
                    prev_node = cur_node

            iterator = tqdm(range(num_merges)) if progress else range(num_merges)
            for _ in iterator:
                # choose most frequence pair, tie breaking rule is lexicographical order
                p1, p2 = max(occurences, key=lambda x: (occurences[x], vocab[x[0]], vocab[x[1]]))
                c1 = vocab[p1]
                c2 = vocab[p2]

                # add to vocab and merge
                new_idx = len(vocab)
                merges.append((c1, c2))
                vocab[new_idx] = c1 + c2

                # update index and occurences
                for node in index[(p1, p2)]:
                    node: Node
                    if node.value != p1 or node.next is None or node.next.value != p2:
                        # invalid node, skip
                        continue
                    # update occurences
                    occurences[(p1, p2)] -= 1
                    if node.prev:
                        occurences[(node.prev.value, node.value)] -= 1
                        occurences[(node.prev.value, new_idx)] += 1

                    if node.next.next:
                        occurences[(node.next.value, node.next.next.value)] -= 1
                        occurences[(new_idx, node.next.next.value)] += 1

                    node.next.delete()
                    node.value = new_idx

                    # update index
                    if node.prev:
                        index[(node.prev.value, new_idx)].append(node.prev)
                    if node.next:
                        index[(new_idx, node.next.value)].append(node)
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
        chunks = (match.group() for match in re.finditer(self.pattern, corpus))
        chunks = [chunk.encode("utf-8") for chunk in chunks]
        return chunks


if __name__ == "__main__":
    # Train tokenizer on tiny stories
    tokenizer = BBPE()
    tokenizer.train("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["|endoftext|"], progress=True)
