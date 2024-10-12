import sys
import time
import pdb

from llm.tokenizers.stdtoken import TokenPair


NUM_ITEMS = 10_000_000


def _test_tuple_dict():
    d = {}
    for i in range(NUM_ITEMS):
        pair = (i, i + 1)
        d[pair] = 1
    return d


def _test_token_pair_dict():
    d = {}
    for i in range(NUM_ITEMS):
        pair = TokenPair(i, i + 1)
        d[pair] = 1
    return d


def main():
    """Entrypoint."""
    start = time.monotonic()
    d = _test_tuple_dict()
    end = time.monotonic()
    duration = end - start
    size = sys.getsizeof(d)
    print(f"{duration=:.3f}, {size=:,}")
    pdb.set_trace()


if __name__ == "__main__":
    main()
