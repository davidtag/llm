"""Sample training and inference for the RegexTokenizer implementation."""

from pathlib import Path

from llm.tokenizers.benchmarks.profile import Profile
from llm.tokenizers.regex_tokenizer import RegexTokenizer


def _load_text_file(file_path: str) -> str:
    with open(file_path, mode="r", encoding="utf-8") as f:
        data = f.read()
    return data


def _load_train_test_split(file_path: str, val_length: int = 2 * 1024 * 1024) -> tuple[str, str]:
    data = _load_text_file(file_path)
    train = data[:-val_length]
    val = data[-val_length:]
    return train, val


def train():
    train_text, _ = _load_train_test_split(file_path="data/blob/t8.shakespeare.txt")
    split_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    tokenizer = RegexTokenizer.train(
        text=train_text,
        split_pattern=split_pattern,
        num_merges=10000,
    )
    print(len(tokenizer.trained_cache))
    new_tokenizer = tokenizer.train_piece_cache(
        text=train_text,
        num_extra_pieces=10000,
    )
    print(len(new_tokenizer.trained_cache))
    new_tokenizer.save(Path("out/"))


def test():
    tokenizer = RegexTokenizer.load(Path("out/"))
    print(len(tokenizer.trained_cache), len(tokenizer.runtime_cache))

    print()
    print("-- Basic Test ------------------------------------------------------")
    _input = "hello world!!!? (안녕하세요!) lol123 😉"
    print(f"{_input=}")
    tokens = tokenizer.encode(_input)
    print(f"{tokens=}")
    _output = tokenizer.decode(tokens)
    print(f"{_output=}")
    assert _input == _output

    print()
    print("-- Large Test ------------------------------------------------------")
    _, val_text = _load_train_test_split(file_path="data/blob/t8.shakespeare.txt")
    with Profile() as prof:
        tokens = tokenizer.encode(val_text)
    print(f"  val_text={len(val_text):,} -> tokens={len(tokens):,}: elapsed={prof.milliseconds_formatted}")

    with Profile() as prof:
        val_out = tokenizer.decode(tokens)
    print(f"  tokens={len(tokens):,} -> val_out={len(val_out):,}: elapsed={prof.milliseconds_formatted}")
    assert len(val_out) == len(val_text)
    assert val_out == val_text

    print()
    print(len(tokenizer.trained_cache), len(tokenizer.runtime_cache))


if __name__ == "__main__":
    test()
