# llm
Ground-up implementation of LLMs with `NumPy` and `regex` as the only dependencies.

The purpose of this repo is educational - explore the main concepts of LLMs without relying on
3rd party packages for the heavy lifting. In particular, the computation of gradients (i.e., the
"backward pass") is done manually, instead of relying on autograd systems, like the ones in
PyTorch or Tensorflow. The Byte-Pair Encoding (BPE) algorithm is also manually implemented with
no dependencies other than the `regex` library, and supports training.


## Quick Start

1. Setup a virtual environment and activate it. Code has been tested on Python 3.12.
```shell
python3.12 -m venv venv
source venv/bin/activate
make install_requirements
```

2. Compile Cython extension modules
```shell
make python_extensions
```

3. Run tests
```shell
make test
```

4. Download data
```shell
make download_text
```

5. Train the BPE tokenizer:
```shell
PYTHONPATH=. python scripts/train_tokenizer.py -y -n default_10k
```

6. Tokenize the data splits for use in training:
```shell
PYTHONPATH=. python scripts/tokenize_splits.py -n default_10k
```

7. Train the Transformer model:
```shell
PYTHONPATH=. python scripts/train_model.py -y -n v1 -t default_10k -bs 8 -nb 100
PYTHONPATH=. python scripts/train_model.py -y -n v2 -t default_10k -bs 8 -nb 100 -s v1 -c 100
```

8. Generate output from the model:
```shell
PYTHONPATH=. python scripts/generate_text.py -t default_10k -n v2 -c 100
```

## TODO

P0:
- Support for linters and type checkers:
    - mypy
    - flake8
    - pylint
- Unit tests
    - bpe.pyx
    - RegexTokenizer
    - New modules in llm/data
- RegexTokenizer
    - Add support for special tokens
- Analyze the embeddings
    - 2D PCA for viz
        - do we see natural groups? whitespace, punctuation, character names
    - Clustering

P1:
- Revisit hack in LayerNorm when n_input==2
- Should FeedForward have normalization layer before activation?
    - See Karpathy's "Building makemore part3", 1:07:00
    - [Linear -> Norm -> ReLU] * N -> Linear -> Norm - (+identity) -> Relu
- Dropout layer for training deeper networks and avoiding overfitting
    - See Karpathy's "building GPT from scratch" video ~1:40:00 and also Transformer/GPT-1 papers
- Explore GPU acceleration via CuPy (dropin) or Numba (JIT compiler for Python)
