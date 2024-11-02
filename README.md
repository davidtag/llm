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


# TODO

- Tokenizer (enc, dec, train)
    - Add unit tests for bpe.pyx
    - Add unit tests for RegexTokenizer
    - Create a Tokenizer class that uses regex-style logic
        - Add runner scripts
            - train model
            - train piece cache: as part of regular runner, or with separate text
        - Registry for model files and ability to load them from disk (include gpt-4 and o-1 encoders)
    - Add support for special tokens
- Diagnose FeedForward training after 1 iter
    - Distribution of activations
    - Distribution of gradients
    - Ratio of gradients to activations
    - Should FeedForward have normalization layer before activation?
        # See Karpathy's "Building makemore part3", 1:07:00
        # [Linear -> Norm -> ReLU] * N -> Linear -> Norm - (+identity) -> Relu
- Support on-disk model dumping and loading (factory method): dump_model(file), load_model(file)
    - Transformer (parameters + settings)
- Create virtual env and helper scripts
    - Use latest Python version or one just before it
    - Use latest numpy
        - add copy=False to all reshape()s
    - Add requirements files
    - Add test runner
- Data Loader & batcher
- Dropout layer for training deeper networks and avoiding overfitting
    - See Karpathy's "building GPT from scratch" video ~1:40:00 and also Transformer/GPT-1 papers
- Revisit hack in LayerNorm when n_input==2
- Explore GPU acceleration via CuPy (dropin) or Numba (JIT compiler for Python)
- Make script to download all data assets:
    https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
- Analyze the embeddings
    - 2D PCA for viz
        - do we see natural groups? whitespace, punctuation, character names
    - Clustering
- Support for linters and type checkers:
    - mypy
    - flake8
    - pylint