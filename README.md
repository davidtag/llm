# llm
Ground-up implementation of LLMs with NumPy as the only dependency.

The purpose of this repo is educational - explore the main concepts of LLMs without relying on
3rd party packages for the heavy lifting. In particular, the computation of gradients (i.e., the
"backward pass") is done manually, instead of relying on autograd systems, like the ones in
PyTorch or Tensorflow.

# TODO

- Tokenizer (enc, dec, train)
    - Implemente a BaseTokenizer class with:
        - Data types for merges / vocab
        - Helpers for byte_pair_encode() and decode() and decode_bytes()
    - Create a Tokenizer class that uses regex-style logic
        - Ability to train and export a model file (have a runner script)
        - Ability to train a piece cache -- as part of regular runner, or with separate text
        - Registry for model files and ability to load them from disk
    - Add support for special tokens
- Develop a simple Embedding class and refactor TextEmbedding to use 2 of them (token + position)
- Diagnose FeedForward training after 1 iter
    - Distribution of activations
    - Distribution of gradients
    - Ratio of gradients to activations
    - Should FeedForward have normalization layer before activation?
        # See Karpathy's "Building makemore part3", 1:07:00
        # [Linear -> Norm -> ReLU] * N -> Linear -> Norm - (+identity) -> Relu
- Support in-memory parameter dumping and loading: get_params() -> dict, set_params(dict)
    - Linear
    - FeedForward
    - LayerNorm
    - MultiHeadAttention
    - Block
    - BlockStack
    - Embedding
    - Transformer
    # {
    #   "layer_1": {"w": ..., "b": ...},
    #   "norm_1": {"gamma": ..., "beta": ...}
    # }
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
