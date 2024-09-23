# llm
Ground-up implementation of LLMs with minimal dependencies


# TODO

- Fix weight initialization
    - MultiHeadAttention
    - Embedding (should embeddings be standard normal?)
- Diagnose FeedForward training after 1 iter
    - Distribution of activations
    - Distribution of gradients
    - Ratio of gradients to activations
    - Should FeedForward have normalization layer before activation?
        # See Karpathy's "Building makemore part3", 1:07:00
        # [Linear -> Norm -> ReLU] * N -> Linear -> Norm - (+identity) -> Relu
- Support `bias` param for Linear layer
- Support dtype
    - Block
    - BlockStack
    - Embedding
    - Transformer
- Support batching
    - MultiHeadAttention
    - Block
    - BlockStack
    - Embedding
    - Transformer
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
- Support on-disk parameter dumping and loading: dump_params(file), load_params(file)
    - Transformer
- Create virtual env and helper scripts
    - Use latest Python version or one just before it
    - Use latest numpy
    - Add requirements files
    - Add test runner
- Rename context_window to context_length
- Data Loader & batcher
- Tokenizer (enc, dec, train)
- Revisit hack in LayerNorm when n_input==2
