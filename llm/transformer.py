import numpy as np


def convert_vector_to_column_matrix(v: np.ndarray) -> np.ndarray:
    """Convert a N-dimensional vector to a Nx1 matrix."""
    assert len(v.shape) == 1, "Input must be a vector"
    m = np.expand_dims(v, axis=1)
    return m


def convert_vector_to_row_matrix(v: np.ndarray) -> np.ndarray:
    """Convert a N-dimensional vector to a 1xN matrix."""
    assert len(v.shape) == 1, "Input must be a vector"
    m = np.expand_dims(v, axis=0)
    return m


def log_sum_exp(x: np.ndarray) -> np.ndarray:
    """Compute the log of the sum of exponentials of `x` along the last dimension.

    Args:
        x: Real-valued array of dimension (*, N)

    Returns:
        y: Real-valued array of dimension (*, 1)
    """
    dims = len(x.shape)
    max_x = np.max(x, axis=dims - 1, keepdims=True)
    x_shifted = x - max_x
    exp_x_shifted = np.exp(x_shifted)
    sum_exp_x_shifted = np.sum(exp_x_shifted, axis=dims - 1, keepdims=True)
    lse_shifted = np.log(sum_exp_x_shifted)
    out = lse_shifted + max_x
    return out


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Compute the log of the softmax of `x` along the last dimension.

    Args:
        x: Real-valued array of dimension (*, N)

    Returns:
        y: Real-valued array of dimension (*, 1)
    """
    return x - log_sum_exp(x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute the softmax of `x` along the last dimension.

    Args:
        x: Real-valued array of dimension (*, N)

    Returns:
        y: Collection of probability distributions of dimension (*, N)
    """
    dims = len(x.shape)
    max_x = np.max(x, axis=dims - 1, keepdims=True)
    x_shifted = x - max_x
    exp_x_shifted = np.exp(x_shifted)
    sum_exp_x_shifted = np.sum(exp_x_shifted, axis=dims - 1, keepdims=True)
    out = exp_x_shifted / sum_exp_x_shifted
    return out


def softmax_vector(x: np.ndarray) -> np.ndarray:
    """Convert a real-valued vector to a probability distribution of the same dimension.

    Args:
        x: Real-valued vector of dimension N

    Returns:
        y: Probability distribution of dimension N
    """
    assert len(x.shape) == 1, "Input must be a vector"
    x_max = np.max(x)
    x_shifted = x - x_max
    x_shifted_exp = np.exp(x_shifted)
    total = np.sum(x_shifted_exp)
    out = x_shifted_exp / total
    return out


def softmax_jacobian(x: np.ndarray) -> np.ndarray:
    """Evaluate the Jacobian of the softmax() function at x.

    Args:
        x: Real-valued vector of dimension N

    Returns:
        J: Jacobian of the softmax evaluated at X. Dimension NxN.
    """
    assert len(x.shape) == 1, "Input must be a vector"
    N = x.shape[0]
    I = np.eye(N)
    s = softmax_vector(x)
    S_column = convert_vector_to_column_matrix(s)
    S_row = convert_vector_to_row_matrix(s)
    out = S_column * (I - S_row)
    return out


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> int:
    """Compute the scaled dot-product attention for a set of queries, Q, and key-value matrices K and V.

    Args:
        Q: query vectors arranged row-wise. Dimension N x d_k
        K: query vectors arranged row-wise. Dimension M x d_k
        V: value vectors arranged row-wise. Dimension M x d_v

    Returns:
        A: parallel weighted average of the value vectors. Dimension N x d_v
    """
    N, d_k = Q.shape
    M, d_k2 = K.shape
    assert d_k2 == d_k, "Query and Key matrix dimensions aren't compatible"
    M2, d_v = V.shape
    assert M2 == M, "Key and Value matrix dimensions aren't compatible"
    scale = np.sqrt(d_k)
    logits = np.matmul(Q, np.transpose(K)) / scale
    assert logits.shape == (N, M)
    W = softmax(logits)
    assert W.shape == (N, M)
    out = np.matmul(W, V)
    assert out.shape == (N, d_v)
    cache = {}
    cache["softmax"] = W
    return out, cache


class AdamOptimizer(object):
    """."""

    def __init__(
        self,
        w: np.ndarray,
        beta_1: float = 0.9,
        beta_2: float = 0.98,
        epsilon: float = 1e-9,
    ) -> None:
        """Initialize the optimizer."""
        self.w_shape = w.shape
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.t = 0
        self.m = np.zeros_like(w)
        self.v = np.zeros_like(w)

    def get_update(self, dw: np.ndarray) -> np.ndarray:
        """."""
        self.t = self.t + 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * dw
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(dw)
        m_hat = self.m / (1 - np.power(self.beta_1, self.t))
        v_hat = self.v / (1 - np.power(self.beta_2, self.t))
        dw_hat = (m_hat) / (np.sqrt(v_hat) + self.epsilon)
        return dw_hat


class MultiHeadAttention(object):
    """A single multi-head attention sub-layer of a Transformer block."""

    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 8,
        enable_grad: bool = True,
    ):
        """Initialize the layer."""
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.enable_grad = enable_grad
        self.cache = {}
        self.cache["head_cache"] = [{} for _ in range(h)]

        self.w_q = np.random.normal(loc=0, scale=np.sqrt(2 / d_model), size=(h, d_model, d_k))
        self.w_k = np.random.normal(loc=0, scale=np.sqrt(2 / d_model), size=(h, d_model, d_k))
        self.w_v = np.random.normal(loc=0, scale=np.sqrt(2 / d_model), size=(h, d_model, d_v))
        self.w_o = np.random.normal(loc=0, scale=np.sqrt(2 / d_model), size=(h * d_v, d_model))

        self.w_q_opt = AdamOptimizer(self.w_q)
        self.w_k_opt = AdamOptimizer(self.w_k)
        self.w_v_opt = AdamOptimizer(self.w_v)
        self.w_o_opt = AdamOptimizer(self.w_o)

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return (
            self.h * self.d_model * self.d_k
            + self.h * self.d_model * self.d_k
            + self.h * self.d_model * self.d_v
            + self.h * self.d_v * self.d_model
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input.

        Input and output are both N x d_model.
        """
        assert len(x.shape) == 2 and x.shape[1] == self.d_model

        Q = np.matmul(x, self.w_q)  # size = (h, N, d_k)
        K = np.matmul(x, self.w_k)  # size = (h, N, d_k)
        V = np.matmul(x, self.w_v)  # size = (h, N, d_v)

        heads = []
        for i in range(self.h):
            # TODO(dtag): parallelize?
            head_i, cache_i = scaled_dot_product_attention(Q[i], K[i], V[i])  # size = (N, d_v)
            if self.enable_grad:
                self.cache["head_cache"][i] = cache_i
            heads.append(head_i)
        concat = np.hstack(heads)  # size = (N, h * d_v)

        out = np.matmul(concat, self.w_o)  # size = (N, d_model)

        if self.enable_grad:
            self.cache["x"] = x
            self.cache["Q"] = Q
            self.cache["K"] = K
            self.cache["V"] = V
            self.cache["concat"] = concat

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x = self.cache["x"]
        Q = self.cache["Q"]
        K = self.cache["K"]
        V = self.cache["V"]
        concat = self.cache["concat"]

        dw_o = np.matmul(np.transpose(concat), dout)  # (h * d_v, d_model)
        dconcat = np.matmul(dout, np.transpose(self.w_o))  # (N, h * d_v)

        dQ_chunks = []
        dK_chunks = []
        dV_chunks = []

        dheads = np.hsplit(dconcat, indices_or_sections=self.h)
        for i, dhead_i in enumerate(dheads):
            # dhead_i.size = (N, d_v)
            cache_i = self.cache["head_cache"][i]

            dV_i = np.matmul(np.transpose(cache_i["softmax"]), dhead_i)
            dsoftmax_i = np.matmul(dhead_i, np.transpose(V[i]))

            # TODO: double-check the next 3 lines
            dlogits_i = cache_i["softmax"] * (
                dsoftmax_i - np.sum(dsoftmax_i * cache_i["softmax"], axis=1, keepdims=True)
            )
            dQ_i = (1 / np.sqrt(self.d_k)) * np.matmul(dlogits_i, K[i])
            dK_i = (1 / np.sqrt(self.d_k)) * np.matmul(np.transpose(Q[i]), dlogits_i).transpose()

            dQ_chunks.append(dQ_i)
            dK_chunks.append(dK_i)
            dV_chunks.append(dV_i)

        dQ = np.array(dQ_chunks)
        dK = np.array(dK_chunks)
        dV = np.array(dV_chunks)

        dw_q = np.matmul(np.transpose(x), dQ)
        dw_k = np.matmul(np.transpose(x), dK)
        dw_v = np.matmul(np.transpose(x), dV)

        # TODO: Double check this
        # TODO: look into axes parameter of transpoe to avoid list comprehension
        dx = (
            sum([np.matmul(dQ[i], np.transpose(self.w_q[i])) for i in range(self.h)])
            + sum([np.matmul(dK[i], np.transpose(self.w_k[i])) for i in range(self.h)])
            + sum([np.matmul(dV[i], np.transpose(self.w_v[i])) for i in range(self.h)])
        )

        self.cache["dx"] = dx
        self.cache["dw_q"] = dw_q
        self.cache["dw_k"] = dw_k
        self.cache["dw_v"] = dw_v
        self.cache["dw_o"] = dw_o

    def step(self, alpha: float) -> None:
        """Take a gradient step based on the most recent backward pass."""
        assert self.enable_grad, "Cannot take a gradient step with enable_grad=False"

        # self.w_q -= alpha * self.cache["dw_q"]
        # self.w_k -= alpha * self.cache["dw_k"]
        # self.w_v -= alpha * self.cache["dw_v"]
        # self.w_o -= alpha * self.cache["dw_o"]

        self.w_q -= alpha * self.w_q_opt.get_update(self.cache["dw_q"])
        self.w_k -= alpha * self.w_k_opt.get_update(self.cache["dw_k"])
        self.w_v -= alpha * self.w_v_opt.get_update(self.cache["dw_v"])
        self.w_o -= alpha * self.w_o_opt.get_update(self.cache["dw_o"])


class Linear(object):
    """Implements a linear layer with a weight matrix and bias."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        enable_grad: bool = True,
    ):
        """Initialize the layer."""
        self.n_input = n_input
        self.n_output = n_output
        self.enable_grad = enable_grad
        self.cache = {}

        self.w = np.random.normal(loc=0, scale=np.sqrt(2 / n_input), size=(n_input, n_output))
        self.b = np.random.normal(loc=0, scale=np.sqrt(2 / n_input), size=(1, n_output))

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.n_input * self.n_output + 1 * self.n_output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.shape[-1] == self.n_input

        if self.enable_grad:
            self.cache["x"] = x

        out = np.matmul(x, self.w) + self.b

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        x = self.cache["x"]

        dims = len(dout.shape)

        db = np.sum(dout, axis=tuple(np.arange(dims - 1)), keepdims=True)
        dw = np.matmul(np.transpose(x), dout)
        dx = np.matmul(dout, np.transpose(self.w))

        self.cache["dx"] = dx
        self.cache["dw"] = dw
        self.cache["db"] = db

    def step(self, alpha: float) -> None:
        """Take a gradient step based on the most recent backward pass."""
        assert self.enable_grad, "Cannot take a gradient step with enable_grad=False"

        self.w -= alpha * self.cache["dw"]
        self.b -= alpha * self.cache["db"]


def relu(x: np.ndarray) -> np.ndarray:
    """Compute the element-wise recitified linear unit."""
    return np.maximum(0, x)


class FeedForward(object):
    """A 2-layer feed-forward network with ReLU activation."""

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        enable_grad: bool = True,
    ):
        """Initialize the layer."""
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.enable_grad = enable_grad
        self.cache = {}

        self.layer_1 = Linear(n_input=n_input, n_output=n_hidden, enable_grad=enable_grad)
        self.layer_2 = Linear(n_input=n_hidden, n_output=n_output, enable_grad=enable_grad)

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return self.layer_1.n_params + self.layer_2.n_params

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.shape[-1] == self.n_input

        h = self.layer_1.forward(x)
        a = relu(h)
        out = self.layer_2.forward(a)

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"

        self.layer_2.backward(dout)
        da = self.layer_2.cache["dx"]
        a = self.layer_2.cache["x"]
        dh = da * (a > 0)
        self.layer_1.backward(dh)

        self.cache["dx"] = self.layer_1.cache["dx"]

    def step(self, alpha: float) -> None:
        """Take a gradient step based on the most recent backward pass."""
        assert self.enable_grad, "Cannot take a gradient step with enable_grad=False"

        self.layer_2.step(alpha=alpha)
        self.layer_1.step(alpha=alpha)


class LayerNorm(object):
    """Implements layer normalization."""

    def __init__(
        self,
        n_dims: int,
        enable_grad: bool = True,
    ):
        """Initialize the layer."""
        self.n_dims = n_dims
        self.enable_grad = enable_grad
        self.cache = {}

        self.gamma = np.random.normal(loc=0, scale=np.sqrt(2 / n_dims), size=(1, n_dims))
        self.beta = np.random.normal(loc=0, scale=np.sqrt(2 / n_dims), size=(1, n_dims))

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return 1 * self.n_dims + 1 * self.n_dims

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        assert x.shape[-1] == self.n_dims
        n_dims = len(x.shape)  # todo: inconsistent with naming in constructor
        layer_dim = x.shape[-1]

        x_mean = np.mean(x, axis=n_dims - 1, keepdims=True)
        x_var = np.mean(np.square(x - x_mean), axis=n_dims - 1, keepdims=True)
        x_std = np.sqrt(x_var)
        if layer_dim == 2:  # n_dims == 2
            # need random noise if layer_dim == 2 to avoid z values all being exactly 1 or -1
            x_std += 5 * np.random.random(size=(*x.shape[:-1], 1))
        z = (x - x_mean) / x_std
        out = z * self.gamma + self.beta

        if self.enable_grad:
            self.cache["z"] = z
            self.cache["x_std"] = x_std

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        z = self.cache["z"]
        x_std = self.cache["x_std"]

        dgamma = np.sum(dout * z, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        dx = (
            (
                self.n_dims * dout * self.gamma
                - np.matmul(dout, np.transpose(self.gamma))
                - z * np.matmul(dout * z, np.transpose(self.gamma))
            )
            / self.n_dims
            / x_std
        )

        self.cache["dx"] = dx
        self.cache["dgamma"] = dgamma
        self.cache["dbeta"] = dbeta

    def step(self, alpha: float) -> None:
        """Take a gradient step based on the most recent backward pass."""
        assert self.enable_grad, "Cannot take a gradient step with enable_grad=False"

        self.gamma -= alpha * self.cache["dgamma"]
        self.beta -= alpha * self.cache["dbeta"]


class Block(object):
    """A single block in a Transformer architecture."""

    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 8,
        d_ff: int = 2048,
        enable_grad: bool = True,
    ) -> None:
        """Initialize the block."""
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.enable_grad = enable_grad
        self.cache = {}

        self.sublayer_1 = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, enable_grad=enable_grad)
        self.norm_1 = LayerNorm(n_dims=d_model, enable_grad=enable_grad)
        self.sublayer_2 = FeedForward(
            n_input=d_model,
            n_hidden=d_ff,
            n_output=d_model,
            enable_grad=enable_grad,
        )
        self.norm_2 = LayerNorm(n_dims=d_model, enable_grad=enable_grad)

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return (
            self.sublayer_1.n_params + self.norm_1.n_params + self.sublayer_2.n_params + self.norm_2.n_params
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        h1 = self.sublayer_1.forward(x)
        a1 = x + h1
        o1 = self.norm_1.forward(a1)

        h2 = self.sublayer_2.forward(o1)
        a2 = o1 + h2
        o2 = self.norm_2.forward(a2)

        return o2

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"

        do2 = dout
        self.norm_2.backward(do2)
        da2 = self.norm_2.cache["dx"]
        dh2 = da2
        self.sublayer_2.backward(dh2)

        do1 = da2 + self.sublayer_2.cache["dx"]
        self.norm_1.backward(do1)
        da1 = self.norm_1.cache["dx"]
        dh1 = da1
        self.sublayer_1.backward(dh1)

        dx = da1 + self.sublayer_1.cache["dx"]
        self.cache["dx"] = dx

    def step(self, alpha: float) -> None:
        """Take a gradient step based on the most recent backward pass."""
        assert self.enable_grad, "Cannot take a gradient step with enable_grad=False"

        self.norm_2.step(alpha=alpha)
        self.sublayer_2.step(alpha=alpha)
        self.norm_1.step(alpha=alpha)
        self.sublayer_1.step(alpha=alpha)


class BlockStack(object):
    """A stack of block in a Transformer architecture."""

    def __init__(
        self,
        n_blocks: int,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 8,
        d_ff: int = 2048,
        enable_grad: bool = True,
    ) -> None:
        """Initialize the block."""
        assert n_blocks >= 1
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.enable_grad = enable_grad
        self.cache = {}

        self.blocks = [
            Block(
                d_model=d_model,
                d_k=d_k,
                d_v=d_v,
                h=h,
                d_ff=d_ff,
                enable_grad=enable_grad,
            )
            for _ in range(n_blocks)
        ]

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return sum([block.n_params for block in self.blocks])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the layer output for a given input."""
        out = x

        for block in self.blocks:
            out = block.forward(out)

        return out

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"

        cur_dout = dout

        for block in reversed(self.blocks):
            block.backward(cur_dout)
            cur_dout = block.cache["dx"]

        self.cache["dx"] = cur_dout

    def step(self, alpha: float) -> None:
        """Take a gradient step based on the most recent backward pass."""
        assert self.enable_grad, "Cannot take a gradient step with enable_grad=False"

        for block in self.blocks:
            block.step(alpha=alpha)


class Transformer(object):
    """."""

    def __init__(
        self,
        vocab_size: int = 3,
        n_encoding: int = 6,
        n_decoding: int = 6,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        h: int = 8,
        d_ff: int = 2_048,
        enable_grad: bool = True,
    ) -> None:
        """Initialize the model."""
        self.vocab_size = vocab_size
        self.n_encoding = n_encoding
        self.n_decoding = n_decoding
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.enable_grad = enable_grad
        self.cache = {"dembedding_matrix": np.zeros(shape=(vocab_size, d_model))}

        self.embedding_matrix = np.random.normal(
            loc=(1 / d_model),
            scale=np.sqrt(2 / d_model),
            size=(vocab_size, d_model),
        )
        self.encoder = BlockStack(
            n_blocks=n_encoding,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            d_ff=d_ff,
            enable_grad=enable_grad,
        )
        self.decoder = BlockStack(
            n_blocks=n_decoding,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            d_ff=d_ff,
            enable_grad=enable_grad,
        )
        self.pre_softmax = Linear(n_input=d_model, n_output=vocab_size)

    @property
    def n_params(self) -> int:
        """The number of parameters in the layer."""
        return (
            self.vocab_size * self.d_model
            + self.encoder.n_params
            + self.decoder.n_params
            + self.pre_softmax.n_params
        )

    def forward(self, input_sequence: np.ndarray) -> np.ndarray:
        """Compute the logits for a given input sequence."""
        assert len(input_sequence.shape) == 1
        n_tokens = input_sequence.shape[0]

        embeddings = self.embedding_matrix[input_sequence]  # (n_tokens, d_model)
        assert embeddings.shape == (n_tokens, self.d_model)

        encoding = self.encoder.forward(embeddings)
        decoding = self.decoder.forward(encoding)

        logits = self.pre_softmax.forward(decoding)

        if self.enable_grad:
            self.cache["input_sequence"] = input_sequence

        return logits

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """Generate a probability distribution for a given input sequence."""
        logits = self.forward(input_sequence)
        probabilities = softmax(logits)
        return probabilities

    def backward(self, dout: np.ndarray) -> None:
        """Compute the layer gradients given the upstream gradient."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"

        dlogits = dout

        self.pre_softmax.backward(dlogits)
        ddecoding = self.pre_softmax.cache["dx"]

        self.decoder.backward(ddecoding)
        dencoding = self.decoder.cache["dx"]

        self.encoder.backward(dencoding)
        dembeddings = self.encoder.cache["dx"]

        input_sequence = self.cache["input_sequence"]
        dembedding_matrix = self.cache["dembedding_matrix"]
        dembedding_matrix = 0

        for i, row in enumerate(dembeddings):
            self.cache["dembedding_matrix"][input_sequence[i]] += row

    def step(self, alpha: float) -> None:
        """Take a gradient step based on the most recent backward pass."""
        assert self.enable_grad, "Cannot take a gradient step with enable_grad=False"

        self.pre_softmax.step(alpha=alpha)
        self.decoder.step(alpha=alpha)
        self.encoder.step(alpha=alpha)
        self.embedding_matrix -= alpha * self.cache["dembedding_matrix"]


class CrossEntropyLoss(object):
    """Implements the cross-entropy, or negative loglikelihood, loss function."""

    def __init__(
        self,
        enable_grad: bool = True,
    ) -> None:
        """Initialize the loss head."""
        self.enable_grad = enable_grad
        self.cache = {}

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute the loss for a set of logits and targets."""
        assert len(logits.shape) == 2 and len(targets.shape) == 1
        assert logits.shape[0] == targets.shape[0]

        if self.enable_grad:
            self.cache["logits"] = logits
            self.cache["targets"] = targets

        n = logits.shape[0]
        log_probabilities = log_softmax(logits)
        loss_elems = log_probabilities[np.arange(n), targets]
        loss = -np.mean(loss_elems)

        return loss

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss with respect to the logits."""
        assert self.enable_grad, "Cannot compute the backward pass with enable_grad=False"
        logits = self.cache["logits"]
        targets = self.cache["targets"]

        n = logits.shape[0]
        I = np.zeros(shape=logits.shape)
        I[np.arange(n), targets] = 1
        probabilities = softmax(logits)
        grad = (1 / n) * (probabilities - I)

        return grad


def train_one_block(
    alpha: float = 0.1,
    num_iters: int = 10_000,
):
    data = np.array(
        [
            [-0.80672381, -0.08818247, 0.002],
            [0.63413982, 1.32233656, 0.332],
            [0.1814214, -0.50674539, -0.0223],
            [1.16085551, -0.15033837, -0.332],
        ]
    )
    target = np.array([0, 1, 1, 2])

    model = Block(d_model=3)
    loss_fn = CrossEntropyLoss()

    print("-----------------------------------")
    for i in range(num_iters):
        logits = model.forward(data)
        loss = loss_fn.forward(logits, target)

        if i % 100 == 0:
            print(loss)
        if i == num_iters - 1:
            break

        dlogits = loss_fn.backward()
        model.backward(dlogits)
        model.step(alpha)
    print("-----------------------------------")

    logits = model.forward(data)
    probabilities = softmax(logits)
    print(probabilities)


def train_one_block_stack(
    alpha: float = 0.1,  # 0.000005524,
    num_iters: int = 10_000,
):
    data = np.array(
        [
            [-0.80672381, -0.08818247, 0.002],
            [0.63413982, 1.32233656, 0.332],
            [0.1814214, -0.50674539, -0.0223],
            [1.16085551, -0.15033837, -0.332],
        ]
    )
    target = np.array([0, 1, 1, 2])

    model = BlockStack(n_blocks=1, d_model=3)
    loss_fn = CrossEntropyLoss()

    print("-----------------------------------")
    for i in range(num_iters):
        logits = model.forward(data)
        loss = loss_fn.forward(logits, target)

        if i % 100 == 0:
            print(loss)
        if i == num_iters - 1:
            break

        dlogits = loss_fn.backward()
        model.backward(dlogits)
        model.step(alpha)
    print("-----------------------------------")

    logits = model.forward(data)
    probabilities = softmax(logits)
    print(probabilities)


def train_two_block_stacks(
    alpha: float = 0.5,
    num_iters: int = 10_000,
):
    data = np.array(
        [
            [-0.80672381, -0.08818247, 0.002],
            [0.63413982, 1.32233656, 0.332],
            [0.1814214, -0.50674539, -0.0223],
            [1.16085551, -0.15033837, -0.332],
        ]
    )
    target = np.array([0, 1, 1, 2])

    layer1 = BlockStack(n_blocks=1, d_model=3)
    layer2 = BlockStack(n_blocks=1, d_model=3)
    loss_fn = CrossEntropyLoss()

    print("-----------------------------------")
    for i in range(num_iters):
        hidden = layer1.forward(data)
        logits = layer2.forward(hidden)
        loss = loss_fn.forward(logits, target)

        if i % 100 == 0:
            print(loss)
        if i == num_iters - 1:
            break

        dlogits = loss_fn.backward()
        layer2.backward(dlogits)
        dhidden = layer2.cache["dx"]
        layer1.backward(dhidden)

        layer2.step(alpha)
        layer1.step(alpha)
    print("-----------------------------------")

    hidden = layer1.forward(data)
    logits = layer2.forward(hidden)
    probabilities = softmax(logits)
    print(probabilities)


def train_transformer(
    alpha: float = 1e-2,
    num_iters: int = 10_000,
):
    data = np.array([2, 0, 0, 1])
    target = np.array([0, 1, 1, 2])

    model = Transformer(vocab_size=3, d_model=2, n_encoding=1, n_decoding=1)
    loss_fn = CrossEntropyLoss()

    print("-----------------------------------")
    for i in range(num_iters):
        logits = model.forward(data)
        loss = loss_fn.forward(logits, target)

        if i % 100 == 0:
            print(loss)
        if i == num_iters - 1:
            break

        dlogits = loss_fn.backward()
        model.backward(dlogits)
        model.step(alpha)
    print("-----------------------------------")

    probabilities = model.predict(data)
    print(probabilities)


def main():
    ## LSE
    # x = np.array([
    #     [1,2,],
    #     [3,5,],
    # ])
    # y = log_sum_exp(x)
    # print(y)
    # l = log_softmax(x)
    # print(l)

    ## Softmax
    # x = np.array((1,2,3))
    # print(f"{x=}")
    # out = softmax_vector(x)
    # print(f"{out=}")
    # j = softmax_jacobian(x)
    # print(j)
    # step = 0.05
    # eps = np.array([0,0,1])
    # out2 = softmax_vector(x + step * eps)
    # print(out2)
    # estimate = out + step * np.squeeze(j[2:])
    # print(estimate)

    ## Attention
    # Q = np.random.random((10, 256))
    # K = np.random.random((5, 256))
    # V = np.random.random((5, 512))
    # out = scaled_dot_product_attention(Q, K, V)
    # print(out)

    ## Multihead attention
    # Q = np.random.standard_normal((10, 512))
    # K = np.random.standard_normal((5, 512))
    # V = np.random.standard_normal((5, 512))
    # a = MultiHeadAttention(d_model=512)
    # out = a.forward(Q, K, V)
    # print(out.shape)

    ## Linear
    # l = Linear(n_input=20, n_output=10)
    # x = np.random.standard_normal(size=(37, 20))
    # out = l.forward(x)
    # print(out.shape)

    ## FeedForward
    # f = FeedForward(n_input=20, n_hidden=2048, n_output=10)
    # x = np.random.standard_normal(size=(37, 20))
    # out = f.forward(x)
    # print(out.shape)

    ## LayerNorm
    # l = LayerNorm(n_dims=512)
    # x = np.random.standard_normal(size=(37, 512))
    # out = l.forward(x)
    # print(out.shape)

    # ## Block
    # b = Block()
    # x = np.random.standard_normal(size=(37, 512))
    # out = b.forward(x)
    # print(out.shape)

    ## Transformer
    # print("-----------------------------------")
    # print("Initializing...")
    # t = Transformer()
    # print(f"{t.n_params=:,}")
    # print("-----------------------------------")
    # input_sequence = np.array([0, 1, 0, 1, 0])
    # output_predictions = t.predict(input_sequence)
    # print(output_predictions.shape)
    # print(output_predictions[-1])

    ## CrossEntropyLoss
    # loss_fn = CrossEntropyLoss()
    # cur_input = np.array([
    #     [-0.80672381, -0.08818247],
    #     [ 0.63413982, 1.32233656],
    #     [ 0.1814214, -0.50674539],
    #     [ 1.16085551, -0.15033837],
    # ])
    # target = np.array([0,1,1,0])
    # alpha = 0.1
    # num_iters = 5_000
    # print("-----------------------------------")
    # for i in range(num_iters):
    #     loss = loss_fn.forward(cur_input, target)
    #     if i % 100 == 0:
    #         print(loss)
    #     if i == num_iters - 1:
    #         break
    #     grad = loss_fn.backward()
    #     cur_input = cur_input - alpha * grad
    # print("-----------------------------------")
    # probabilities = softmax(cur_input)
    # print(probabilities)

    ## CrossEntropyLoss + Linear
    # model = FeedForward(n_input=3, n_hidden=10, n_output=2)
    # norm = LayerNorm(n_dims=2)
    # model = Block(d_model=3)
    # model2 = Block(d_model=3)
    # loss_fn = CrossEntropyLoss()
    # data = np.array(
    #     [
    #         [-0.80672381, -0.08818247, 0.002],
    #         [0.63413982, 1.32233656, 0.332],
    #         [0.1814214, -0.50674539, -0.0223],
    #         [1.16085551, -0.15033837, -0.332],
    #     ]
    # )
    # target = np.array([0, 1, 1, 2])
    # alpha = 0.01
    # num_iters = 10_000
    # print("-----------------------------------")
    # for i in range(num_iters):
    #     # logits = model.forward(data)
    #     hidden = model.forward(data)
    #     print(hidden)
    #     break
    #     logits = model2.forward(hidden)

    #     loss = loss_fn.forward(logits, target)
    #     if i % 100 == 0:
    #         print(loss)
    #     if i == num_iters - 1:
    #         break
    #     dlogits = loss_fn.backward()

    #     model2.backward(dlogits)
    #     # dhidden = model2.cache["dx"]
    #     # model.backward(dhidden)

    #     # norm.backward(dlogits)
    #     # dhidden = norm.cache["dx"]
    #     # model.backward(dhidden)
    #     ##model.backward(dlogits)

    #     # norm.step(alpha)
    #     ##model.step(alpha)
    #     model2.step(alpha)
    #     # model.step(alpha)
    # print("-----------------------------------")
    # hidden = model.forward(data)
    # logits = model2.forward(hidden)
    # probabilities = softmax(logits)
    # print(probabilities)
    train_one_block_stack()


if __name__ == "__main__":
    main()
