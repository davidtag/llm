"""Data types for handling tokenization."""

cdef class TokenPair:
    """Stores a pair of tokens.

    Requires about 1/2 as much memory and runs in 1/2 the time when construction and
    storing these objects in a Python dict, as compared to a regular tuple.
    """

    def __cinit__(self, Token first, Token second):
        self.first = first
        self.second = second


cdef class TokenPairNode:
    """A single node in a min-heap, representing a TokenPair and it's frequency in a TokenSequence.

    Implements `<` comparison to order by max-count, with tie-breaking by token values in ascending order.

    We store the TokenPair data directly in-line to avoid Python object overhead.
    TODO(dtag): define a new struct to avoid this code duplication.
    """

    def __cinit__(
        self,
        Token first,
        Token second,
        int count,
        # note: I needed to accept these as separate args because when accepting a TokenPair, I
        # get a [-Wmaybe-uninitialized] compiler warning on the generated Cython code.
        #Token token_1,
        #Token token_2,
        bint ignore = False
    ):
        self.first = first
        self.second = second
        self.count = count
        #self.pair = (token_1, token_2)
        self.ignore = ignore

    @property
    def count(self) -> int:
        return self.count

    @count.setter
    def count(self, int count) -> None:
        self.count = count

    @property
    def first(self) -> Token:
        return self.first

    @first.setter
    def first(self, Token first) -> None:
        self.first = first

    @property
    def second(self) -> Token:
        return self.second

    @second.setter
    def second(self, Token second) -> None:
        self.second = second

    @property
    def pair(self) -> TokenPair:
        return TokenPair(self.first, self.second)

    @pair.setter
    def pair(self, pair: TokenPair) -> None:
        self.first = pair.first
        self.second = pair.second

    @property
    def ignore(self) -> bool:
        return self.ignore

    @ignore.setter
    def ignore(self, bint ignore) -> None:
        self.ignore = ignore

    def __eq__(self, other: TokenPairNode) -> bool:
        return (
            self.first == other.first
            and self.second == other.second
            and self.count == other.count
            and self.ignore == other.ignore
        )

    def __lt__(self, other: TokenPairNode) -> bool:
        self_order = (-self.count, self.first, self.second)
        other_order = (-other.count, other.first, other.second)
        return  self_order <  other_order

    def __str__(self) -> str:
        return f"TokenPairNode(first={self.first}, second={self.second}, count={self.count}, ignore={self.ignore})"

    def __repr__(self) -> str:
        return self.__str__()
