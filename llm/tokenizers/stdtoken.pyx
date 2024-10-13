"""Cython data types for handling tokenization."""

# To support efficient hashing of TokenPair, we define a maximum value so that a pair
# of token values can be uniquely represented by an integer. Token values should be in
# [0, 1, ..., TOKEN_VALUE_UBOUND - 1].
cdef uint64_t TOKEN_VALUE_UBOUND = 1_000_000


cdef class TokenPair:
    """Stores an immutable pair of tokens.

    An instance of this type occupies 32 bytes of memory, compared to 56 bytes for a tuple
    of two ints. It is also about 30% faster when building a dict of TokenPair instead of
    a dict of Tuple[int, int].
    """

    def __cinit__(
        self,
        token_t first,
        token_t second,
    ):
        self._first = first
        self._second = second
        self._unique = self._first * TOKEN_VALUE_UBOUND + self._second

    @property
    def first(self) -> int:
        return self._first

    @property
    def second(self) -> int:
        return self._second

    def __eq__(self, other: TokenPair) -> bool:
        return self._unique == other._unique

    def __lt__(self, other: TokenPair) -> bool:
        return self._unique < other._unique

    def __hash__(self) -> int:
        return self._unique

    def __str__(self) -> str:
        return f"TokenPair(first={self._first}, second={self._second})"

    def __repr__(self) -> str:
        return self.__str__()



cdef class TokenPairNode:
    """A single node in a min-heap, representing a TokenPair and it's frequency in a token sequence.

    Implements `<` comparison to order by max-count, with tie-breaking by token values in ascending order.

    We store the TokenPair data directly in-line to avoid Python object overhead.
    """

    def __cinit__(
        self,
        token_t first,
        token_t second,
        int32_t count,
        bint deleted = False
    ):
        self._first = first
        self._second = second
        self._count = count
        self._deleted = deleted

    @property
    def first(self) -> int:
        return self._first

    @property
    def second(self) -> int:
        return self._second

    @property
    def pair(self) -> TokenPair:
        return TokenPair(self._first, self._second)

    @property
    def count(self) -> int:
        return self._count

    @count.setter
    def count(self, int32_t count) -> None:
        self._count = count

    @property
    def deleted(self) -> bool:
        return self._deleted

    @deleted.setter
    def deleted(self, bint deleted) -> None:
        self._deleted = deleted

    def __eq__(self, other: TokenPairNode) -> bool:
        return (
            self._first == other._first
            and self._second == other._second
            and self._count == other._count
            and self._deleted == other._deleted
        )

    def __lt__(self, other: TokenPairNode) -> bool:
        self_order = (-self._count, self._first, self._second)
        other_order = (-other._count, other._first, other._second)
        return  self_order <  other_order

    def __str__(self) -> str:
        return (
            "TokenPairNode("
            f"first={self._first}, second={self._second}, count={self._count}, deleted={self._deleted}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()
