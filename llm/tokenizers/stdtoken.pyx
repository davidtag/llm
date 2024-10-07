"""Cython data types for handling tokenization."""

# To support efficient hashing of TokenPair, we define a maximum value so that a pair
# of token values can be uniquely represented by an integer. This is EXClusive.
cdef uint32_t TOKEN_VALUE_UBOUND = 1_000_000


cdef class TokenPair:
    """Stores an immutable pair of tokens.

    Requires about 1/2 as much memory and runs in 1/2 the time when constructing and
    storing these objects in a Python dict, as compared to a regular tuple.
    """

    def __cinit__(
        self,
        token_t first,
        token_t second,
    ):
        self.first = first
        self.second = second
        self._unique = self.first * TOKEN_VALUE_UBOUND + self.second

    @property
    def first(self) -> int:
        return self.first

    @property
    def second(self) -> int:
        return self.second

    def __eq__(self, other: TokenPair) -> bool:
        return self._unique == other._unique

    def __hash__(self) -> int:
        return self._unique

    def __str__(self) -> str:
        return f"TokenPair(first={self.first}, second={self.second})"

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
        self.first = first
        self.second = second
        self.count = count
        self.deleted = deleted

    @property
    def first(self) -> int:
        return self.first

    @property
    def second(self) -> int:
        return self.second

    @property
    def pair(self) -> TokenPair:
        return TokenPair(self.first, self.second)

    @property
    def count(self) -> int:
        return self.count

    @count.setter
    def count(self, int32_t count) -> None:
        self.count = count

    @property
    def deleted(self) -> bool:
        return self.deleted

    @deleted.setter
    def deleted(self, bint deleted) -> None:
        self.deleted = deleted

    def __eq__(self, other: TokenPairNode) -> bool:
        return (
            self.first == other.first
            and self.second == other.second
            and self.count == other.count
            and self.deleted == other.deleted
        )

    def __lt__(self, other: TokenPairNode) -> bool:
        self_order = (-self.count, self.first, self.second)
        other_order = (-other.count, other.first, other.second)
        return  self_order <  other_order

    def __str__(self) -> str:
        return (
            "TokenPairNode("
            f"first={self.first}, second={self.second}, count={self.count}, deleted={self.deleted}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()
