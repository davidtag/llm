"""Registry of available split patterns for the RegexTokenizer."""


class SplitPattern:
    """Registry of available split patterns for the RegexTokenizer."""

    _PATTERNS = {
        "gpt-4": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  # noqa: E501
    }

    @classmethod
    def default_pattern_name(cls) -> str:
        """Get the default split pattern name."""
        return "gpt-4"

    @classmethod
    def all_pattern_names(cls) -> list[str]:
        """Get all valid split pattern names."""
        return list(cls._PATTERNS.keys())

    @classmethod
    def get_pattern(cls, pattern_name: str) -> str:
        """Get the split pattern of the given name."""
        if pattern_name not in cls._PATTERNS:
            raise ValueError(f"Unrecognized pattern: '{pattern_name}'")
        return cls._PATTERNS[pattern_name]
