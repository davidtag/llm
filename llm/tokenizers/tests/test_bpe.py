"""Unit tests for bpe.py."""

import unittest

from llm.tokenizers.bpe import (
    MergeList,
    MergeDict,
    Vocabulary,
    ReverseVocabulary,
    PieceCache,
    convert_merge_list_to_merge_dict,
    convert_merge_list_to_vocabulary,
    convert_vocabulary_to_reverse_vocabulary,
    convert_vocabulary_to_piece_cache,
    encode_piece,
    decode_bytes,
    render_bytes,
)
from llm.tokenizers.cython.stdtoken import TokenPair


class TestConvertMergeListToMergeDict(unittest.TestCase):
    """Unit tests for convert_merge_list_to_merge_dict()."""

    def test_empty(self) -> None:
        out = convert_merge_list_to_merge_dict([])
        self.assertDictEqual(out, {})

    def test_single_item(self) -> None:
        merge_list: MergeList = [
            (TokenPair(0, 1), 256),
        ]
        expected: MergeDict = {
            TokenPair(0, 1): 256,
        }
        out = convert_merge_list_to_merge_dict(merge_list)
        self.assertDictEqual(out, expected)

    def test_multiple_items(self) -> None:
        merge_list: MergeList = [
            (TokenPair(0, 1), 256),
            (TokenPair(13, 19), 257),
        ]
        expected: MergeDict = {
            TokenPair(0, 1): 256,
            TokenPair(13, 19): 257,
        }
        out = convert_merge_list_to_merge_dict(merge_list)
        self.assertDictEqual(out, expected)


class TestConvertMergeListToVocabulary(unittest.TestCase):
    """Unit tests for convert_merge_list_to_vocabulary()."""

    def _assert_valid_base_tokens(self, out: Vocabulary) -> None:
        self.assertGreaterEqual(len(out), 256)
        self.assertEqual(out[0], b"\x00")
        self.assertEqual(out[32], b" ")
        self.assertEqual(out[65], b"A")
        self.assertEqual(out[255], b"\xff")

    def test_empty(self) -> None:
        out = convert_merge_list_to_vocabulary([])
        self.assertEqual(256, len(out))
        self._assert_valid_base_tokens(out)

    def test_single_item(self) -> None:
        merge_list: MergeList = [
            (TokenPair(32, 65), 256),
        ]
        out = convert_merge_list_to_vocabulary(merge_list)
        self.assertEqual(256 + 1, len(out))
        self._assert_valid_base_tokens(out)
        self.assertEqual(out[256], b" A")

    def test_recursive_items(self) -> None:
        merge_list: MergeList = [
            (TokenPair(32, 65), 256),
            (TokenPair(32, 32), 257),
            (TokenPair(256, 65), 258),
            (TokenPair(257, 256), 259),
        ]
        out = convert_merge_list_to_vocabulary(merge_list)
        self.assertEqual(256 + 4, len(out))
        self._assert_valid_base_tokens(out)
        self.assertEqual(out[256], b" A")
        self.assertEqual(out[257], b"  ")
        self.assertEqual(out[258], b" AA")
        self.assertEqual(out[259], b"   A")


class TestConvertVocabularyToReverseVocabulary(unittest.TestCase):
    """Unit tests for convert_vocabulary_to_reverse_vocabulary()."""

    def test_empty(self) -> None:
        out = convert_vocabulary_to_reverse_vocabulary([])
        self.assertDictEqual(out, {})

    def test_multiple_items(self) -> None:
        vocab: Vocabulary = [
            b" ",
            b" A",
        ]
        expected: ReverseVocabulary = {
            b" ": 0,
            b" A": 1,
        }
        out = convert_vocabulary_to_reverse_vocabulary(vocab)
        self.assertDictEqual(out, expected)


class TestConvertVocabularyToPieceCache(unittest.TestCase):
    """Unit tests for convert_vocabulary_to_piece_cache()."""

    def test_empty(self) -> None:
        out = convert_vocabulary_to_piece_cache([])
        self.assertDictEqual(out, {})

    def test_single_valid_unicode(self) -> None:
        vocab: Vocabulary = [
            b" A",
        ]
        expected: PieceCache = {
            " A": [0],
        }
        out = convert_vocabulary_to_piece_cache(vocab)
        self.assertDictEqual(out, expected)

    def test_single_invalid_unicode(self) -> None:
        vocab: Vocabulary = [
            b"\xff",
        ]
        out = convert_vocabulary_to_piece_cache(vocab)
        self.assertDictEqual(out, {})

    def test_mix_valid_invalid_unicode(self) -> None:
        vocab: Vocabulary = [
            b" ",
            b" A",
            b"\xff",
            b"b",
            b"\xf0\x9f\x98\x89",  # == "😉".encode("utf-8")
        ]
        expected: PieceCache = {
            " ": [0],
            " A": [1],
            # pos 2 is not valid unicode
            "b": [3],
            "😉": [4],
        }
        out = convert_vocabulary_to_piece_cache(vocab)
        self.assertDictEqual(out, expected)


class TestEncodePiece(unittest.TestCase):
    """Unit tests for encode_piece()."""

    def test_empty_merge_empty_input(self) -> None:
        merge_dict: MergeDict = {}
        out = encode_piece("", merge_dict)
        self.assertListEqual(out, [])

    def test_empty_merge_single_input_char(self) -> None:
        merge_dict: MergeDict = {}
        out = encode_piece(" ", merge_dict)
        self.assertListEqual(out, [32])

    def test_empty_merge_multi_input_char(self) -> None:
        merge_dict: MergeDict = {}
        out = encode_piece(" A  b", merge_dict)
        self.assertListEqual(out, [32, 65, 32, 32, 98])

    def test_single_merge_empty_input(self) -> None:
        merge_dict: MergeDict = {
            TokenPair(32, 32): 256,
        }
        out = encode_piece("", merge_dict)
        self.assertListEqual(out, [])

    def test_single_merge_single_input_char(self) -> None:
        merge_dict: MergeDict = {
            TokenPair(32, 32): 256,
        }
        out = encode_piece("A", merge_dict)
        self.assertListEqual(out, [65])

    def test_single_merge_multi_input_char(self) -> None:
        merge_dict: MergeDict = {
            TokenPair(32, 32): 256,
        }
        out = encode_piece(" A  B", merge_dict)
        self.assertListEqual(out, [32, 65, 256, 66])

    def test_repeated_merge(self) -> None:
        merge_dict: MergeDict = {
            TokenPair(32, 32): 256,
        }
        out = encode_piece(" " * 4, merge_dict)
        self.assertListEqual(out, [256, 256])

    def test_2nd_order_merge(self) -> None:
        merge_dict: MergeDict = {
            TokenPair(32, 32): 256,
            TokenPair(256, 256): 257,
        }
        out = encode_piece(" " * 4, merge_dict)
        self.assertListEqual(out, [257])

    def test_merge_left_to_right(self) -> None:
        merge_dict: MergeDict = {
            TokenPair(32, 32): 256,
            TokenPair(256, 256): 257,
        }
        out = encode_piece(" " * 5, merge_dict)
        self.assertListEqual(out, [257, 32])  # not: 32, 257

        out = encode_piece(" " * 6, merge_dict)
        self.assertListEqual(out, [257, 256])  # not: 256, 257

        out = encode_piece(" " * 7, merge_dict)
        self.assertListEqual(out, [257, 256, 32])

        out = encode_piece(" " * 8, merge_dict)
        self.assertListEqual(out, [257, 257])

    def test_merge_priority(self) -> None:
        merge_dict: MergeDict = {
            TokenPair(32, 32): 256,
            TokenPair(32, 65): 257,
        }

        out = encode_piece(" A", merge_dict)
        self.assertListEqual(out, [257])

        out = encode_piece("  A", merge_dict)
        self.assertListEqual(out, [256, 65])  # not: 32, 257


class TestDecodeBytes(unittest.TestCase):
    """Unit tests for decode_bytes()."""

    def test_empty_input(self) -> None:
        vocab: Vocabulary = []
        out = decode_bytes([], vocab)
        self.assertEqual(out, b"")

    def test_single_input_token(self) -> None:
        vocab: Vocabulary = [
            b"A",
        ]
        out = decode_bytes([0], vocab)
        self.assertEqual(out, b"A")

    def test_multiple_input_token(self) -> None:
        vocab: Vocabulary = [
            b"A",
            b"B",
            b"\xf0\x9f\x98\x89",
        ]
        out = decode_bytes([2, 0, 1], vocab)
        self.assertEqual(out, b"\xf0\x9f\x98\x89AB")


class TestRenderBytes(unittest.TestCase):
    """Unit tests for render_bytes()."""

    def test_empty(self) -> None:
        out = render_bytes(b"")
        self.assertEqual(out, "")

    def test_new_line(self) -> None:
        out = render_bytes(b"\n")
        self.assertEqual(out, "\\u000a")

    def test_sequence_with_control_chars(self) -> None:
        out = render_bytes(b"a\xf0\x9f\x98\x89b\ncde")
        self.assertEqual(out, "a😉b\\u000acde")
