# coding: utf-8
"""
Equivalence tests for the table-driven CRC-16 implementation.

The lookup table is derived at import time from the bit-by-bit routine
(:func:`nadamq.NadaMq._crc_update_bitwise`), which is kept as the ground
truth; these tests assert that the two agree for every ``(crc, byte)`` pair
the table is ever indexed with.
"""
import random

import numpy as np
import pytest

from nadamq.NadaMq import (_CRC16_TABLE, _REFLECT8_TABLE, _crc_update_bitwise,
                           compute_crc16, crc_reflect, crc_update,
                           crc_update_byte)


def test_check_value():
    """CRC-16/ARC check value: CRC of b'123456789' is 0xBB3D."""
    assert compute_crc16(b'123456789') == 0xBB3D


def test_tables_are_derived_from_the_bitwise_routine():
    assert len(_CRC16_TABLE) == 256
    assert list(_CRC16_TABLE) == [_crc_update_bitwise(i << 8, b'\x00')
                                  for i in range(256)]
    assert list(_REFLECT8_TABLE) == [crc_reflect(i, 8) for i in range(256)]


@pytest.mark.parametrize('crc', [0x0000, 0x0001, 0x00FF, 0x1234, 0x8000,
                                 0xABCD, 0xFFFF])
def test_all_bytes_match_bitwise(crc):
    """Table and bitwise routine agree for all 256 byte values."""
    for value in range(256):
        data = bytes([value])
        assert crc_update(crc, data) == _crc_update_bitwise(crc, data)


def test_dense_state_sample_matches_bitwise():
    """All 256 byte values against a dense sample of CRC states."""
    rng = random.Random(0xBB3D)
    states = list(range(256)) + [rng.randrange(1 << 16) for _ in range(512)]
    for crc in states:
        for value in range(256):
            data = bytes([value])
            assert crc_update(crc, data) == _crc_update_bitwise(crc, data)


def test_random_payloads_match_bitwise():
    """Whole-payload CRCs agree for random payloads of varying length."""
    rng = random.Random(1234)
    for _ in range(500):
        payload = bytes(rng.randrange(256)
                        for _ in range(rng.randrange(0, 301)))
        assert crc_update(0, payload) == _crc_update_bitwise(0, payload)
        # Splitting a payload across two calls must fold identically.
        cut = rng.randrange(0, len(payload) + 1)
        assert (crc_update(crc_update(0, payload[:cut]), payload[cut:]) ==
                crc_update(0, payload))


def test_empty_payload_is_a_no_op():
    for crc in (0x0000, 0x1234, 0xFFFF):
        assert crc_update(crc, b'') == crc
        assert crc_update(crc, bytearray()) == crc
        assert crc_update(crc, np.array([], dtype=np.uint8)) == crc


@pytest.mark.parametrize('value', [0, 1, 0x7F, 0x80, 0xFE, 0xFF])
def test_input_types_are_interchangeable(value):
    """`int`, `numpy.uint8`, `bytes`, `bytearray` and arrays all agree."""
    for crc in (0x0000, 0x1234, 0xFFFF):
        expected = _crc_update_bitwise(crc, bytes([value]))
        assert crc_update(crc, bytes([value])) == expected
        assert crc_update(crc, bytearray([value])) == expected
        assert crc_update(crc, [value]) == expected
        assert crc_update(crc, [np.uint8(value)]) == expected
        assert crc_update(crc, np.array([value], dtype=np.uint8)) == expected
        assert crc_update_byte(crc, value) == expected


def test_return_type_is_int():
    for data in (b'\x01\x02', bytearray(b'\x01\x02'),
                 np.array([1, 2], dtype=np.uint8), [np.uint8(1), np.uint8(2)]):
        assert type(crc_update(0, data)) is int
    assert type(compute_crc16(b'hello')) is int
