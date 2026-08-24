# coding: utf-8
"""
Tests for :attr:`nadamq.NadaMq.cPacketParser.bytes_consumed`.

Contract
--------
``bytes_consumed`` is assigned by *every* :meth:`parse` call:

- a packet was returned -- index of the completing byte **+ 1** (i.e. the
  number of bytes of ``data`` that were parsed; the rest is untouched);
- ``False`` was returned -- ``len(data)`` (everything was consumed, error
  recovery included), and ``0`` for empty input.
"""
import numpy as np
import pytest

from nadamq.NadaMq import (HEADER_ONLY_PACKET_SIZE, PACKET_TYPES, cPacket,
                           cPacketParser)

DATA = cPacket(iuid=3, type_=PACKET_TYPES.DATA, data=b'hello').tobytes()
STREAM = cPacket(iuid=4, type_=PACKET_TYPES.STREAM, data=b'{"a": 1}').tobytes()
ACK = cPacket(iuid=7, type_=PACKET_TYPES.ACK).tobytes()
NACK = cPacket(iuid=8, type_=PACKET_TYPES.NACK).tobytes()
ID_REQUEST = cPacket(iuid=9, type_=PACKET_TYPES.ID_REQUEST).tobytes()
CORRUPT = bytes(bytearray(DATA)[:-1]) + bytes([DATA[-1] ^ 0xFF])


def extract(parser, data):
    """Drain `data` using `bytes_consumed`; return (type_, iuid, payload)."""
    packets = []
    while data:
        packet = parser.parse(data)
        data = data[parser.bytes_consumed:]
        if packet is not False:
            packets.append((packet.type_, packet.iuid, packet.data()))
    return packets


def extract_byte_at_a_time(parser, data):
    """Same, using the historical one-byte-per-call idiom."""
    packets = []
    for i in range(len(data)):
        packet = parser.parse(data[i:i + 1])
        if packet is not False:
            packets.append((packet.type_, packet.iuid, packet.data()))
            parser.reset()
        elif parser.error:
            parser.reset()
    return packets


def test_initial_value_is_zero():
    assert cPacketParser().bytes_consumed == 0


@pytest.mark.parametrize('empty', [b'', bytearray(), [],
                                   np.array([], dtype=np.uint8)])
def test_empty_input(empty):
    parser = cPacketParser()
    assert parser.parse(empty) is False
    assert parser.bytes_consumed == 0


def test_single_byte_feeds_always_report_one():
    parser = cPacketParser()
    for i in range(len(DATA)):
        parser.parse(DATA[i:i + 1])
        assert parser.bytes_consumed == 1


@pytest.mark.parametrize('raw', [DATA, STREAM, ACK, NACK, ID_REQUEST])
def test_completing_index_with_trailing_bytes(raw):
    """A packet must not consume the bytes that follow it."""
    parser = cPacketParser()
    packet = parser.parse(raw + b'TRAILING')
    assert packet is not False
    assert parser.bytes_consumed == len(raw)


@pytest.mark.parametrize('raw', [ACK, NACK, ID_REQUEST])
def test_header_only_packets_consume_six_bytes(raw):
    parser = cPacketParser()
    assert parser.parse(raw + DATA) is not False
    assert parser.bytes_consumed == HEADER_ONLY_PACKET_SIZE == len(raw)


def test_multi_packet_chunk_offsets():
    parser = cPacketParser()
    raw = ACK + DATA + STREAM + NACK
    consumed = []
    data = raw
    while data:
        parser.parse(data)
        consumed.append(parser.bytes_consumed)
        data = data[parser.bytes_consumed:]
    assert consumed == [len(ACK), len(DATA), len(STREAM), len(NACK)]


@pytest.mark.parametrize('blob', [b'ABCDEF', b'|||', DATA[:6], b'\x00' * 100])
def test_incomplete_input_consumes_everything(blob):
    parser = cPacketParser()
    assert parser.parse(blob) is False
    assert parser.bytes_consumed == len(blob)


def test_error_only_consumes_everything():
    parser = cPacketParser()
    assert parser.parse(CORRUPT) is False
    assert parser.error
    assert parser.bytes_consumed == len(CORRUPT)


def test_error_recovery_mid_chunk():
    """A corrupt packet followed by a valid one recovers in a single call."""
    parser = cPacketParser()
    raw = CORRUPT + DATA
    packet = parser.parse(raw)
    assert packet is not False
    assert packet.data() == b'hello'
    # The corrupt packet *and* the recovered one were consumed.
    assert parser.bytes_consumed == len(raw)


@pytest.mark.parametrize('wrap', [bytes, bytearray,
                                  lambda raw: np.frombuffer(raw, dtype='uint8'),
                                  list])
def test_input_types(wrap):
    parser = cPacketParser()
    packet = parser.parse(wrap(DATA + b'xx'))
    assert packet is not False
    assert parser.bytes_consumed == len(DATA)


def test_reset_does_not_clobber_the_report():
    parser = cPacketParser()
    parser.parse(DATA + b'zz')
    consumed = parser.bytes_consumed
    parser.reset()
    assert parser.bytes_consumed == consumed


def test_chunk_fed_matches_byte_at_a_time():
    raw = ACK + DATA + STREAM + NACK + CORRUPT + DATA
    assert (extract(cPacketParser(), raw) ==
            extract_byte_at_a_time(cPacketParser(), raw))
    assert [name for name, _, _ in extract(cPacketParser(), raw)] == [
        PACKET_TYPES.ACK, PACKET_TYPES.DATA, PACKET_TYPES.STREAM,
        PACKET_TYPES.NACK, PACKET_TYPES.DATA]


@pytest.mark.parametrize('size', [1, 2, 3, 5, 7, 16, 64])
def test_arbitrary_chunk_boundaries(size):
    """Any chunking yields the same packets as byte-at-a-time feeding."""
    raw = DATA + ACK + STREAM + CORRUPT + NACK + DATA + ID_REQUEST
    parser = cPacketParser()
    packets = []
    pending = b''
    for offset in range(0, len(raw), size):
        pending += raw[offset:offset + size]
        while pending:
            packet = parser.parse(pending)
            pending = pending[parser.bytes_consumed:]
            if packet is not False:
                packets.append((packet.type_, packet.iuid, packet.data()))
            else:
                break
    assert packets == extract_byte_at_a_time(cPacketParser(), raw)
