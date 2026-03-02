# coding: utf-8
import pytest
from nadamq.NadaMq import cPacket, PACKET_TYPES


def test_buffer_auto():
    """
    Test auto-buffer based on data content.
    """
    p = cPacket(iuid=1234, type_=PACKET_TYPES.DATA, data=b'hello, world!')
    assert p.buffer_size == len(b'hello, world!')
    assert p.data() == b'hello, world!'


def test_buffer_larger_than_data():
    """
    Test setting data while allocating larger buffer.
    """
    p = cPacket(iuid=1234, type_=PACKET_TYPES.DATA, data=b'hello, world!',
                buffer_size=1024)
    assert p.data() == b'hello, world!'
    assert p.buffer_size == 1024


def test_buffer():
    """
    Test setting allocating buffer without setting data contents.
    """
    p = cPacket(iuid=1234, type_=PACKET_TYPES.DATA, buffer_size=1024)
    assert p.buffer_size == 1024
    assert p.data() == b''


def test_no_buffer():
    """
    Test no buffer allocation.
    """
    p = cPacket(iuid=1234, type_=PACKET_TYPES.DATA)
    with pytest.raises(RuntimeError, match='No buffer has been set/allocated.'):
        p.data()


def test_default_packet():
    """
    Test default packet allocation, i.e., no iuid, type, or data/buffer.
    """
    p = cPacket()
    assert p.iuid == 0
    assert p.type_ == PACKET_TYPES.NONE
    assert p.buffer_size == 0


def test_cPacket():
    """
    Test serialization of ``cPacket`` containing ``"hello, world!"``.
    """
    packet = cPacket(data=b'hello, world!', type_=PACKET_TYPES.DATA)
    assert packet.tobytes() == b'|||\x00\x00d\x00\rhello, world!\xfa5'
