# coding: utf-8
from nadamq.NadaMq import cPacket, parse_from_string, PACKET_TYPES


def test_parse_data():
    a = cPacket(iuid=1234, type_=PACKET_TYPES.DATA, data=b'hello')
    b = parse_from_string(a.tobytes())

    assert a.type_ == b.type_
    assert a.iuid == b.iuid
    assert a.crc == b.crc
    assert a.data() == b.data()


def test_parse_ack():
    a = cPacket(iuid=1010, type_=PACKET_TYPES.ACK)
    b = parse_from_string(a.tobytes())

    assert a.type_ == b.type_
    assert a.iuid == b.iuid


def test_parse_nack():
    a = cPacket(iuid=4321, type_=PACKET_TYPES.NACK)
    b = parse_from_string(a.tobytes())

    # TODO Should parse fail with `NACK` type?  It currently does.
    assert not b


def test_parse_id_response():
    """
    .. versionadded:: 0.13
    """
    a = cPacket(iuid=1234, type_=PACKET_TYPES.ID_RESPONSE,
                data=b'{"id": "my device name"}')
    b = parse_from_string(a.tobytes())

    assert a.type_ == b.type_
    assert a.iuid == b.iuid
    assert a.crc == b.crc
    # assert a.data() == b.data()


def test_parse_id_request():
    """
    .. versionadded:: 0.13
    """
    a = cPacket(iuid=1234, type_=PACKET_TYPES.ID_REQUEST)
    b = parse_from_string(a.tobytes())

    assert a.type_ == b.type_
    assert a.iuid == b.iuid
