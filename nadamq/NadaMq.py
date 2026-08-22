import struct
from typing import Optional, Union, TypeVar, Generic, List, Dict, Set
from collections import deque
import numpy as np
from io import BytesIO
from logging_helpers import _L
from enum import Enum, auto
from dataclasses import dataclass


class PacketType:
    """Packet types supported by the protocol."""
    def __init__(self):
        self.NONE = 0
        self.ACK = ord('a')
        self.NACK = ord('n')
        self.DATA = ord('d')
        self.STREAM = ord('s')
        self.ID_REQUEST = ord('i')
        self.ID_RESPONSE = ord('I')

    def __setattr__(self, name, value):
        # Prevent modification of already-set attributes.
        if name in self.__dict__:
            raise AttributeError(f"'{name}' is read-only")
        self.__dict__[name] = value


# Create singleton instance
PACKET_TYPE = PacketType()


class _Flags:
    START = b'|||'


class _PacketTypes:
    NONE = PACKET_TYPE.NONE
    ACK = PACKET_TYPE.ACK
    NACK = PACKET_TYPE.NACK
    DATA = PACKET_TYPE.DATA
    STREAM = PACKET_TYPE.STREAM
    ID_REQUEST = PACKET_TYPE.ID_REQUEST
    ID_RESPONSE = PACKET_TYPE.ID_RESPONSE


PACKET_TYPES = _PacketTypes()
FLAGS = _Flags()

PACKET_NAME_BY_TYPE: Dict[int, str] = {
    PACKET_TYPE.NONE: 'NONE',
    PACKET_TYPE.ACK: 'ACK',
    PACKET_TYPE.NACK: 'NACK',
    PACKET_TYPE.DATA: 'DATA',
    PACKET_TYPE.STREAM: 'STREAM',
    PACKET_TYPE.ID_REQUEST: 'ID_REQUEST',
    PACKET_TYPE.ID_RESPONSE: 'ID_RESPONSE'
}

#: Packet types that consist of a header *only* on the wire, i.e.::
#:
#:     START(3) + IUID(2) + TYPE(1)
#:
#: These packets carry **no** ``LENGTH``, ``PAYLOAD`` or ``CRC`` fields, so
#: they are complete as soon as the type field has been read.  Every other
#: packet type uses the full framing::
#:
#:     START(3) + IUID(2) + TYPE(1) + LENGTH(2) + PAYLOAD(n) + CRC(2)
HEADER_ONLY_TYPES: Set[int] = frozenset({PACKET_TYPES.ACK, PACKET_TYPES.NACK,
                                         PACKET_TYPES.ID_REQUEST})

#: Number of bytes in a header-only packet (see :data:`HEADER_ONLY_TYPES`).
HEADER_ONLY_PACKET_SIZE = 6


def crc_init() -> int:
    """Initialize CRC-16 value."""
    return 0x0000


def crc_reflect(data: int, data_len: int) -> int:
    """Reflect all bits of a data word of data_len bits."""
    if data_len <= 0:
        return 0
    ret = data & 0x01
    for _ in range(1, data_len):
        data >>= 1
        ret = (ret << 1) | (data & 0x01)
    return ret


def crc_update(crc: int, data: Union[bytes, bytearray, np.ndarray]) -> int:
    """Update CRC with a sequence of bytes."""
    for byte in data:
        c = byte
        for _ in range(8):  # Process each bit
            bit = bool(crc & 0x8000)
            if c & 0x01:
                bit = not bit
            crc <<= 1
            if bit:
                crc ^= 0x8005  # CRC-16 polynomial
            c >>= 1
        crc &= 0xFFFF
    return crc


def crc_update_byte(crc: int, byte: int) -> int:
    """Update CRC with a single byte."""
    return crc_update(crc, bytes([byte]))


def crc_finalize(crc: int) -> int:
    """Finalize CRC computation."""
    return crc_reflect(crc, 16) ^ 0x0000


def compute_crc16(data: Union[bytes, bytearray]) -> int:
    """Compute CRC-16 for given data."""
    crc = crc_init()
    crc = crc_update(crc, data)
    return crc_finalize(crc)


class FixedPacket:
    def __init__(self, type_=PACKET_TYPE.NONE, iuid=0, data=None,
                 buffer_=None, buffer_size=None):
        self.iuid_ = iuid
        self._type = type_
        self.payload_length_ = 0
        # `buffer_size_` is the *capacity of the payload buffer*, i.e. zero
        # until a buffer is set/allocated.  It deliberately does **not**
        # include the framing overhead (start flag/IUID/type/length/CRC).
        self.buffer_size_ = 0
        self.payload_buffer_ = None
        self.crc_ = 0
        self.buffer_ = None

        if data is not None:
            if buffer_ is not None:
                if len(buffer_) < len(data):
                    raise ValueError(f'Supplied buffer is not long enough to hold `data`, {len(buffer_)} < {len(data)}')
                elif buffer_size is not None:
                    raise ValueError('Buffer size must not be specified when a buffer is supplied')
                else:
                    self.set_buffer(buffer_)
            elif buffer_size is None:
                # Auto-size the payload buffer to exactly hold `data`.  N.B.
                # at least one byte is allocated so that an empty payload
                # still results in an *allocated* (but empty) buffer.
                buffer_size = max(len(data), 1)
        elif buffer_ is None and buffer_size is None and type_ in HEADER_ONLY_TYPES:
            # A header-only packet can never carry a payload, so allocate an
            # empty payload buffer.  This keeps `data()` valid (returning
            # ``b''``) for e.g. `cPacket(type_=PACKET_TYPES.ACK)`, which
            # callers log/measure alongside packets that do have payloads.
            buffer_size = 0

        if buffer_size is not None and buffer_size >= 0:
            self.alloc_buffer(buffer_size)

        if data is not None:
            self.set_data(data)

    @property
    def max_buffer_size(self) -> int:
        return (1 << 16) - 1

    @property
    def crc(self) -> int:
        return self.crc_

    @property
    def buffer_size(self) -> int:
        return self.buffer_size_

    def set_data(self, data: Union[bytes, bytearray]) -> None:
        if len(data) > self.buffer_size_:
            raise ValueError(f'Data length is too large for buffer, {len(data)} > {self.buffer_size_}')
        
        self.payload_buffer_[:len(data)] = data
        self.payload_length_ = len(data)
        self.compute_crc()

    def data(self) -> bytes:
        """
        Returns
        -------
        bytes
            Payload contents (empty if a buffer has been allocated but no
            payload has been set, e.g. a header-only or zero-length packet).

        Raises
        ------
        RuntimeError
            If no payload buffer has ever been set/allocated.
        """
        if self.payload_buffer_ is None:
            raise RuntimeError('No buffer has been set/allocated.')
        if self.payload_length_ > 0:
            return bytes(self.payload_buffer_[:self.payload_length_])
        return b''

    def data_ptr(self) -> int:
        """Return the pointer to the payload buffer."""
        if self.payload_buffer_ is None:
            return 0
        return id(self.payload_buffer_)

    def compute_crc(self) -> None:
        """Compute CRC-16 for payload data only."""
        self.crc_ = compute_crc16(self.data())

    def tobytes(self) -> bytes:
        """
        Serialize the packet to its NadaMQ wire representation.

        The format is selected by the *packet type* (**not** by the size of
        any allocated buffer)::

            header-only types  START(3) + IUID(2) + TYPE(1)
            all other types    START(3) + IUID(2) + TYPE(1) + LENGTH(2)
                               + PAYLOAD(n) + CRC(2)

        Notes
        -----
        Exceptions are deliberately **not** swallowed: silently emitting a
        zero-byte packet corrupts the stream far more subtly than raising.
        """
        # START + IUID(2, big-endian) + TYPE(1)
        header = struct.pack('>HB', self.iuid_, self._type)

        if self._type in HEADER_ONLY_TYPES:
            if self.payload_length_ > 0:
                _L().warning(
                    f'Payload of {self.payload_length_} byte(s) dropped: '
                    f'{PACKET_NAME_BY_TYPE.get(self._type, self._type)} '
                    'packets are header-only.')
            return FLAGS.START + header

        header += struct.pack('>H', self.payload_length_)  # Length (2 bytes)
        packet = FLAGS.START + header
        if self.payload_length_ > 0:
            packet += self.data()
        packet += struct.pack('>H', self.crc_)
        return packet

    def tostring(self) -> bytes:
        import warnings
        warnings.warn("tostring() is deprecated, use tobytes() instead",
                      DeprecationWarning, stacklevel=2)
        return self.tobytes()

    @property
    def type_(self) -> int:
        return self._type

    @type_.setter
    def type_(self, value: int) -> None:
        self._type = value

    @property
    def iuid(self) -> int:
        return self.iuid_

    @iuid.setter
    def iuid(self, value: int) -> None:
        self.iuid_ = value

    def clear_buffer(self) -> None:
        """Deallocate buffer if it has been allocated."""
        self.payload_buffer_ = None
        self.buffer_ = None
        self.buffer_size_ = 0
        self.payload_length_ = 0

    def realloc_buffer(self, buffer_size: int) -> None:
        """Allocate the specified buffer size, deallocating the existing buffer."""
        if buffer_size > self.max_buffer_size:
            raise RuntimeError(f'Max buffer size is {self.max_buffer_size}')
        self.clear_buffer()
        self.alloc_buffer(buffer_size)

    def alloc_buffer(self, buffer_size: int) -> None:
        """Allocate the specified buffer size."""
        if self.buffer_ is not None:
            raise RuntimeError('Buffer has already been allocated.')
        if buffer_size > self.max_buffer_size:
            raise RuntimeError(f'Max buffer size is {self.max_buffer_size}')
        self.buffer_ = bytearray(buffer_size)
        self.set_buffer(self.buffer_, overwrite=True)
        self.buffer_size_ = buffer_size

    def set_buffer(self, data: Union[bytes, bytearray, np.ndarray], overwrite: bool = False) -> None:
        """Assign the specified data buffer as the payload buffer of the packet."""
        if self.payload_buffer_ is not None and not overwrite:
            raise RuntimeError('Packet already has a payload buffer allocated. Must use `overwrite=True` to set buffer anyway.')
        self.payload_buffer_ = bytearray(data)
        # `buffer_size_` tracks the payload *capacity* only.
        self.buffer_size_ = len(data)
        self.payload_length_ = 0

    def __str__(self) -> str:
        try:
            return (f"FixedPacket(type={PACKET_NAME_BY_TYPE.get(self._type, 'UNKNOWN')}, "
                    f"iuid={self.iuid_}, length={self.payload_length_}, "
                    f"data={self.data() if self.payload_length_ > 0 else b''})")
        except Exception as e:
            return f"FixedPacket(ERROR: {str(e)})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def is_valid(self) -> bool:
        """Check if packet is in a valid state."""
        return (self.type_ in PACKET_NAME_BY_TYPE and
                self.payload_length_ <= self.max_buffer_size and
                (self.payload_length_ == 0 or self.payload_buffer_ is not None))


class PacketState(Enum):
    """States for packet parsing."""
    START = auto()
    HEADER = auto()
    LENGTH = auto()
    PAYLOAD = auto()
    CRC = auto()


class cPacketParser:
    """Parser for fixed-size packets with CRC validation.
    
    The parser implements a state machine to process incoming bytes and construct
    complete packets. The packet format is:
    START(3) + IUID(2) + TYPE(1) + [LENGTH(2) + PAYLOAD(n) + CRC(2)]
    
    Attributes:
        payload_bytes_received_: Number of payload bytes received
        payload_bytes_expected_: Expected payload length
        message_completed_: True if the most recent :meth:`parse` call
            completed a message
        parse_error_: True if the most recent :meth:`parse` call encountered a
            parse error
        state: Current state of the parser (:class:`PacketState`)
        buffer: Bytes consumed so far for the field currently being parsed
        packet: Packet currently being assembled

    Version log
    -----------
    .. versionchanged:: 0.54
        Header-only packets (see :data:`HEADER_ONLY_TYPES`) are completed as
        soon as the type field has been read, so they parse correctly when fed
        one byte at a time (and no longer consume the start flag of the packet
        that follows them).

    .. versionchanged:: 0.54
        :attr:`message_completed` and :attr:`error` describe the outcome of
        the *most recent* :meth:`parse` call and remain readable until the
        next call to :meth:`parse` (previously both flags were consumed before
        :meth:`parse` returned, making them useless to callers).
    """
    def __init__(self, buffer_size: int = (1 << 16) - 1):
        #: Maximum accepted payload length.  A packet declaring a longer
        #: payload is treated as a parse error *before* anything is allocated.
        self.max_payload_size = max(0, min(int(buffer_size), (1 << 16) - 1))
        self.payload_bytes_received_ = 0
        self.payload_bytes_expected_ = 0
        self.message_completed_ = False
        self.parse_error_ = False
        self.crc_ = 0
        self.packet = FixedPacket()
        self.state = PacketState.START
        self.buffer = bytearray()
        self.reset()

    def _reset_state(self) -> None:
        """
        Reset the state machine (but **not** the public outcome flags).
        """
        self.state = PacketState.START
        self.buffer = bytearray()
        self.payload_bytes_received_ = 0
        self.payload_bytes_expected_ = 0
        self.crc_ = 0
        if hasattr(self.packet, 'clear_buffer'):
            self.packet.clear_buffer()
        # Clear the header fields so a stale type/IUID from a previous
        # (aborted) packet can never be mistaken for a partially parsed one.
        self.packet.type_ = PACKET_TYPE.NONE
        self.packet.iuid_ = 0

    def reset(self) -> None:
        """
        Reset parser state and packet, clearing the :attr:`message_completed`
        and :attr:`error` flags.

        Safe to call at any time, including immediately after :meth:`parse`
        has returned a completed packet (the state machine has already been
        reset internally in that case, so this is a no-op apart from clearing
        the flags).
        """
        self._reset_state()
        self.message_completed_ = False
        self.parse_error_ = False

    def parse(self, data: Union[bytes, bytearray, np.ndarray]) -> Union[FixedPacket, bool]:
        """
        Parse packet data and return the packet if one was completed.

        Returns
        -------
        FixedPacket or bool
            Completed packet, or ``False`` if no packet was completed by this
            call.  N.B. ``False`` (**not** ``None``) is returned for the
            incomplete case, since callers test ``result is not False``.
        """
        # The flags describe the outcome of *this* call only.
        self.message_completed_ = False
        self.parse_error_ = False
        # Whether *any* byte in this call triggered a parse error.  Tracked
        # separately from `self.parse_error_`, which must be cleared after each
        # error so that the bytes *following* the error are parsed normally
        # (rather than each one re-triggering the error branch below, which
        # made it impossible to recover a valid packet from the remainder of a
        # chunk that started with corrupt data).
        error_occurred = False

        for byte in data:
            self.parse_byte(int(byte))

            if self.message_completed_:
                packet = self.packet
                # Create a new packet for the next message.
                self.packet = FixedPacket()
                self._reset_state()
                # Keep the outcome observable to the caller until its next
                # call to `parse()`.
                self.message_completed_ = True
                self.parse_error_ = False
                return packet
            elif self.parse_error_:
                _L().debug('Error parsing packet - resetting and continuing')
                # Reset state but continue processing remaining data.
                self._reset_state()
                self.parse_error_ = False
                error_occurred = True
        # No packet completed during this call; report whether an error was
        # encountered along the way.
        self.parse_error_ = error_occurred
        return False

    def parse_byte(self, byte: int) -> None:
        """Parse a single byte."""
        try:
            self.buffer.append(byte)

            if self.state == PacketState.START:
                # Only the trailing 3 bytes are ever inspected for the start
                # flag, so discard anything older to keep the buffer bounded
                # on a noisy line.
                if len(self.buffer) > len(FLAGS.START):
                    del self.buffer[:-len(FLAGS.START)]
                if bytes(self.buffer) == FLAGS.START:
                    self.state = PacketState.HEADER
                    self.buffer = bytearray()

            elif self.state == PacketState.HEADER:  # IUID(2) + TYPE(1)
                if len(self.buffer) >= 3:
                    self.packet.iuid_ = struct.unpack('>H', bytes(self.buffer[0:2]))[0]
                    self.packet.type_ = self.buffer[2]
                    self.buffer = bytearray()

                    # Validate header
                    if self.packet.type_ not in PACKET_NAME_BY_TYPE:
                        _L().error(f"Invalid packet type: {hex(self.packet.type_)}")
                        self.parse_error_ = True
                        return

                    if self.packet.type_ in HEADER_ONLY_TYPES:
                        # Header-only packet: there is no LENGTH, PAYLOAD or
                        # CRC field on the wire, so the packet is complete.
                        # N.B. an *empty* payload buffer is allocated so that
                        # `packet.data()` returns `b''` rather than raising.
                        self.payload_bytes_expected_ = 0
                        self.packet.alloc_buffer(0)
                        self.packet.payload_length_ = 0
                        self.packet.compute_crc()
                        self.message_completed_ = True
                        return

                    self.state = PacketState.LENGTH

            elif self.state == PacketState.LENGTH:
                if len(self.buffer) >= 2:  # LENGTH(2)
                    payload_length = struct.unpack('>H', bytes(self.buffer[0:2]))[0]
                    self.buffer = bytearray()

                    # Validate payload length *before* allocating anything.
                    if payload_length > self.max_payload_size:
                        _L().warning(f"Payload length too large: {payload_length} > {self.max_payload_size}")
                        self.parse_error_ = True
                        return

                    self.payload_bytes_expected_ = payload_length
                    # Always allocate (even for a zero-length payload) so that
                    # `packet.data()` is valid for every parsed packet.
                    self.packet.alloc_buffer(payload_length)
                    self.packet.payload_length_ = 0
                    if payload_length > 0:
                        self.state = PacketState.PAYLOAD
                    else:
                        self.packet.compute_crc()
                        self.state = PacketState.CRC

            elif self.state == PacketState.PAYLOAD:
                if len(self.buffer) >= self.payload_bytes_expected_:
                    try:
                        payload_data = self.buffer[:self.payload_bytes_expected_]
                        self.packet.set_data(payload_data)
                        self.state = PacketState.CRC
                        self.buffer = bytearray()
                    except ValueError as e:
                        _L().warning(f"Error setting payload: {e}")
                        self.parse_error_ = True
                        return

            elif self.state == PacketState.CRC:
                if len(self.buffer) >= 2:
                    try:
                        received_crc = struct.unpack('>H', bytes(self.buffer[:2]))[0]

                        if received_crc == self.packet.crc_:
                            self.message_completed_ = True
                        else:
                            # Mark as error but don't reset immediately
                            _L().debug(f"CRC mismatch: {received_crc:#06x} != {self.packet.crc_:#06x}")
                            self.parse_error_ = True
                        self.buffer = bytearray()
                    except struct.error as e:
                        _L().warning(f"Error parsing CRC: {e}")
                        self.parse_error_ = True
                        return

        except Exception as e:
            _L().exception(f"Unexpected error while parsing: {e}")
            self.parse_error_ = True
            return

    @property
    def message_completed(self) -> bool:
        return self.message_completed_

    @property
    def error(self) -> bool:
        return self.parse_error_

    @property
    def crc(self) -> int:
        return self.crc_


def parse_from_string(packet_str: Union[str, bytes]) -> Optional[FixedPacket]:
    """Parse a packet from a string or bytes object.
    
    Args:
        packet_str: Input string or bytes to parse
        
    Returns:
        FixedPacket if parsing successful, None otherwise

    Raises:
        TypeError: If input is neither string nor bytes

    Version log
    -----------
    .. versionchanged:: 0.54
        Return ``None`` (rather than ``False``) when no complete packet is
        found, matching the documented/annotated return type.
    """
    if not isinstance(packet_str, (str, bytes)):
        raise TypeError("Input must be string or bytes")
    if isinstance(packet_str, str):
        packet_str = packet_str.encode('utf-8')
    parser = cPacketParser()
    result = parser.parse(np.frombuffer(packet_str, dtype='uint8'))
    return None if result is False else result


def byte_pair(value: int) -> tuple:
    """Split a 16-bit value into two bytes."""
    return (value >> 8) & 0xFF, value & 0xFF


class FixedSizeBufferPool:
    """Provide pooled allocation of fixed size buffers."""
    
    def __init__(self, buffer_size: int, count: int):
        self.buffer_size = buffer_size
        self.count = count
        self.padded_buffer_size = buffer_size + 8  # size_t for buffer ID
        self.super_buffer = bytearray(count * self.padded_buffer_size)
        self.occupied = [False] * count
        self.next_free = 0
        self.free_count = count
        
        # Initialize buffer IDs
        for i in range(count):
            offset = i * self.padded_buffer_size
            struct.pack_into('>Q', self.super_buffer, offset, i)
    
    def alloc(self) -> Optional[bytearray]:
        """Allocate a buffer from the pool."""
        if self.free_count == 0:
            return None
        
        while self.occupied[self.next_free]:
            self.next_free = (self.next_free + 1) % self.count
            
        buffer_start = self.next_free * self.padded_buffer_size + 8
        buffer_end = buffer_start + self.buffer_size
        self.occupied[self.next_free] = True
        self.free_count -= 1
        
        return self.super_buffer[buffer_start:buffer_end]
    
    def free(self, buffer: bytearray):
        """Free a buffer back to the pool."""
        buffer_id = struct.unpack('>Q', self.super_buffer[buffer.start - 8:buffer.start])[0]
        if buffer_id < self.count and self.occupied[buffer_id]:
            self.occupied[buffer_id] = False
            self.free_count += 1
    
    def available(self) -> int:
        """Return number of available buffers."""
        return self.free_count


T = TypeVar('T')

class DequeNode(Generic[T]):
    """Node in a double-ended queue."""
    
    def __init__(self, value: T):
        self.value = value
        self.prev = None
        self.next = None


class Deque(Generic[T]):
    """Double-ended queue implementation."""
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.item_count = 0
    
    def append(self, item: T):
        """Add item to end of deque."""
        node = DequeNode(item)
        if self.empty():
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        self.item_count += 1
    
    def push(self, item: T):
        """Add item to front of deque."""
        node = DequeNode(item)
        if self.empty():
            self.head = node
            self.tail = node
        else:
            self.head.prev = node
            node.next = self.head
            self.head = node
        self.item_count += 1
    
    def pop_tail(self) -> T:
        """Remove and return item from end of deque."""
        if self.empty():
            raise IndexError("Deque is empty")
        item = self.tail.value
        self.tail = self.tail.prev
        if self.tail:
            self.tail.next = None
        self.item_count -= 1
        return item
    
    def pop_head(self) -> T:
        """Remove and return item from front of deque."""
        if self.empty():
            raise IndexError("Deque is empty")
        item = self.head.value
        self.head = self.head.next
        if self.head:
            self.head.prev = None
        self.item_count -= 1
        return item
    
    def empty(self) -> bool:
        """Return True if deque is empty."""
        return self.item_count == 0
    
    def size(self) -> int:
        """Return number of items in deque."""
        return self.item_count


class BoundedDeque(Deque[T]):
    """Deque with maximum size limit."""
    
    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size
    
    def append(self, item: T) -> bool:
        """Add item to end if not full."""
        if not self.full():
            super().append(item)
            return True
        return False
    
    def push(self, item: T) -> bool:
        """Add item to front if not full."""
        if not self.full():
            super().push(item)
            return True
        return False
    
    def full(self) -> bool:
        """Return True if deque is at maximum size."""
        return self.size() == self.max_size


class CircularBuffer:
    """Circular buffer implementation."""
    
    def __init__(self, size: int):
        self.buffer = bytearray(size)
        self.size = size
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
    
    def push(self, value: int) -> bool:
        """Add byte to buffer if not full."""
        if self.count < self.size:
            self.buffer[self.write_pos] = value
            self.write_pos = (self.write_pos + 1) % self.size
            self.count += 1
            return True
        return False
    
    def pop(self, value: bytearray) -> bool:
        """Remove and return byte from buffer if not empty."""
        if self.count > 0:
            value[0] = self.buffer[self.read_pos]
            self.read_pos = (self.read_pos + 1) % self.size
            self.count -= 1
            return True
        return False
    
    def available(self) -> int:
        """Return number of bytes available to read."""
        return self.count


class PacketAllocator:
    """Factory for packet allocation using a buffer pool."""
    
    def __init__(self, buffer_size: int = 128, count: int = 10):
        self.buffer_allocator = FixedSizeBufferPool(buffer_size, count)
    
    def create_packet(self) -> FixedPacket:
        """Create a new packet with allocated buffer."""
        packet = FixedPacket()
        buffer = self.buffer_allocator.alloc()
        if buffer is not None:
            packet.reset_buffer(len(buffer), buffer)
        return packet
    
    def free_packet_buffer(self, packet: FixedPacket):
        """Free packet's buffer back to pool."""
        if packet.payload_buffer_:
            self.buffer_allocator.free(packet.payload_buffer_)
    
    def available(self) -> int:
        """Return number of available buffers."""
        return self.buffer_allocator.available()


class PacketStream:
    """Stream interface for reading packets."""
    
    def __init__(self, packet_allocator: PacketAllocator, max_queue_length: int = 1024):
        self.allocator = packet_allocator
        self.packet_queue = BoundedDeque[FixedPacket](max_queue_length)
        self.data = None
        self.bytes_available = 0
    
    def packet_available(self) -> int:
        """Return number of bytes left unread in active packet."""
        if self.packet_queue.empty() or self.packet_queue.tail.value.payload_length_ == 0:
            return 0
        return len(self.packet_queue.tail.value.payload_buffer_) - (self.data - self.packet_queue.tail.value.payload_buffer_)
    
    def prepare_active_packet(self):
        """Prepare next packet for reading."""
        while not self.packet_queue.empty() and self.packet_available() == 0:
            self.allocator.free_packet_buffer(self.packet_queue.pop_tail())
            if not self.packet_queue.empty():
                self.data = self.packet_queue.tail.value.payload_buffer_
    
    def push(self, packet: FixedPacket) -> bool:
        """Add packet to queue."""
        self.bytes_available += packet.payload_length_
        if self.data is None:
            self.data = packet.payload_buffer_
        return self.packet_queue.append(packet)
    
    def available(self) -> int:
        """Return number of bytes available to read."""
        return self.bytes_available
    
    def read(self) -> int:
        """Read next byte from stream."""
        if self.available() <= 0:
            return -1
        
        self.prepare_active_packet()
        value = self.data[0]
        self.data = self.data[1:]
        self.bytes_available -= 1
        
        if self.packet_available() == 0:
            self.prepare_active_packet()
        
        return value

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        while not self.packet_queue.empty():
            self.allocator.free_packet_buffer(self.packet_queue.pop_tail())
        self.data = None
        self.bytes_available = 0


class PacketSocket:
    """Base class for packet socket implementation."""
    
    def __init__(self) -> None:
        self.idle_state: int = -1
        self.cs: int = 0
    
    def idle(self) -> bool:
        """Return True if socket is idle."""
        return self.idle_state >= 0 and self.cs == self.idle_state
    
    def state(self) -> int:
        """Return current state."""
        return self.cs
    
    def set_state(self, value: int) -> None:
        """Set current state."""
        self.cs = value
    
    def reset(self):
        """Reset socket state."""
        self.cs = 0
    
    def parse_byte(self, byte: int):
        """Parse incoming byte."""
        pass


@dataclass
class PacketConfig:
    """Configuration for packet handling."""
    buffer_size: int = 128
    max_queue_length: int = 1024
    event_queue_length: int = 256
    rx_queue_length: int = 64
    tx_queue_length: int = 64


class StreamPacketSocket(PacketSocket):
    """Socket for streaming packets."""
    
    def __init__(self, parser: cPacketParser, allocator: PacketAllocator,
                 config: PacketConfig = PacketConfig()):
        super().__init__()
        self.parser = parser
        self.allocator = allocator
        self.rx_queue = BoundedDeque[FixedPacket](config.rx_queue_length)
        self.tx_queue = BoundedDeque[FixedPacket](config.tx_queue_length)
        self.event_queue = CircularBuffer(config.event_queue_length)
        self.parser_packet = allocator.create_packet()
        self.rx_packet = None
        self.tx_packet = None
        
        self.parser.reset(self.parser_packet)
        self.push_event(b'i')
    
    def push_event(self, event: bytes) -> bool:
        """Add event to queue."""
        return self.event_queue.push(event[0])
    
    def process_rx_packet(self):
        """Process received packet."""
        if self.rx_queue.empty():
            self.push_event(b'N')
            return
        
        self.rx_packet = self.rx_queue.pop_tail()
        self.allocator.free_packet_buffer(self.rx_packet)
        self.push_event(b'q')
    
    def process_tx_packet(self):
        """Process packet for transmission."""
        if self.tx_queue.empty():
            self.push_event(b'N')
            return
        
        self.tx_packet = self.tx_queue.pop_tail()
        self.allocator.free_packet_buffer(self.tx_packet)
        self.push_event(b's')
    
    def handle_data_packet(self):
        """Handle data packet."""
        if self.rx_queue.full():
            self.push_event(b'f')
        else:
            self.push_event(b'r')
    
    def handle_ack_packet(self):
        """Handle acknowledgment packet."""
        self.parser.reset(self.parser_packet)
        self.push_event(b'Y')
    
    def handle_nack_packet(self):
        """Handle negative acknowledgment packet."""
        self.parser.reset(self.parser_packet)
        self.push_event(b'N')


class CommandProcessor:
    """Process commands from packets."""
    
    def process_command(self, request: bytes, buffer_size: int) -> bytes:
        """Process command and return response."""
        # This is a simplified version - extend based on your needs
        return request


class CommandPacketHandler:
    """Handle command packets."""
    
    def __init__(self, output_stream: BytesIO, command_processor: CommandProcessor):
        self.output_stream = output_stream
        self.command_processor = command_processor
    
    def _validate_packet(self, packet: FixedPacket) -> bool:
        """Validate packet before processing."""
        if packet is None:
            _L().error("Received null packet")
            return False
        if packet.type_ not in PACKET_NAME_BY_TYPE:
            _L().error(f"Invalid packet type: {packet.type_}")
            return False
        if packet.payload_length_ > packet.max_buffer_size:
            _L().error(f"Payload too large: {packet.payload_length_}")
            return False
        return True

    def process_packet(self, packet: FixedPacket):
        """Process packet containing command."""
        if not self._validate_packet(packet):
            return
        
        if packet.type_ == PACKET_TYPE.DATA and packet.payload_length_ > 0:
            response = self.command_processor.process_command(
                packet.payload_buffer_[:packet.payload_length_],
                packet.buffer_size_
            )
            packet.set_data(response)
        self.write_packet(packet)
    
    def write_packet(self, packet: FixedPacket):
        """Write packet to output stream."""
        self.output_stream.write(packet.tobytes())


class cPacket(FixedPacket):
    """Python implementation of the Cython cPacket wrapper"""
    def __init__(self, type_=PACKET_TYPE.NONE, iuid=0, data=None,
                 buffer_=None, buffer_size=None):
        super().__init__(type_, iuid, data, buffer_, buffer_size)

    def reset_buffer(self, buffer_size: int, buffer_: Union[bytes, bytearray]):
        """Reset buffer with new size and data."""
        self.clear_buffer()
        # N.B. `set_buffer()` assigns `buffer_size_` itself, so the explicit
        # size must be applied *afterwards* (otherwise it is overwritten).
        self.set_buffer(buffer_, overwrite=True)
        self.buffer_size_ = buffer_size
