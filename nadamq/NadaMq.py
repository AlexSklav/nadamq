import struct
import enum
from typing import Optional, Union, TypeVar, Generic, List
from collections import deque
import numpy as np
from io import BytesIO


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

PACKET_NAME_BY_TYPE = {
    PACKET_TYPE.NONE: 'NONE',
    PACKET_TYPE.ACK: 'ACK',
    PACKET_TYPE.NACK: 'NACK',
    PACKET_TYPE.DATA: 'DATA',
    PACKET_TYPE.STREAM: 'STREAM',
    PACKET_TYPE.ID_REQUEST: 'ID_REQUEST',
    PACKET_TYPE.ID_RESPONSE: 'ID_RESPONSE'
}


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
        self.buffer_size_ = 6
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
                buffer_size = 6 + max(len(data), 1) + 4  # Add 2 bytes for CRC and 2 bytes for length

        if buffer_size is not None and buffer_size > 0:
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
        return self.tostring()

    def tostring(self) -> bytes:
        try:
            # Format: START + IUID + TYPE + [LENGTH + PAYLOAD + CRC if payload exists]
            header = struct.pack('>H', self.iuid_)  # IUID (2 bytes)
            header += struct.pack('B', self._type)  # Type (1 byte)

            if self.buffer_size_ == 6:
                packet = FLAGS.START + header
                return packet
            else:
                header += struct.pack('>H', self.payload_length_)  # Length (2 bytes)
                packet = FLAGS.START + header
                if self.payload_length_ > 0:
                    packet += self.data()
                packet += struct.pack('>H', self.crc_)
                return packet

        except Exception as e:
            print(f"Error converting packet to string: {e}")
            return b''

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
        self.buffer_size_ = 6
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
        self.buffer_size_ = len(data) + 4  # Add 2 bytes for CRC and 2 bytes for length
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


class cPacketParser:
    """Parser for fixed-size packets with CRC validation.
    
    The parser implements a state machine to process incoming bytes and construct
    complete packets. The packet format is:
    START(3) + IUID(2) + TYPE(1) + [LENGTH(2) + PAYLOAD(n) + CRC(2)]
    
    Attributes:
        payload_bytes_received_: Number of payload bytes received
        payload_bytes_expected_: Expected payload length
        message_completed_: True if a complete message has been parsed
        parse_error_: True if an error occurred during parsing
        state: Current state of the parser ('START', 'HEADER', etc.)
    """
    def __init__(self, buffer_size: int = 8 << 10):
        self.payload_bytes_received_ = 0
        self.payload_bytes_expected_ = 0
        self.message_completed_ = False
        self.parse_error_ = False
        self.crc_ = 0
        self.packet = FixedPacket(buffer_size=buffer_size)
        self.state = 'START'
        self.buffer = bytearray()
        self.reset()

    def reset(self) -> None:
        """Reset parser state and packet."""
        self.state = 'START'
        self.buffer = bytearray()
        self.message_completed_ = False
        self.parse_error_ = False
        self.payload_bytes_received_ = 0
        self.payload_bytes_expected_ = 0
        self.crc_ = 0
        if hasattr(self.packet, 'clear_buffer'):
            self.packet.clear_buffer()

    def parse(self, data: Union[bytes, bytearray, np.ndarray]) -> Union[FixedPacket, bool]:
        """Parse packet data and return packet if complete."""
        for i, byte in enumerate(data):
            self.parse_byte(byte)
            if self.message_completed_ or (i == len(data)-1 == 5): # handle the case where the message is short START(3) + IUID(2) + TYPE(1)
                self.packet.buffer_size_ = i+1
                packet = self.packet
                # Create a new packet for next message
                self.packet = FixedPacket()
                self.reset()
                return packet
            elif self.parse_error_:
                print(f'Error parsing packet - resetting and continuing')
                # Reset state but continue processing
                self.reset()
                self.parse_error_ = False
        return False

    def parse_byte(self, byte: int) -> None:
        """Parse a single byte."""
        try:
            self.buffer.append(byte)
            
            if self.state == 'START':
                if len(self.buffer) >= 3:
                    if bytes(self.buffer[-3:]) == FLAGS.START:
                        self.state = 'HEADER'
                        self.buffer = bytearray()

            elif self.state == 'HEADER':  # IUID(2) + TYPE(1)
                if len(self.buffer) >= 3:
                    self.packet.iuid_ = struct.unpack('>H', self.buffer[0:2])[0]
                    self.packet.type_ = self.buffer[2]
                    self.state = 'LENGTH'
                    self.buffer = bytearray()

                    # Validate header
                    if self.packet.type_ not in PACKET_NAME_BY_TYPE:
                        print(f"Invalid packet type: {hex(self.packet.type_)}")
                        self.parse_error_ = True
                        return

            elif self.state == 'LENGTH':
                if len(self.buffer) >= 2:  # LENGTH(2)
                    self.payload_bytes_expected_ = struct.unpack('>H', self.buffer[0:2])[0]

                    if self.payload_bytes_expected_ > 0:
                        self.packet.alloc_buffer(self.payload_bytes_expected_)
                        self.state = 'PAYLOAD'
                    else:
                        self.packet.payload_length_ = 0
                        self.state = 'CRC'
                    self.buffer = bytearray()

                    # Validate payload length
                    if self.payload_bytes_expected_ > self.packet.max_buffer_size:
                        print(f"Payload length too large: {self.payload_bytes_expected_} > {self.packet.max_buffer_size}")
                        self.parse_error_ = True
                        return

            elif self.state == 'PAYLOAD':
                if len(self.buffer) >= self.payload_bytes_expected_:
                    try:
                        payload_data = self.buffer[:self.payload_bytes_expected_]
                        self.packet.set_data(payload_data)
                        self.state = 'CRC'
                        self.buffer = bytearray()
                    except ValueError as e:
                        print(f"Error setting payload: {e}")
                        self.parse_error_ = True
                        return

            elif self.state == 'CRC':
                if len(self.buffer) >= 2:
                    try:
                        received_crc = struct.unpack('>H', bytes(self.buffer[:2]))[0]

                        if received_crc == self.packet.crc_:
                            self.message_completed_ = True
                        else:
                            # Mark as error but don't reset immediately
                            self.parse_error_ = True
                        self.buffer = bytearray()
                    except struct.error as e:
                        print(f"Error parsing CRC: {e}")
                        self.parse_error_ = True
                        return

        except Exception as e:
            print(f"Unexpected error while parsing: {e}")
            # Catch any other errors and continue processing
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
    """
    if not isinstance(packet_str, (str, bytes)):
        raise TypeError("Input must be string or bytes")
    if isinstance(packet_str, str):
        packet_str = packet_str.encode('utf-8')
    parser = cPacketParser()
    return parser.parse(np.array([v for v in packet_str], dtype='uint8'))


def byte_pair(value: int) -> tuple:
    """Split a 16-bit value into two bytes."""
    return chr((value >> 8) & 0xFF), chr(value & 0xFF)


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


class StreamPacketSocket(PacketSocket):
    """Socket for streaming packets."""
    
    def __init__(self, parser: cPacketParser, allocator: PacketAllocator,
                 event_queue_length: int, rx_queue_length: int, tx_queue_length: int):
        super().__init__()
        self.parser = parser
        self.allocator = allocator
        self.rx_queue = BoundedDeque[FixedPacket](rx_queue_length)
        self.tx_queue = BoundedDeque[FixedPacket](tx_queue_length)
        self.event_queue = CircularBuffer(event_queue_length)
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
    
    def process_packet(self, packet: FixedPacket):
        """Process packet containing command."""
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
        self.buffer_size_ = buffer_size
        self.set_buffer(buffer_, overwrite=True)