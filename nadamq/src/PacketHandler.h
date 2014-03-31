#ifndef ___PACKET_HANDLER__HPP___
#define ___PACKET_HANDLER__HPP___

#ifndef AVR
/* Assume STL libraries are not available on AVR devices, so don't include
 * methods using them when targeting AVR architectures. */
#include <iostream>
#else
#include "output_buffer.h"
#endif

#include "PacketParser.h"


template <typename Parser, typename Stream>
class StreamPacketParser {
  /* # `StreamPacketParser` #
   *
   * Provide the following API described below to read full packets from a
   * stream.
   *
   * ## Accessors ##
   *
   *  - `available()`: `true` if there is data ready on the stream.
   *  - `ready()`: `true` if a packet has been parsed successfully
   *    _(accessible through the `packet()` method)_.
   *  - `error()`: Returns error code if there was an error parsing the
   *    current packet.
   *   - `L`: Data too large for buffer.
   *   - `e`: Parse error, e.g., failed checksum.
   *  - `packet(Packet &output)`: Copy current packet contents to `output`.
   *
   * ## Mutators ##
   *
   *  - `read_available()`: Parse available characters from the stream.  Stop
   *    parsing if either a full packet has been parsed, or if an error is
   *    encountered.
   *  - `reset()`: Reset state of parser.  __NB__ After parsing a complete
   *    packet or encountering an error, this method must be called before
   *    `parse_available()` will parse any new characters. */
public:
  /* Infer type of packet from `Parser` type. */
  typedef typename Parser::packet_type packet_type;

  Parser &parser_;
  Stream &stream_;

  StreamPacketParser(Parser &parser, Stream &stream)
    : parser_(parser), stream_(stream) {}

  virtual bool ready() { return parser_.message_completed_; }
  virtual uint8_t error() {
    if (parser_.parse_error_) {
      return 'e';
    }
  }
  virtual void reset() { parser_.reset(); }
  virtual packet_type const &packet() const { return *parser_.packet_; }

  virtual int available() { return stream_.available(); }

  virtual int parse_available() {
    /* We have encountered an error, or we have parsed a full packet
     * successfully.  The state of the parser must be explicitly reset using
     * the `reset()` method before we continue parsing. */
    if (ready() || error()) { return 0; }

    int bytes_read = 0;
    while (stream_.available() > 0) {
      uint8_t byte = stream_.read();
      parser_.parse_byte(&byte);
      bytes_read++;
      /* Continue parsing until we have encountered an error, or we have parsed
       * a full packet successfully. */
      if (ready() || error()) { break; }
    }
    return bytes_read;
  }
};


template <typename Parser, typename Stream>
class PacketHandlerBase : public StreamPacketParser<Parser, Stream> {
  /* # `PacketHandlerBase` #
   *
   * Extend the `StreamPacketParser` API to read as many full packets as are
   * available from a stream for each call to `parse_available()` _(as opposed
   * to stopping after each completed packet or error)_.  This implements a
   * "reactor"-style parser using handler methods for handling success and
   * error cases.
   *
   * ## New methods ##
   *
   *  - `handle_packet(packet_type &packet)`: Handle a successfully parsed packet.
   *  - `handle_error(packet_type &packet)`: Handle an error, provided with the
   *    contents of the incomplete parsed packet. */
public:
  typedef StreamPacketParser<Parser, Stream> base_type;
  typedef typename base_type::packet_type packet_type;

  using base_type::parser_;
  using base_type::stream_;
  using base_type::ready;
  using base_type::error;
  using base_type::reset;

  PacketHandlerBase(Parser &parser, Stream &stream)
    : base_type(parser, stream) {}

  /* No default behaviour for handling a successfully parse packet.  This
   * method must be implemented in a concrete sub-class. */
  virtual void handle_packet(packet_type &packet) = 0;
  /* Enable special handling of parse errors. */
  virtual void handle_error(packet_type &packet) {}

  virtual int parse_available() {
    int bytes_read = 0;
    while (stream_.available() > 0) {
      bytes_read += base_type::parse_available();

      /* If we've encountered an error or successfully parsed a packet, call the
       * corresponding handler method. */
      if(error()) {
        /* Parse error */
        handle_error(*(parser_.packet_));
        /* Reset parser to process next packet. */
        reset();
      } else if (ready()) {
        /* # Process packet #
          *
          * Do something with successfully parsed packet.
          *
          * Note that the packet should be copied if it will be needed
          * afterwards, since the packet will be reset below, before the next
          * input scan. */
        handle_packet(*(parser_.packet_));
        /* Reset parser to process next packet. */
        reset();
      }
    }
    return bytes_read;
  }
};


#ifdef AVR
template <typename Parser, typename IStream, typename OStream>
class VerbosePacketHandler : public PacketHandlerBase<Parser, IStream> {
  /* # `VerbosePacketHandler` _(`AVR`-variant, Arduino-compatible)_ #
   *
   * This class implements a simple packet parsing reactor, which dumps a
   * message to the provided output stream whenever either a packet is
   * successfully parsed or an error is encountered. */
  public:

  typedef PacketHandlerBase<Parser, IStream> base_type;
  typedef typename base_type::packet_type packet_type;

  using base_type::parser_;

  OStream &ostream_;

  VerbosePacketHandler(Parser &parser, IStream &istream, OStream &ostream)
    : base_type(parser, istream), ostream_(ostream) {}

  virtual void handle_packet(packet_type &packet) {
    ostream_.println(P("# success #"));
    ostream_.println("");
    ostream_.print("type: ");
    ostream_.println(packet.type_);
    ostream_.print("payload length: ");
    ostream_.println(packet.payload_length_);
    ostream_.print("payload: '");
    for (int i = 0; i < packet.payload_length_; i++) {
      ostream_.print((char)packet.payload_buffer_[i]);
    }
    ostream_.println("'");
    ostream_.println("");
  }

  virtual void handle_error(packet_type &packet) {
    ostream_.println("# ERROR #");
  }
};
#else
template <typename Parser, typename Stream>
class VerbosePacketHandler : public PacketHandlerBase<Parser, Stream> {
  /* # `VerbosePacketHandler` _(STL-variant)_ #
   *
   * This class implements a simple packet parsing reactor, which dumps a
   * message to the provided output stream whenever either a packet is
   * successfully parsed or an error is encountered. */
  public:

  typedef PacketHandlerBase<Parser, Stream> base_type;
  typedef typename base_type::packet_type packet_type;

  using base_type::parser_;

  VerbosePacketHandler(Parser &parser, Stream &stream)
    : base_type(parser, stream) {}

  virtual void handle_packet(packet_type &packet) {
    std::cout << "# Packet parsed successfully #" << std::endl;
    std::cout << std::setw(24) << "iuid: " << std::dec
              << static_cast<int>(0x00FFFF & packet.iuid_) << std::endl;
    std::cout << std::setw(24) << "type: " << std::hex
              << static_cast<int>(0x00FF & packet.type()) << std::endl;
    std::cout << std::setw(24) << "payload length: "
              << packet.payload_length_ << std::endl;
    std::cout << std::setw(24) << "payload: " << "'";
    std::copy(packet.payload_buffer_, packet.payload_buffer_ +
              packet.payload_length_,
              std::ostream_iterator<uint8_t>(std::cout, ""));
    std::cout << "'" << std::endl;
  }

  virtual void handle_error(packet_type &packet) {
    std::cout << "# ERROR: Parse error #" << std::endl;
  }
};
#endif  // #ifdef AVR


#endif  // #ifndef ___PACKET_HANDLER__HPP___
