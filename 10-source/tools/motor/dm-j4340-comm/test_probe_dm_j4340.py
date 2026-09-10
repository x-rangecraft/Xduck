import struct
import unittest

from probe_dm_j4340 import (
    AdapterFrame,
    FrameStream,
    Register,
    build_register_read,
    decode_register_reply,
    parse_ids,
)


class ProtocolTest(unittest.TestCase):
    def test_register_read_matches_vendor_u2can_layout(self):
        frame = build_register_read(0x123, 0x0E)
        self.assertEqual(len(frame), 30)
        self.assertEqual(frame[:5], bytes([0x55, 0xAA, 0x1E, 0x03, 0x01]))
        self.assertEqual(frame[13:15], bytes([0xFF, 0x07]))
        self.assertEqual(frame[21:29], bytes([0x23, 0x01, 0x33, 0x0E, 0, 0, 0, 0]))

    def test_fragmented_reply_is_parsed_and_decoded(self):
        payload = bytes([0x02, 0x00, 0x33, 0x3C]) + struct.pack("<f", 24.5)
        raw = bytes([0xAA, 0x11, 0x00, 0x12, 0, 0, 0]) + payload + bytes([0x55])
        parser = FrameStream()
        self.assertEqual(list(parser.feed(b"noise" + raw[:6])), [])
        frames = list(parser.feed(raw[6:]))
        self.assertEqual(len(frames), 1)
        reply = decode_register_reply(frames[0], 2, Register(0x3C, "bus_voltage", "f32"))
        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertAlmostEqual(reply.value, 24.5)
        self.assertEqual(reply.response_can_id, 0x12)

    def test_wrong_motor_reply_is_ignored(self):
        frame = AdapterFrame(0x11, 0x11, bytes([2, 0, 0x33, 7, 0x11, 0, 0, 0]), b"x" * 16)
        self.assertIsNone(decode_register_reply(frame, 1, Register(7, "master_id", "u32")))

    def test_id_list_and_ranges(self):
        self.assertEqual(parse_ids("1,0x03,4-5,3"), [1, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
