"""Extract executable .text from a little-endian ELF32 TPC object."""
import struct
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_bytes()
if raw[:6] != b"\x7fELF\x01\x01":
    raise ValueError("Expected little-endian ELF32")
offset = struct.unpack_from("<I", raw, 32)[0]
entry_size, count, names_index = struct.unpack_from("<HHH", raw, 46)
sections = [struct.unpack_from("<IIIIIIIIII", raw, offset + i * entry_size) for i in range(count)]
names_header = sections[names_index]
names = raw[names_header[4]:names_header[4] + names_header[5]]
for section in sections:
    name = names[section[0]:].split(b"\0", 1)[0]
    if name == b".text":
        Path(sys.argv[2]).write_bytes(raw[section[4]:section[4] + section[5]])
        break
else:
    raise ValueError("Missing .text")
