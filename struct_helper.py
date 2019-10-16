import re
import io
import struct
import collections


class BinaryData:
    def __init__(self, name, fields, fmt):
        self.converter = struct.Struct(fmt)
        self.data_ctor = collections.namedtuple(name, fields)

    def get_size(self):
        return self.converter.size

    def read(self, src_file, restore_file_pos):
        bin_data = src_file.read(self.converter.size)
        if len(bin_data) != self.converter.size:
            return None
        data = self.converter.unpack(bin_data)
        obj = self.data_ctor._make(data)
        if restore_file_pos:
            src_file.seek(-self.converter.size, io.SEEK_CUR)
        return obj

    def write(self, dst_file, *args, **kwds):
        blob = self.pack(*args, **kwds)
        dst_file.write(blob)

    def recv(self, soc):
        bin_data = soc.recv(self.converter.size)
        if len(bin_data) != self.converter.size:
            return None
        data = self.converter.unpack(bin_data)
        obj = self.data_ctor._make(data)
        return obj

    def pack(self, *args, **kwds):
        obj = self.data_ctor(*args, **kwds)
        blob = self.converter.pack(*obj)
        return blob

    def unpack(self, buffer, offset=0):
        data = self.converter.unpack_from(buffer, offset)
        obj = self.data_ctor._make(data)
        return obj


_type2idx = {
    'BYTE':  'B',
    'DWORD': 'L',
    'WORD':  'H',
    'SHORT': 'h',
    'LONG':  'l',
    'long':  'l',
    'char':  's',
    '__int64': 'q'
}

def _make_type_desc(t, l):
    if l:
        assert(t in ('char', 'BYTE'))
        return l + 's'
    else:
        return _type2idx[t]


def make_struct(desc):
    p = re.match('\s*(?:class|struct)\s+(?P<name>\w+)\s*\{(?P<items>[^}]+)};+', desc)
    if not p:
        raise Exception("bad struct definition:" + desc)
    name = p.group('name')
    line_pattern = '\s*(?P<type>\w+)\s+(?P<name>\w+)\s*(?:\[(?P<len>\d+)\])?\s*;'
    desc = [(g.group('type', 'len'), g.group('name')) for g in re.finditer(line_pattern, p.group('items'))]
    types, fields = zip(*desc)
    fmt = '<' + ''.join((_make_type_desc(*t) for t in types))
    return BinaryData(name, fields, fmt)


def test():
    writer = make_struct("""
        struct OpenData {
            LONG        id;
            char[30]    name;
        }; """)
    blob = writer.write(14, '[494]_2012_04_05_02'.encode())
    print(' '.join(map(hex, blob)))


if __name__ == '__main__':
    test()

