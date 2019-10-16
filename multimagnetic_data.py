"""@package docstring

работа с многоканальными магнитными данными, чтание запись
"""

import enum
import functools
import struct
import zlib
from io import BytesIO
import xml.etree.ElementTree as ET


import numpy as np

import struct_helper

_struct_file_hdr = struct_helper.make_struct('''
    struct SFileHeader
    {
        char       	sign[4];						///< сигнатура начала файла = eSign
        BYTE		nVersion;					///< номер версии
        BYTE		flags;						///< флаги
        BYTE		reserve1[10];
    };''')

struct_block_hdr = struct_helper.make_struct('''
    struct SBlockHeader
    {
        char	    sign[4];			///< сигнатура блока = eSign
        __int64 	prev;				///< позиция предудущего блока SBlockHeader
        __int64 	next;				///< позиция следующего блока SBlockHeader
        BYTE		type;				///< тип содержащихся данных eBlockType
        BYTE		reserv[3];
    };''')

_struct_block_format = struct_helper.make_struct('''
    struct SBlockFormat
    {
        char	    sign[4];			///< сигнатура блока описания = eSign
        BYTE		nCount;				///< количество блоков с описанием каналов, SChannelFormat
        BYTE		reserv[3];
    };''')

_struct_channel_format = struct_helper.make_struct('''
    struct SChannelFormat
    {
        char	    sign[4];				///< сигнатура блока описания = eSign
        WORD		nStepMM;			///< шаг сбора данных в мм, количество данных в блоке не должно превышать nDataBlockLenMM/wStepMM
        BYTE		nRail;				///< номер рельса
        BYTE		nChannel;			///< номер канала
        BYTE		nBytePerSample;		///< количество байт на отсчет
        BYTE		reserved[1];
        WORD		nTextLen;			///< длинна в байтах строки с описанием канала, включая \0
    };''')

_reader_block_offsets = struct_helper.make_struct('''
    struct SBlockOffset
    {
        char	    sign[4];				///< сигнатура блока = eSign
        DWORD	    dwBaseCoord;			///< координата первого блока данных сохраненного за этой таблицей, координата должна быть кратна eBlockLenght * nBlockStep * nTableSize
        __int64 	oftNextIndexBlock;		///< смещение SBlockHeader со следующей таблицей индексов

        BYTE		nBlockStep;				///< количество блоков данных между сохранениями
        BYTE		nTableSize;				///< количество записей в таблице
        BYTE		reserv[2];
    };''')

struct_block_data = struct_helper.make_struct('''
    struct SBlockData
    {
        char	    sign[4];				///< сигнатура блока с данными= eSign
        DWORD   	dwCoord;			///< системная координата
        BYTE		nCount;				///< количество расположенных следом SData
        BYTE		flags;				///< набор флагов, EFlags
        BYTE		reserved[2];
    }; ''')

_struct_channel_data = struct_helper.make_struct('''
    struct SChannelData
    {
        char	    sign[4];			///< сигнатура блока с данными= eSign
        WORD		nDataSizeByte;		///< размер данных в блоке, количество отсчетов = nDataBlockSizeByte / nBytePerSample

        BYTE		nRail;
        BYTE		nChannel;
        BYTE		nReserved[4];
    };''')


class BlockType(enum.Enum):
    EMPTY = 0
    DATA = 1
    DESCRIPTION = 2
    OFFSETS = 3



class obj(object):
    """ преобразование словаря в объект """

    def __init__(self, **kwargs):
        for a, b in kwargs.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


class _Joiner:
    """ склейка диапазонов """

    def __init__(self, start, end, dtype):
        """ задается начало, конец и тип выходной массив """
        self.start = start
        self.dst = np.zeros(end - start, dtype=dtype)

    def append(self, coord, src):
        """ скопировать на нужное место новый кусок данных """
        diff = coord - self.start
        left = (coord + len(src)) - (self.start + len(self.dst))
        if diff < 0:
            if left < 0:
                self.dst[:left] = src[-diff:]
            elif left > 0:
                self.dst[:] = src[-diff:-left]
            else:
                self.dst[:] = src[-diff:]
        else:
            if left < 0:
                self.dst[diff:left] = src[:]
            elif left > 0:
                self.dst[diff:] = src[:-left]
            else:
                self.dst[diff:] = src[:]


class WrongSignature(Exception):
    pass


class Reader:
    """ чтение данных """

    def __init__(self, path):
        self._path = path
        self._file = open(self._path, 'rb')
        hdr_file = _struct_file_hdr.read(self._file, False)
        if hdr_file.sign != b'FHDR':
            raise WrongSignature('file header')
        self._formats = self._scan_format()
        self._offsets = self._scan_offsets()
        self._range = self._scan_range()

    def get_format(self):
        """ получить описанеи данных """
        return self._formats

    def get_range(self):
        """ получить диапазон данных """
        return self._range

    def get_data(self, channels, req_range):
        """ получить данные по запрошенным каналам на запрошенный диапазон """
        req_range[0] = max(req_range[0], self._range[0])
        req_range[1] = min(req_range[1], self._range[1])
        if req_range[0] > req_range[1] or not channels:
            return [], []

        res = [_Joiner(req_range[0], req_range[1], dtype=np.int16) for _ in channels]
        for coord in range(req_range[0] // 0x1000 * 0x1000, req_range[1] + 1, 0x1000):
            block = self.get_block(coord)
            if not block:
                continue
            for i, (r, c) in enumerate(channels):
                src = block.get((r, c), None)
                dst = res[i]
                if src is None:
                    continue
                dst.append(coord, src)
        return np.arange(req_range[0], req_range[1]), [j.dst for j in res]

    @functools.lru_cache(1000)  # 1000 блоков ~ 4 км
    def get_block(self, coord, supply_missing_samples=True):
        """
        получить блок данных по координате
        :param coord координата запрашиваемого блока
        :param supply_missing_samples заполнить недостающие отсчеты в блоке нулями
        """
        assert coord % 0x1000 == 0
        offset = self._get_offset(coord)
        for hdr_block in self.enum_blocks(offset, True):
            if hdr_block.type != BlockType.DATA.value:
                continue
            hdr_data = struct_block_data.read(self._file, False)
            if hdr_data.sign != b'DBLC':
                raise WrongSignature('data block header')
            if hdr_data.dwCoord < coord:
                continue
            if hdr_data.dwCoord > coord:
                break
            assert hdr_data.flags & 0x01
            len_pack, len_unpacked = struct.unpack('LL', self._file.read(8))
            data_unpacked = zlib.decompress(self._file.read(len_pack))
            assert len(data_unpacked) == len_unpacked
            data_block_stream = BytesIO(data_unpacked)
            res = {}
            for c in range(hdr_data.nCount):
                hdr_channel_data = _struct_channel_data.read(data_block_stream, False)
                if hdr_channel_data.sign != b'CDTA':
                    raise WrongSignature('channel data block header')
                channel_data = data_block_stream.read(hdr_channel_data.nDataSizeByte)
                key = (hdr_channel_data.nRail, hdr_channel_data.nChannel)
                fmt = self._formats.get(key)
                if fmt:
                    if fmt.bps == 2:
                        dt = np.dtype(np.int16)
                        # dt = dt.newbyteorder('>')
                        channel_data = np.frombuffer(channel_data, dtype=dt)
                    else:
                        raise Exception('unknow data format, bps = %d' % fmt.bps)
                    if supply_missing_samples:
                        if len(channel_data) < 0x1000:
                            channel_data = np.append(channel_data, np.zeros(0x1000 - len(channel_data), dtype=channel_data.dtype))
                        elif len(channel_data) > 0x1000:
                            channel_data = channel_data[:0x1000]
                    res[key] = channel_data
            return res

    def _get_offset(self, coord):
        """ вичислить смещенее блока в файле по координате используя блоки смещений.
        полученное смещенее может относится к блоку с меньшей кординатой и потребуется
        последовательный перебор (не более 16 блоков) для поиска нужного """
        assert coord % 0x1000 == 0
        if self._range[0] <= coord <= self._range[1]:
            if len(self._offsets) > 1:
                step = self._offsets[1][0] - self._offsets[0][0]
                idex = (coord - self._offsets[0][0]) // step
                assert idex < len(self._offsets)
                return self._offsets[idex][1]
            else:
                return _struct_file_hdr.get_size()
        else:
            raise Exception('coord out of range')

    def _scan_range(self):
        """ прочитать первый и последний блоки данных и сохранить их координаты """
        def get_coord():
            data = struct_block_data.read(self._file, False)
            if data.sign != b'DBLC':
                raise WrongSignature('data block header')
            return data.dwCoord
        range = [0, 0]
        for hdr_block in self.enum_blocks(_struct_file_hdr.get_size(), True):
            if hdr_block.type != BlockType.DATA.value:
                continue
            range[0] = get_coord()
            break
        self._file.seek(-struct_block_hdr.get_size(), 2)
        for hdr_block in self.enum_blocks(self._file.tell(), False):
            if hdr_block.type != BlockType.DATA.value:
                continue
            range[1] = get_coord()
            break
        return range

    def _scan_offsets(self):
        """ пройти по всем блокам со смещенийми и посотроить таблицу смещений """
        offsets = []
        file_pos = _struct_file_hdr.get_size()
        while file_pos != 0:
            for hdr_block in self.enum_blocks(file_pos, True):
                if hdr_block.type != BlockType.OFFSETS.value:
                    continue
                block_offsets = _reader_block_offsets.read(self._file, False)
                if block_offsets.sign != b'BOFT':
                    raise WrongSignature('block offsets')
                struct_offsets = struct.Struct('q'*block_offsets.nTableSize)
                block = struct_offsets.unpack(self._file.read(struct_offsets.size))
                for i, o in enumerate(block):
                    if o == 0:
                        break
                    coord = block_offsets.dwBaseCoord + i * block_offsets.nBlockStep * 0x1000
                    offsets.append((coord, o))
                file_pos = block_offsets.oftNextIndexBlock
                break
        return offsets

    def _scan_format(self):
        """ найти блок описаний каналов """
        for hdr_block in self.enum_blocks(_struct_file_hdr.get_size(), True):
            if hdr_block.type != BlockType.DESCRIPTION.value:
                continue
            block_format = _struct_block_format.read(self._file, False)
            if block_format.sign != b'BFMT':
                raise WrongSignature('block format')
            formats = {}
            for c in range(block_format.nCount):
                channel_format = _struct_channel_format.read(self._file, False)
                if channel_format.sign != b'CFMT':
                    raise WrongSignature('block format')
                rail, channel = channel_format.nRail, channel_format.nChannel
                desc = self._file.read(channel_format.nTextLen)[:-1].decode('UTF-8')
                node = ET.fromstring(desc)
                dst = node.attrib.get('dst%d' % channel_format.nRail)
                if dst:
                    rail, channel = map(int, dst.split(','))
                f = obj(rail=rail, cannel=channel, step=channel_format.nStepMM,
                         bps=channel_format.nBytePerSample, desc=desc)
                formats[(rail, channel)] = f
            return formats

    def enum_blocks(self, pos, forward):
        """ функция для последовательного преход по блокам """
        if pos is None:
            pos = _struct_file_hdr.get_size()
        while pos:
            self._file.seek(pos)
            hdr_block = struct_block_hdr.read(self._file, False)
            if hdr_block.sign != b'BHDR':
                raise WrongSignature('block header')
            if hdr_block.type != BlockType.EMPTY.value:
                yield hdr_block
            pos = hdr_block.next if forward else hdr_block.prev


class Writer:
    """ Запись данных """

    class BlockWriter:
        """ запись заголовков блоков и самих блоков """
        def __init__(self, file):
            self._file = file
            self._prev_pos = 0

        def write(self, block_type, block):
            current_pos = self._file.tell()
            next_pos = current_pos + struct_block_hdr.get_size() + len(block)
            hdr = struct_block_hdr.pack(sign=b'BHDR', prev=self._prev_pos, next=next_pos,
                                        type=block_type, reserv=b'')
            self._file.write(hdr)
            self._file.write(block)
            self._prev_pos = current_pos
            tmp = self._file.tell()
            return current_pos

        def finish(self):
            struct_block_hdr.write(self._file, sign=b'BHDR', prev=self._prev_pos, next=0, type=0, reserv=b'')

    class Offsets:
        """ запись смещений """
        BLOCK_STEP = 16
        TABLE_SIZE = 64

        def __init__(self, file):
            self._file = file
            self._block_offset = 0
            self._start_coord = 0
            self._table = [0 for _ in range(self.TABLE_SIZE)]

        def init(self, writer, coord):
            if self._block_offset:
                return
            self._start_coord = coord
            block = self._build_block()
            self._block_offset = writer.write(3, block) + struct_block_hdr.get_size()

        def save_offset(self, coord, offset):
            diff = coord - self._start_coord
            index = int(diff / 0x1000 / self.BLOCK_STEP)
            if len(self._table) <= index:
                raise Exception("only single offset table allowed")
            if not self._table[index]:
                self._table[index] = offset

        def write(self):
            if not self._block_offset:
                return
            block = self._build_block()
            self._file.seek(self._block_offset)
            self._file.write(block)

        def _build_block(self):
            block = BytesIO()
            _reader_block_offsets.write(block, sign=b'BOFT', dwBaseCoord=self._start_coord, oftNextIndexBlock=0,
                                        nBlockStep=self.BLOCK_STEP, nTableSize=self.TABLE_SIZE, reserv=b'')
            b = struct.pack('q'*self.TABLE_SIZE, *self._table)
            block.write(b)
            return block.getvalue()

    def __init__(self, path):
        self._path = path
        self._file = open(self._path, 'wb+')
        _struct_file_hdr.write(self._file, sign=b'FHDR', nVersion=1, flags=0, reserve1=b'')
        self._block_writer = Writer.BlockWriter(self._file)
        self._offsets = Writer.Offsets(self._file)
        self._current_block = {}
        self._current_block_coord = 0

    def write_format(self, format):
        block = BytesIO()
        _struct_block_format.write(block, sign=b'BFMT', nCount=len(format), reserv=b'')
        for f in format:
            _struct_channel_format.write(block, **f)
        self._block_writer.write(2, block.getvalue())

    def write_data(self, coord, rail, channel, data):
        """ записать данные канала """
        assert coord % 0x1000 == 0
        assert coord >= self._current_block_coord
        self._offsets.init(self._block_writer, coord)

        if self._current_block_coord != coord:
            self._save_cur_block()
        self._current_block_coord = coord
        self._current_block[(rail, channel)] = data

    def flash(self):
        """ записать закешированные блоки """
        self._save_cur_block()
        self._block_writer.finish()
        self._offsets.write()
        self._file.close()

    def _save_cur_block(self):
        if not self._current_block:
            return
        buffer_data_block = self._make_data_block(self._current_block_coord, self._current_block)
        self._current_block = {}
        offset = self._block_writer.write(1, buffer_data_block)
        self._offsets.save_offset(self._current_block_coord, offset)


    @staticmethod
    def _make_data_block(coord, channels):
        raw_block = BytesIO()
        for (rail, channel), data in channels.items():
            _struct_channel_data.write(raw_block, sign=b'CDTA', nDataSizeByte=len(data), nRail=rail, nChannel=channel, nReserved=b'')
            raw_block.write(data)
        raw_block = raw_block.getvalue()
        packed_block = zlib.compress(raw_block)

        block = BytesIO()
        struct_block_data.write(block, sign=b'DBLC', dwCoord=coord, nCount=len(channels), flags=1, reserved=b'')
        block.write(struct.pack('LL', len(packed_block), len(raw_block)))
        block.write(packed_block)
        return block.getvalue()

# =========================== TESTS ============================

import unittest

class TestDataReader(unittest.TestCase):

    @unittest.skip
    def test_joiner(self):
        j = _Joiner(104, 108, dtype=np.int16)
        j.append(100, np.arange(100, 110))
        j.append(110, np.arange(110, 120))
        pass

    def test_write(self):
        format = [dict(sign=b'CFMT', nStepMM=1, nRail=r, nChannel=c, nBytePerSample=2, reserved=b'', nTextLen=0)
                  for r in range(1,3) for c in range(8)]
        writer = Writer('1.mmag')
        writer.write_format(format)
        writer.write_data(0x1000, 1, 1, b'abcdefgh')
        writer.write_data(0x2000, 1, 2, b'12345678')
        writer.flash()


    @unittest.skip
    def test_dataa(self):
        data_path = r'D:\ATapeXP\Main\тест\[951]_2017_01_09_14.mmag'
        data_reader = Reader(data_path)
        c = 12 * 1000 * 1000 // 0x1000 * 0x1000
        data = data_reader.get_block(c)
        data1 = data_reader.get_block(c)
        c2 = 11382784
        coords, datas = data_reader.get_data([(1,1)], [c2+100, c2 + 0x1000 + 1000])
        assert len(coords) == len(datas[0])
        print(1)
        pass


if __name__ == '__main__':
    unittest.main()
