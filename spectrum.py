import os
import numpy as np
import ujson
import zlib
import msgpack
from chardet import detect

nums = [str(i) for i in range(10)]
Color_Dict = {'红': 'RED', '绿': 'GREEN', '蓝': 'BLUE'}
YName_Dict = {'G': 'G', 'CD': 'CD', 'ABSORBANCE': 'ABS'}

if msgpack.version[0] >= 1:
    msg_load_kwarg = {'strict_map_key': False}
else:
    msg_load_kwarg = {'encoding': 'utf-8'}


def save_as_json(filename, something):
    """Save something as json into file"""  #将某些内容作为json保存到文件中
    with open(filename, 'wt') as f:
        ujson.dump(something, f)  #读取json最快的模块是ujson，json.dump()将数据以json的数据类型写入文件中


def load_from_json(filename):
    """Load something from json file"""  #从json文件加载一些东西
    if not os.path.exists(filename):  #命令 exists：文件名字符串作为参数，如果文件存在的话，它将返回 True，否则将返回 False。
        raise AssertionError('File not found: %s' % filename)
    with open(filename, 'rt') as f:
        something = ujson.load(f)  #json.load()从json文件中读取数据
    return something


def msg_dumps_and_zip(something):
    return zlib.compress(msgpack.dumps(something, use_bin_type=True))


def msg_unzip_and_loads(something):
    return msgpack.loads(zlib.decompress(something), use_list=False, **msg_load_kwarg)


def load_from_msgzip(filename):
    """Load something from msgzip file"""
    if not os.path.exists(filename):
        raise AssertionError('File not found: %s' % filename)
    with open(filename, 'rb') as f:
        something = f.read()
    return msg_unzip_and_loads(something)


def save_as_msgzip(filename, something):
    """Save something as msgzip into file"""
    with open(filename, 'wb') as f:
        f.write(msg_dumps_and_zip(something))


def read_file(filename):
    """Read .json/msgz file"""
    assert os.path.isfile(filename), 'ERROR: Fail to find file %s\n' % filename
    temp = os.path.splitext(os.path.split(filename)[1])
    ext = temp[1][1:]
    if ext == 'json':
        return load_from_json(filename)
    elif ext == 'msgz':
        return load_from_msgzip(filename)
    else:
        raise AssertionError('Unrecognized file format!')


def save_file(filename, something):
    """Save .json/msgz file"""
    temp = os.path.splitext(os.path.split(filename)[1])
    ext = temp[1][1:]
    if ext == 'json':
        return save_as_json(filename, something)
    elif ext == 'msgz':
        return save_as_msgzip(filename, something)
    else:
        raise AssertionError('Unrecognized file format!')


def read_txt_lines(FileName, SPLIT=False, IncludeComment=False):
    """Read lines from txt file, skipping blank and comment lines. Return a list."""  #从txt文件中读取行，跳过空白行和注释行。返回一个列表。
    with open(FileName, 'rb') as f:
        cur_encoding = detect(f.read(10000))['encoding']
    with open(FileName, 'rt', encoding=cur_encoding) as f:
        lines = f.readlines()
    TreatMethod = str.split if SPLIT else str.strip
    return list(TreatMethod(l) for l in lines if l.strip() and (IncludeComment or l.strip()[0] != '#'))


class SpectrumData(object):
    """Informations of Spectrum Data"""  #光谱数据信息

    def __init__(self, ID=None, NAME='', COLOR='', ANGLE=0, THICK=0, STRAIN=0, GRAY=0, POLAR=''):
        self.ID = ID
        self.Name = NAME
        self.Color = COLOR  # 字符串
        self.Angle = ANGLE  # 浮点数，方便直接作为参数用于机器学习
        self.Thickness = THICK  # 浮点数
        self.Strain = STRAIN  # 浮点数
        self.Grayscale = GRAY  # 浮点数
        self.PolarType = POLAR  # 字符串
        self.X = []
        self.G = []
        self.CD = []
        self.ABS = []

    def __eq__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        return (self.Name == other.Name)

    def __gt__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        if self.Color != other.Color:
            return (self.Color > other.Color)
        elif self.PolarType != other.PolarType:
            return (self.PolarType > other.PolarType)
        elif self.Angle != other.Angle:
            return (self.Angle > other.Angle)
        elif self.Thickness != other.Thickness:
            return (self.Thickness > other.Thickness)
        elif self.Strain != other.Strain:
            return (self.Strain > other.Strain)
        elif self.Grayscale != other.Grayscale:
            return (self.Grayscale > other.Grayscale)

    def __lt__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        if self.Color != other.Color:
            return (self.Color < other.Color)
        elif self.PolarType != other.PolarType:
            return (self.PolarType < other.PolarType)
        elif self.Angle != other.Angle:
            return (self.Angle < other.Angle)
        elif self.Thickness != other.Thickness:
            return (self.Thickness < other.Thickness)
        elif self.Strain != other.Strain:
            return (self.Strain < other.Strain)
        elif self.Grayscale != other.Grayscale:
            return (self.Grayscale < other.Grayscale)

    def __ne__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        return not (self == other)

    def __ge__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        return (self == other) or (self > other)

    def __le__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        return (self == other) or (self < other)

    @classmethod  #直接类名.方法名()来调用，不需要self参数，但第一个参数需要是表示自身类的cls参数
    def get_from_file(cls, FileName):  #cls类似于self，可以用cls.xxxx来调用类的属性，类的方法。可以直接使用类名.方法名()来调用
        """Get a new SpectrumData from file"""  #从文件中获取新的SpectrumData
        assert os.path.isfile(FileName), "Fail to find file %s" % FileName

        # Get information from fullname    从全名获取信息
        FullName = os.path.abspath(FileName)  #
        Info = {}
        NameSecs = FullName.split(os.sep)
        for l in NameSecs:
            if '主吸收' in l:
                cname = 'NULL'
                cnum = '0nm'
                for k in l.split('-'):
                    if k in Color_Dict:
                        cname = Color_Dict[k]
                    if '主吸收' in k:
                        cnum = k[k.find('主吸收') + 3:]
                    if '°' in k:
                        Info['Angle'] = float(k[:k.find('°')])
                Info['Color'] = cname + cnum
            if '厚度' in l:
                Info['Thickness'] = float(l[l.find('厚度') + 2:l.find('um')])
            if '％' in l:
                Info['Strain'] = float(l[l.find('-') + 1:l.find('％')])
            if '灰度' in l:
                Info['Grayscale'] = float(l[l.find('灰度') + 2:l.find('.')])

        # Get information from text     从文本中获取信息
        lines = read_txt_lines(FileName)
        Idx_DataStart = None
        DataNames = {0: 'X'}  ## XYDATA 的名字
        for i, l in enumerate(lines):
            KeyList = l.split()
            if KeyList[0] == 'YUNITS':
                DataNames[1] = YName_Dict[KeyList[1]]
            elif KeyList[0] == 'Y2UNITS':
                DataNames[2] = YName_Dict[KeyList[1]]
            elif KeyList[0] == 'Y3UNITS':
                DataNames[3] = YName_Dict[KeyList[1]]
            elif KeyList[0] == 'XYDATA':
                Idx_DataStart = i + 1
                break
        # 初始化 XYDATA
        for value in DataNames.values():
            Info[value] = []
        # 读取 XYDATA
        for dataline in lines[Idx_DataStart:]:
            if dataline.startswith('[Comments]'):
                break
            for i, x in enumerate(dataline.split()):
                Info[DataNames[i]].append(float(x))
        # 正序排列 XYDATA
        idx_sorted = np.argsort(Info['X'])
        for key in DataNames.values():
            Info[key] = np.array(Info[key])[idx_sorted].tolist()
        return cls.get_from_dict(Info)

    def as_dict(self):
        return self.__dict__  #__dict__可以作用在文件、类或者类的对象上，最终返回的结果为一个字典。

    @classmethod
    def get_from_dict(cls, Dict):
        """Get a new SpectrumData from str line"""
        sd = cls()
        sd.__dict__.update(Dict)
        sd.Name = sd.default_name()
        return sd

    def default_name(self):
        """Default name of a SpectrumData"""
        return '%s-%s-%g-%g-%g-%g' % (self.Color, self.PolarType, self.Angle, self.Thickness, self.Strain, self.Grayscale)

    def __str__(self):
        # idstr = 'None' if self.ID is None else '%d' % self.ID
        idstr = str(self.ID)
        strtmp = 'ID: %s\nName: %s\nColor: %s\nPolarType: %s\n' % (idstr, self.Name, self.Color, self.PolarType)
        strtmp += 'Angle: %g°\nThickness: %g\nStrain: %g%%\nGrayscale: %g\n' % (self.Angle, self.Thickness, self.Strain, self.Grayscale)
        strtmp += 'X: %d values %g %g ... %g\n' % (len(self.X), self.X[0], self.X[1], self.X[-1])
        strtmp += 'G: %d values\nCD: %d values\nABS: %d values\n' % (len(self.G), len(self.CD), len(self.ABS))
        return strtmp


class SpectrumDataBase(object):
    """Database of Spectrum Data"""  #光谱数据信息

    def __init__(self, PATH=''):
        self.Path = PATH
        self.DataName = []  # 光谱数据的名称列表
        self.FullData = []  # 数据库中的全部光谱数据

    def save(self):
        """Save database into files"""
        OutputData = {sd.Name: sd.as_dict() for sd in self.FullData}
        save_file(self.Path, OutputData)

    def load(self):
        """Load database from its path"""
        InputData = read_file(self.Path)
        Name, Data = [], []
        for k, v in InputData.items():
            Name.append(k)
            Data.append(SpectrumData.get_from_dict(v))
        self.DataName = Name
        self.FullData = Data
        return self
 

if __name__ == "__main__":
    #调用函数
    SourcePath = r'G:\202406_pianzhen\2024.6偏振数据'
    DataBasePath = './sdb.msgz'  # Your Database
    sdb = SpectrumDataBase(DataBasePath)
    Num = 0
    for TopDir, DirList, FileList in os.walk(SourcePath):
        for file in FileList:
            if file[-4:] != '.txt':
                continue
            sd = SpectrumData.get_from_file(os.path.join(TopDir, file))
            sd.ID = Num
            sd.PolarType = 'C'  # C表示圆偏光，L表示线偏光
            sd.Name = sd.default_name()  # 更新数据名
            sdb.DataName.append(sd.Name)
            sdb.FullData.append(sd)
            Num += 1
    sdb.save()
    # # 测试数据库存储的数据是否正常
    # test = SpectrumDataBase(DataBasePath)
    # test.load()
    # for i in np.random.randint(0, len(test.DataName), 4):
    #     print('SpectrumData:', test.FullData[i])