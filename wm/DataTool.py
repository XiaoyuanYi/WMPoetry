import os
import threading
import copy

def readFile(file_path):
    fin = open(file_path, 'r')
    lines = fin.readlines()
    fin.close()
    return lines

class DataTool(object):
    '''
    The tool class for data operation
    '''
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(DataTool, "_instance"):
            with DataTool._instance_lock:
                if not hasattr(DataTool, "_instance"):
                    DataTool._instance = object.__new__(cls)  
        return DataTool._instance

    def __init__(self):
        self.__root_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
        self.__loadPoemLib("data/other/DuplicateCheckLib.txt")
        self.__loadGL("data/other/pingsheng.txt", "data/other/zesheng.txt")
        self.__loadYun("data/other/cilinList.txt")
        self.__loadPatterns("data/other/GenrePatterns.txt")

    def __loadPatterns(self, path):
        print ("Loading genre patterns...")
        lines = readFile(self.__root_dir+path)

        self.__patterns = []
        '''
        For each line, 
        pattern id, pattern name, ping yun (36) or ze yun (37),
        the number of lines in each paragraph, pattern
        '''
        for line in lines:
            line = line.strip()
            para = line.split("#")
            pas = para[4].split("|")
            newpas = []
            for pa in pas:
                pa = pa.split(" ")
                newpas.append([int(p) for p in pa])
            sen_lens_str = para[3].split(" ")
            sen_lens = [int(l) for l in sen_lens_str]
            # tune name; patterns; 36 ping yun, 37 ze yun; # of lines in each paragraph
            self.__patterns.append((para[1], newpas, int(para[2]), sen_lens))
        print ("Done, %d patterns in total." % (len(self.__patterns)))

    def __loadPoemLib(self, path):
        print ("Loading poemlib...")
        self.__poemLib = {}
        lines = readFile(self.__root_dir+path)
        for line in lines:
            line = line.strip()
            self.__poemLib[line] = 1
        print ("Done, %d lines in total" % (len(self.__poemLib)))

    def __loadGL(self, p_path, z_path):
        '''
        TODO: hold the polyphone problem
        NOTE: We use cilinzhengyun instead of pingshuiyun
            0: either ping or ze; 1~33 rhyme categorizes of cilinzhengyun,
            34: ping, 35: ze
        '''
        print("Loading pz dic...")
        self.__GLDic = {}
        self.__GLDic[34] = []
        self.__GLDic[35] = []
        # get ze-toned char list
        ze_lines = readFile(self.__root_dir+z_path)
        for line in ze_lines:
            line = line.strip()
            for c in line:
                if len(c) == 0:
                    continue
                self.__GLDic[35].append(c)
                 
        # get ping-toned char list
        ping_lines = readFile(self.__root_dir+p_path)
        for line in ping_lines:
            line = line.strip()
            for c in line:
                if len(c) == 0:
                    continue
                self.__GLDic[34].append(c)

    def __loadYun(self, path):
        print("Loading yun dic...")
        lines = readFile(self.__root_dir+path)
        self.__yundic = {}
        for line in lines:
            line = line.strip()
            para = line.split(" ")

            key = para[0]
            yun = int(para[1])

            if key in self.__yundic:
                self.__yundic[key].append(yun)
            else:
                self.__yundic[key] = [yun]

            if yun in self.__GLDic:
                self.__GLDic[yun].append(key)
            else:
                self.__GLDic[yun] = [key]

    # ------------------------
    # public functions
    def buildPHDicForIdx(self, vocab):
        idx_GL = {}
        for yun, vec in self.__GLDic.items():
            idxes = []
            for c in vec:
                if c in vocab:
                    idxes.append(vocab[c])

            idx_GL[yun] = idxes

        return idx_GL

    def checkIfInLib(self, sen):
        sen = sen.strip()
        sen = sen.replace(" ", "")
        if sen in self.__poemLib:
            return True
        else:
            return False

    def getPattern(self, idx):
        return copy.deepcopy(self.__patterns[idx])

data_tool = DataTool()