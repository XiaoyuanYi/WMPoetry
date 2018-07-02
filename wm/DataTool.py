import numpy as np
import os
import threading
import copy

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
        self.__loadGL("data/other/pingsheng.txt",
            "data/other/zesheng.txt")
        self.__loadYun("data/other/cilinList.txt")
        self.__loadPatterns("data/other/GenrePatterns.txt")

    def __loadPatterns(self, path):
        fin = open(self.__root_dir+path, 'r')
        lines = fin.readlines()
        fin.close()
        self.__patterns = []
        '''
        For each line, 
        pattern id, pattern name, ping yun (36) or ze yun (37),
        # of lines in each paragraph, pattern
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

    def __loadPoemLib(self, path):
        print ("Loading poemlib...")
        self.__poemLib = {}
        fin = open(self.__root_dir+path)
        lines = fin.readlines()
        fin.close()
        for line in lines:
            line = line.strip()
            self.__poemLib[line] = 1
        print ("Done, # of lines: %d" % (len(self.__poemLib)))

    def __loadGL(self, p_path, z_path):
        print("Loading pz dic...")
        self.__GLDic = {}
        self.__GLDic[34] = []
        self.__GLDic[35] = []
        # get ze-toned char list
        fin = open(self.__root_dir+z_path, 'r')
        ze = fin.read()
        fin.close()
        ze = ze.decode("utf-8")
        for c in ze:
            c = c.strip()
            if len(c) == 0:
                continue
            self.__GLDic[35].append(c.encode("utf-8"))
                 
        # get ping-toned char list
        fin = open(self.__root_dir+p_path, 'r')
        ping = fin.read()
        fin.close()
        ping = ping.decode("utf-8")
        for c in ping:
            self.__GLDic[34].append(c.encode("utf-8"))

    def __loadYun(self, path):
        print("Loading yun dic...")
        fin = open(self.__root_dir+path, 'r')
        lines = fin.readlines()
        fin.close()
        self.__yundic = {}
        for line in lines:
            line = line.strip()
            para = line.split(" ")

            key = para[0]
            yun = int(para[1])

            if self.__yundic.has_key(key):
                self.__yundic[key].append(yun)
            else:
                self.__yundic[key] = [yun]

            if self.__GLDic.has_key(yun):
                self.__GLDic[yun].append(key)
            else:
                self.__GLDic[yun] = [key]

    # ------------------------
    # public functions
    def buildPHDicForIdx(self, vocab):
        idx_GL = {}
        for yun, vec in self.__GLDic.iteritems():
            idxes = []
            for c in vec:
                if c in vocab:
                    idxes.append(vocab[c])

            idx_GL[yun] = idxes

        return idx_GL

    def checkIfInLib(self, sen):
        sen = sen.strip()
        if sen in self.__poemLib:
            return True
        else:
            return False

    def getPattern(self, idx):
        return copy.deepcopy(self.__patterns[idx-1])

data_tool = DataTool()