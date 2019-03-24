#coding=utf-8
import numpy as np
import argparse
import random

from Yun import Yun as YunTool
from GL import GLJudge as GLTool

def parse_args():
    parser = argparse.ArgumentParser(description="The parametrs for preprocessing.")
    parser.add_argument("--inp", required=True, type=str, help="The input file path")
    parser.add_argument("--out", required=True, type=str, help="The output file path")
    parser.add_argument("--cl", type=int, help="If return a checklib, 0: false, 1:true")
    return parser.parse_args()


def readFile(file_path):
    fin = open(file_path, 'r')
    lines = fin.readlines()
    fin.close()
    return lines

class Preporcess(object):
    """
    A Tool for data preprocess.  Please note that this tool
        is only for Chinese quatrains.
    """
    def __init__(self):
        '''
        NOTE: 
        The two tools, YunTool and GLTool, are both designed  only for
            Chinese quatrains. For other genres of Chinese poetry, we 
            will consider releasing the source codes of related tools
            in the future.
        '''
        self.sens_num = 4  # sens_num must be 4
        self.key_num = 4 # It's better to set a key num <= 4
        self.yuntool = YunTool(yun_list_path="data/cilinList.txt",
            data_path="data/yun.pkl")
        self.gltool = GLTool()

        # 0: ping, 1:ze, 2: ping or ze
        self.SENGL = {7: ["2120011", "2021100", "2100110", "2021001"],
                      5: ["20011", "21100", "00110", "21001"]}

        # NOTE: We use cilinzhengyun instead of pingshuiyun
        # 0: either ping or ze; 1~33 rhyme categorizes of cilinzhengyun,
        #  34: ping, 35: ze
        self.pingList = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]

        #--------------------------------
        # Load data sets
        print ("Reading words...")
        self.wdic = {}
        lines = readFile("data/goodwords.txt")
        for line in lines:
            line = line.strip()
            self.wdic[line] = 1
        print ("%s good words." % (len(self.wdic)))

        # read function dic
        self.fdic = {}
        lines = readFile("data/FunctionWords.txt")
        for line in lines:
            self.fdic[line.strip()] = 1

        lines = readFile("data/fchar.txt")
        for line in lines:
            self.fdic[line.strip()] = 1
        print ("%s function words." % (len(self.fdic)))

        # load tf_idf
        print ("Loading TF-IDF dict...")
        '''
        Here we simply use tf-idf to select the keywords.
            One can also use any other method, e.g, textrank. 
        '''
        self.tf_idf_dic = {}
        fin = open("data/tfidf.txt", 'r')
        line = fin.readline()
        while line:
            line = line.strip()
            para = line.split(" ")
            word = para[0]
            val = float(para[1])
            # We give multi-character words larger weights
            self.tf_idf_dic[para[0]] = val * (1+0.1*len(word))
            line = fin.readline()
        fin.close()
        print ("TF-IDF dic size: %d" % (len(self.tf_idf_dic)))

    def __get_yun(self, sens):
        '''
        Extract the rhymes of each lines.
        Input: four lines
        Outpus: the rhymes of the four lines.
                       If the ryhme can't be determined, an 
                       empty list will be returned
        '''
        yuns = []
        for i in range(0, self.sens_num):
            yun = int(self.yuntool.getYun(sens[i])[0])
            yuns.append(yun)


        if yuns[1] == -1 and yuns[3] != -1:
            yuns[1] = yuns[3]
        elif yuns[1] != -1 and yuns[3] == -1:
            yuns[3] = yuns[1]
        elif yuns[1] == -1 and yuns[3] == -1:
            return []

        if yuns[0] not in self.pingList:
            yuns[0] = 35
        yuns[2] = 35

        yunstr = [str(y) for y in yuns]
        return yunstr
    
    def __get_gls(self, yan, sens):
        # process GL
        gls = []
        for i in range(0, self.sens_num):
            gl = self.gltool.gelvJudge(sens[i])
            if gl < 0:
                return []
            gls.append(self.SENGL[yan][gl])

        return gls

    def __buildGLYun(self, gls, yun):
        glvec = []
        for c in gls:
            if c == '0':
                glvec.append('34')
            elif c == '1':
                glvec.append('35')
            else:
                glvec.append('0')

        glvec = glvec[0:-1] + [yun]
        glstr = " ".join(glvec).strip()
        return glstr

    def __isFunc(self, word):
        if word in self.fdic:
            return True
        for c in word:
            if c in self.fdic:
                return True
        return False

    def __select_by_tfidf(self, words, keynum):
        assert keynum <= len(words)
        valvec = []
        for word in words:
            if word in self.tf_idf_dic:
                valvec.append(self.tf_idf_dic[word])
            else:
                valvec.append(0.01)

        idxes = list(np.argsort(valvec))
        idxes.reverse()
        finwords = [words[idx] for idx in idxes[0:keynum]]
        return finwords

    def __select_keys(self, words, keynum):
        new_words = []
        # Filter function words
        for w in words:
            if self.__isFunc(w):
                continue
            # We suppose each word consists of at most 2 characters
            if len(w) >= 3:
                continue
            new_words.append(w)

        if len(new_words) == 0:
            return []
    
        keywords = []
        leftwords = []
        for word in new_words:
            if word in self.wdic:
                keywords.append(word)
            else:
                leftwords.append(word)

        if len(keywords) == keynum:
            return keywords

        if len(keywords) > keynum:
            return self.__select_by_tfidf(keywords, keynum)

        leftnum = keynum - len(keywords)
        if len(leftwords) >= leftnum:
            return keywords + self.__select_by_tfidf(leftwords, leftnum)

        return keywords

    def __get_keywords(self, sens):
        words = " ".join(sens).split(" ")
        filter_words = []
        for w in words:
            w = w.strip()
            if len(w) == 0:
                continue
            filter_words.append(w)

        keys = self.__select_keys(words, self.key_num)
        finalkeys = list(set(keys))
        return finalkeys

    def __build_pair(self, poem, keywords, num):
        if num > len(keywords):
            return ""

        words = random.sample(keywords, num)

        wstr = " ".join(words)

        pair = wstr + "#" + "|".join(poem)

        return pair

    def __rebuildWindow(self, lines):
        '''
        If the corpus contains other genres of poetry, 
            such as Verses which consists of 8 or 12 lines,
            one needs to split it into different 4-line parts.
        '''
        print ("rebuild windows...")
        new_data = []
        for line in lines:
            line = line.strip()
            sens = line.split("|")
            for i in range(0, len(sens), self.sens_num):
                part = sens[i:i+self.sens_num]
                if len(part) != self.sens_num:
                    continue
                new_data.append(part)

        print ("Rebuilding done. Original num: %d. Rebuilt num: %d." % (len(lines), len(new_data)))
        return new_data

    def process(self, infile, outfile, cl_flag):
        lines = readFile(infile)

        data = self.__rebuildWindow(lines)

        count = 0
        corpus = []

        print ("processing...")
        for i, part in enumerate(data):
            if i % 10000 == 0:
                print ("%d/%d, %.2f%%" % (i, len(data), float(i)/len(data)*100))
            s1 = part[0].replace(" ", "")
            s2 = part[1].replace(" ", "")
            s3 = part[2].replace(" ", "")
            s4 = part[3].replace(" ", "")
            if not (len(s1) == len(s2) == len(s3) == len(s4)):
                continue
            # NOTE: This tool is only for Chinese quatrains
            yan = len(s1)
            if yan != 5 and yan != 7:
                continue
            sens = [s1, s2, s3, s4]

            # Build tonal patterns
            # Extract rhymes for each line
            yuns = self.__get_yun(sens)

            if len(yuns) == 0:
                continue

            gls = self.__get_gls(yan, sens)
            if len(gls) == 0:
                continue

            glvec = []
            for step in range(0, self.sens_num):
                gl = self.__buildGLYun(gls[step], yuns[step])
                glvec.append(gl)

            glstr = "|".join(glvec)

            # Extract keywords
            keywords = self.__get_keywords(part)

            for keynum in range(0, self.key_num):
                pair = self.__build_pair(sens, keywords, keynum+1)
                if len(pair) != 0:
                    corpus.append(pair+ "#" + glstr)
                    
            count += 1

        # Output
        print ("Tonal pattern process: %d/%d, %d failed!" % (count, 
            len(data), len(data)-count))
        print ("keywords process: %d/%d, %d failed!" % (len(corpus), count*self.key_num,
            count*self.key_num-len(corpus)))
        
        print ("shuffling data...")
        random.shuffle(corpus)

        print ("final size: %d, output to %s " % (len(corpus), outfile))
        outFile(outfile, corpus)

        if cl_flag == 1:
            print ("outputing checklib...")
            self.outCheckLib(corpus)

    def outCheckLib(self, data):
        dic = {}
        for line in data:
            para = line.split("#")
            sens = para[1].split("|")
            for sen in sens:
                dic[sen] = 1
        fout = open("DuplicateCheckLib.txt", 'w')
        for k, v in dic.items():
            fout.write(k+"\n")
        fout.close()
 
def outFile(filename, data):
    fout = open(filename, 'w')
    for pair in data:
        fout.write(pair+"\n")
    fout.close()


def main():
    tool = Preporcess()

    args = parse_args()
    infile = args.inp
    outfile = args.out
    cl_flag = args.cl

    tool.process(infile, outfile, cl_flag)


if __name__ == "__main__":
    main()
