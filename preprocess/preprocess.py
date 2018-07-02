#coding=utf-8
import cPickle
import numpy as np
import argparse
import random

from Yun import Yun as YunTool
from GL import GLJudge as GLTool

def parse_args():
    parser = argparse.ArgumentParser(description="The parametrs for preprocessing.")
    parser.add_argument("--inp", required=True, type=str, help="The input file path")
    parser.add_argument("--vratio", required=True, type=float, help="The percentage of validation set.")
    parser.add_argument("--tratio", required=True, type=float, help="The percentage of testing set.")
    parser.add_argument("--vout", required=True, type=str, help="The validation file name.")
    parser.add_argument("--tout", required=True, type=str, help="The testing file name.")
    parser.add_argument("--trout", required=True, type=str, help="The training file name.")
    return parser.parse_args()


class Preporcess(object):
    """
    A Tool for data preprocess.  Please note that this tool
    is only for Chinese quatrains.
    """
    def __init__(self):
        '''
        NOTE: 
        The two tools, YunTool and GLTool, are both designed for
        Chinese quatrains only. For other genres of Chinese poetry,
        we will arrange the source codes of related tools as soon as possible.
        '''
        self.sens_num = 4
        self.yuntool = YunTool()
        self.gltool = GLTool()

        # 0: ping, 1:ze, 2: ping or ze
        self.SENGL = {7: ["2120011", "2021100", "2100110", "2021001"],
                      5: ["20011", "21100", "00110", "21001"]}
        self.pingList = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]

        # Load data sets
        # read words
        self.wdic = {}
        fin = open("data/goodwords.txt", 'r')
        lines = fin.readlines()
        fin.close()

        for line in lines:
            line = line.strip()
            self.wdic[line] = 1

        # read function dic
        self.fdic = {}
        fin = open("data/FunctionWords.txt", 'r')
        lines = fin.readlines()
        fin.close()
        for line in lines:
            self.fdic[line.strip()] = 1

        fin = open("data/fchar.txt", 'r')
        lines = fin.readlines()
        fin.close()
        for line in lines:
            self.fdic[line.strip()] = 1

        # load tf_idf
        '''
        Here we simply use tf-idf to select the keywords.
        One can use any other method, e.g, textrank. 
        '''
        self.tf_idf_dic = {}
        fin = open("data/tfidf.txt", 'r')
        line = fin.readline()
        while line:
            line = line.strip()
            para = line.split(" ")
            word = para[0]
            val = float(para[1])
            # We give multi-character words bigger weights
            self.tf_idf_dic[para[0]] = val * (1+0.1*len(word)/3)
            line = fin.readline()
        fin.close()


    def __get_yun(self, sens):
        '''
        Process yun
        Input: four lines
        Outpus: the tonal pattern of the four lines four
                       If the ryhme can't be determined, an 
                       empty list will be returned
        '''
        yuns = []
        for i in xrange(0, self.sens_num):
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
        for i in xrange(0, self.sens_num):
            gl = self.gltool.gelvJudge(sens[i])
            if gl < 0:
                return []
            gls.append(self.SENGL[yan][gl])

        return gls

    def __buildGLYun(self, gls, yun):
        glstr = ""
        for c in gls:
            if c == '0':
                glstr += '34 '
            elif c == '1':
                glstr += '35 '
            else:
                glstr += '0 '

        glstr = glstr.strip()
        glstr = glstr.split(" ")
        glstr = glstr[0:len(glstr)-1]
        glstr.append(yun)
        glstr = " ".join(glstr)
        glstr = glstr.strip()

        return glstr

    def __isFunc(self, word):
        if word in self.fdic:
            return True

        for i in range(0, len(word), 3):
            c = word[i:i+3]
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
        words = np.array(words)[idxes]
        return list(words[0:keynum])

    def __select_keys(self, words, keynum):
        new_words = []
        # Filt function words
        for w in words:
            if self.__isFunc(w):
                continue
            # We suppose each word 
            # consists of at most 2 characters
            if len(w) >= 9:
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
        words = []
        for sen in sens:
            if len(sen) == 0:
                continue
            ws = sen.split(" ")
            for w in ws:
                words.append(w)

        #print (" ".join(words))
        keys = self.__select_keys(words, 4)
        finalkey = set(keys)
        return list(finalkey)

    def __build_pair(self, poem, keywords, num):
        if num > len(keywords):
            return ""

        words = random.sample(keywords, num)

        wstr = " ".join(words)

        pair = wstr + "#" + poem

        return pair

    def process(self, infile, trfile, vfile, tfile, vratio, tratio):
        fin = open(infile, 'r')
        lines = fin.readlines()
        fin.close()

        
        count = 0
        corpus = []

        print ("processing...")
        for line in lines:
            line = line.strip()
            sens = line.split("|")

            # Build tonal patterns
            sen1 = sens[0].replace(" ", "").decode("utf-8")
            sen2 = sens[1].replace(" ", "").decode("utf-8")
            sen3 = sens[2].replace(" ", "").decode("utf-8")
            sen4 = sens[3].replace(" ", "").decode("utf-8")

            yuns = self.__get_yun([sen1, sen2, sen3, sen4])

            if len(yuns) == 0:
                continue

            len1 = len(sen1)
            len2 = len(sen2)
            len3 = len(sen3)
            len4 = len(sen4)
            assert len1 == len2 == len3 == len4
            yan = len1
            assert yan == 5 or yan == 7

            gls = self.__get_gls(yan, [sen1, sen2, sen3, sen4])
            if len(gls) == 0:
                continue

            glvec = []
            for step in xrange(0, self.sens_num):
                gl = self.__buildGLYun(gls[step], yuns[step])
                glvec.append(gl)

            glstr = "|".join(glvec)

            # Extract keywords
            poem = line.replace(" ", "")
            keywords = self.__get_keywords(sens)
            #print (len(keywords))
            for keynum in xrange(0, 4):
                pair = self.__build_pair(poem, keywords, keynum+1)
                #print (len(pair))
                if len(pair) != 0:
                    corpus.append(pair+ "#" + glstr)

            count += 1

        # Output
        print ("Tonal pattern process: %d/%d, %d failed!" % (count, 
            len(lines), len(lines)-count))
        print ("keywords process: %d/%d, %d failed!" % (len(corpus), count*4,
            count*4-len(corpus)))
        print ("outputing...")
        
        random.shuffle(corpus)
        n = len(corpus)
        n_valid = int(n*vratio)
        n_test = int(n*(tratio))

        valid = corpus[0:n_valid]
        test = corpus[n_valid:n_valid+n_test]
        train = corpus[n_test+n_valid:]

        print ("training size: %d, validation size: %d, testing size: %d" %
         (len(train), len(valid), len(test)))

        outFile(trfile, train)
        outFile(vfile, valid)
        outFile(tfile, test)

        self.outCheckLib(train)

    def outCheckLib(self, data):
        dic = {}
        for line in data:
            para = line.split("#")
            sens = para[1].split("|")
            for sen in sens:
                dic[sen] = 1
        fout = open("DuplicateCheckLib.txt", 'w')
        for k, v in dic.iteritems():
            fout.write(k+"\n")
        fout.close()
 
def outFile(filename, data):
    fout = open(filename, 'w')
    for pair in data:
        fout.write(pair+"\n")
    fout.close()



def main():
    args = parse_args()
    tool = Preporcess()
    infile = args.inp
    vfile = args.vout
    tfile = args.tout
    trfile = args.trout
    vratio = args.vratio
    tratio = args.tratio
    tool.process(infile, trfile, vfile, tfile, vratio, tratio)



if __name__ == "__main__":
    main()
