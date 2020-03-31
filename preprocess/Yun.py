# -*- coding: utf-8 -*-
'''
@author: Jiannan Liang
@ Modified by xiaoyuan Yi, 2019.03
'''
import pickle

class Yun():
    
    def __init__(self, yun_list_path, data_path=None):
        print ("loading YunTool...")
        self.yun_dic = {}
        fin = open(yun_list_path, 'r')
        lines = fin.readlines()
        fin.close()
        
        dk_count = 0
        for line in lines:
            (yun_char, yun_id) = line.strip().split(' ')
            if not yun_char in self.yun_dic:
                self.yun_dic.update({
                    yun_char:[yun_id]})
            else:
                if not yun_id in self.yun_dic[yun_char]:
                    self.yun_dic[yun_char].append(yun_id)
                    dk_count += 1
                #print "Yun has double key %s"%(line[0])
        
        print ("%d chars belong to different yun categories" % (dk_count))
        self.poemyun = {} # the yun id list of each line
        self.mulword_map = {} # to show the yun id list of each bigram or trigram 
        self.word_map = {} # the most likely yun id for each char
        
        # load the calculated data, if there is any
        print ("data file exists, loading...")
        fyun = open(data_path, "rb")
        self.word_map = pickle.load(fyun)
        self.mulword_map = pickle.load(fyun)
        self.poemyun = pickle.load(fyun)
        fyun.close()

    def getYun(self, sen):
        if sen in self.poemyun:
            return self.poemyun[sen]

        last_word = sen[-1]
        if last_word in self.word_map:
            twoword = sen[-2]+sen[-1]
            #print ("search tail bigram %s" % (twoword))
            if twoword in self.mulword_map:
                return self.mulword_map[twoword]
            threeword = sen[-3]+sen[-2]+sen[-1]
            #print ("search tail trigram %s" % (threeword))
            if threeword in self.mulword_map:
                return self.mulword_map[threeword]
            #print ("matching unigram %s" % (last_word))
            return self.word_map[last_word]
        elif last_word in self.yun_dic:
            #print ("search yun dic...")
            return self.yun_dic[last_word]
        else:
            #self.count += 1
            return ['-1']


if __name__ == "__main__":
    yun = Yun(yun_list_path="data/cilinList.txt")

