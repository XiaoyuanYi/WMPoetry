# -*- coding: utf-8 -*-
'''
@author: Jiannan Liang
@ Modified by xiaoyuan Yi, 2019.03
'''
import pickle
import os

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
        if data_path is not None and os.path.exists(data_path):
            print ("data file exists, loading...")
            fyun = open(data_path, "rb")
            self.word_map = pickle.load(fyun)
            self.mulword_map = pickle.load(fyun)
            self.poemyun = pickle.load(fyun)
            fyun.close()
        else:
            # create new model
            self.creat_data()

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

    def creat_data(self):
        print ("creating data...")
        # yun_map, a map to show the times that each char
        #  belong to each yun, [char][yun_idx]
        self.yun_map = {}
        count = 0
        for key in self.yun_dic:
            if len(self.yun_dic[key])>1:
                count += 1
                self.yun_map.update({key: {}})
                for yun in self.yun_dic[key]:
                    self.yun_map[key].update({yun:0})
        print ("%d chars belong to multiple yun categories" % (len(self.yun_map)))

        self.read_total_poemlist()
        # to calculate the most likely yun id for each char
        for word in self.yun_map:
            max_times = -1
            max_yun = ""
            for yun in self.yun_map[word]:
                if self.yun_map[word][yun] > max_times:
                    max_times = self.yun_map[word][yun]
                    max_yun = yun
            self.word_map.update({
                word:[max_yun]
                })

        count1 = 0
        count2 = 0
        delitem = []
        # remove the n-grams that belong to multiple yun categories
        for gram in self.mulword_map:
            count1 += 1
            if len(self.mulword_map[gram]) > 1:
                count2 +=1
                delitem.append(gram)        
        for item in delitem:
            self.mulword_map.pop(item)
        
        '''
        for tw in self.mulword_map:
            if len(self.mulword_map[tw]) > 1:
                print tw
        '''
        print ("mul-word-map size:%d, removed size:%d, final size:%d" % \
            (count1, count2, len(self.mulword_map)))

        print ("saveing data files")
        fyun = open("data/yun.pkl", "wb")
        pickle.dump(self.word_map, fyun)
        pickle.dump(self.mulword_map, fyun)
        pickle.dump(self.poemyun, fyun)
        fyun.close()


    def updateyun(self, sen, yun):
        def update_mulword(mulword):
            if mulword in self.mulword_map:
                if not yun in self.mulword_map[mulword]:
                    self.mulword_map[mulword].append(yun)
            else:
                self.mulword_map.update({
                    mulword:[yun]})
            
        word = sen[-1]
        yun = yun[0]
        if word in self.yun_map:
            self.yun_map[word][yun] +=1
            twoword = sen[-2]+sen[-1]
            update_mulword(twoword)
            threeword = sen[-3]+sen[-2]+sen[-1]
            update_mulword(threeword)
            
    def read_total_poemlist(self):
        print ("reading total poemlist")
        f = open("data/totaljiantipoems_change.txt",'r')
        lines = f.readlines()
        f.close()
        
        poemyun = {}
        sen_list = []
        
        for line in lines:
            line = line.strip().split(" ")
            if line[0] == "Title":
                L = len(sen_list)
                yun_list = []
                for i in range(L):
                    yun_list.append(self.getYun(sen_list[i])) 
                if L >= 1:
                    # get the yun id of the second line
                    # filter untial a yun idx occurs in all rhymed lines
                    tmp = yun_list[1]
                    for i in range(L//2):
                        tmp = [val for val in tmp if val in yun_list[i*2+1]]
                else:
                    tmp = []

                if len(tmp) > 1:
                    tmp = [val for val in tmp if val in yun_list[0]]
                if len(tmp) == 1 and tmp[0] != "-1": # []:50366  ["-1"]: 7104  25967
                    for i in range(L//2):
                        yun_list[i*2+1] = tmp
                        self.updateyun(sen_list[i*2+1], tmp)
                    tmp = [val for val in tmp if val in yun_list[0]]
                    if len(tmp)>0:
                        yun_list[0] = tmp
                        self.updateyun(sen_list[0], tmp)

                for i in range(L):
                    poemyun.update({
                        sen_list[i]:yun_list[i]})
                sen_list = []
            else:
                sen_list.append(line[0])

        self.poemyun = poemyun


if __name__ == "__main__":
    yun = Yun(yun_list_path="data/cilinList.txt")

