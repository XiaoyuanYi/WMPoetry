# -*- coding: utf-8 -*-
'''
@author: Jiannan Liang
'''
class Yun():
    
    def __init__(self):
        print "loading YunList.txt"
        self.yun_dic = {}
        #self.count = 0
        f = open("data/cilinList.txt",'r')
        lines = f.readlines()
        #dk_count = 0
        for line in lines:
            line = line.strip().decode("utf-8").split(' ')
            for i in range(len(line)):
                line[i] = line[i].encode("utf-8")
            if not self.yun_dic.has_key(line[0]):
                self.yun_dic.update({
                    line[0]:[line[1]]
                    })
            else:
                if not line[1] in self.yun_dic[line[0]]:
                    self.yun_dic[line[0]].append(line[1])
                    #dk_count += 1
                #print "Yun has double key %s"%(line[0])
        #print dk_count
        self.poemyun = {}
        self.mulword_map = {}
        self.word_map = {}
        self.mulyun()

    def getYun(self, sen):
        if self.poemyun.has_key(sen):
            return self.poemyun[sen]
        last_word = sen[len(sen)-1].encode('utf-8')
        if self.word_map.has_key(last_word):
            twoword = sen[-2]+sen[-1]
            twoword = twoword.encode("utf-8")
            print twoword
            if self.mulword_map.has_key(twoword):
                return self.mulword_map[twoword]
            threeword = sen[-3]+sen[-2]+sen[-1]
            threeword = threeword.encode("utf-8")
            print threeword
            if self.mulword_map.has_key(threeword):
                return self.mulword_map[threeword]
            print last_word
            return self.word_map[last_word]
        elif self.yun_dic.has_key(last_word):
            return self.yun_dic[last_word]
        else:
            #self.count += 1
            return ['-1']

    def mulyun(self):
        import cPickle as pickle
        import os
        if os.path.exists("data/yun.pkl"):
            fyun = open("data/yun.pkl", "r")
            self.word_map = pickle.load(fyun)
            self.mulword_map = pickle.load(fyun)
            self.poemyun = pickle.load(fyun)
            fyun.close()
            return
        self.yun_map = {}
        count = 0
        for key in self.yun_dic:
            if len(self.yun_dic[key])>1:
                count += 1
                self.yun_map.update({
                    key: {}
                    })
                for yun in self.yun_dic[key]:
                    self.yun_map[key].update({
                        yun:0
                        })
        print count
        #print self.yun_map
        #fout = open("result/yuncount.txt", "w")
        self.totalpoemlist()
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
        #for word in self.yun_map:
            #print>>fout, word
            #print>>fout, self.yun_map[word]
            #print>>fout, self.word_map[word]
        count1 = 0
        count2 = 0
        delitem = []
        for tw in self.mulword_map:
            count1 += 1
            if len(self.mulword_map[tw]) > 1:
                count2 +=1
                #print >> fout, tw
                #print >> fout, self.mulword_map[tw]
                delitem.append(tw)
        for item in delitem:
            self.mulword_map.pop(item)
        for tw in self.mulword_map:
            #print >>fout, tw,self.mulword_map[tw]
            if len(self.mulword_map[tw]) > 1:
                print tw
        #fout.close()
        print count1,count2,len(self.mulword_map)
        fyun = open("data/pkl/yun.pkl", "wb")
        pickle.dump(self.word_map, fyun)
        pickle.dump(self.mulword_map, fyun)
        pickle.dump(self.poemyun, fyun)
        fyun.close()


    def updateyun(self, sen, yun):
        def update_mulword(mulword):
            mulword = mulword.encode("utf-8")
            if self.mulword_map.has_key(mulword):
                if not yun in self.mulword_map[mulword]:
                    self.mulword_map[mulword].append(yun)
            else:
                self.mulword_map.update({
                    mulword:[yun]
                    })
            
        word = sen[-1]
        word = word.encode("utf-8")
        yun = yun[0]
        if self.yun_map.has_key(word):
            self.yun_map[word][yun] +=1
            twoword = sen[-2]+sen[-1]
            update_mulword(twoword)
            threeword = sen[-3]+sen[-2]+sen[-1]
            update_mulword(threeword)

if __name__ == "__main__":
    yun = Yun()

