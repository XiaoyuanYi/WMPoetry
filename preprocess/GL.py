#coding=utf-8
from __future__ import division
'''
Created on 2015.10.31
@author: Ruoyu Li
'''

class GLJudge(object):
    '''
    Get the tonal pattern of a poem line
    '''
    def __init__(self):
        '''
        Constructor
        '''
        print "loading pingsheng.txt"
        f = open("data/pingsheng.txt",'r')
        self.__ping = f.read()
        #print type(self.__ping)
        f.close()
        self.__ping = self.__ping.decode("utf-8")
        f = open("data/zesheng.txt",'r')
        self.__ze = f.read()
        f.close()
        #print type(self.__ze)
        self.__ze = self.__ze.decode("utf-8")
        
        
        
    def gelvJudge(self, sentence):
        '''
        Get the tonal pattern of a poem line
        Input: a sentence with 5 or 7 character.
                   The input sequence must be unicode
        Output: the tonal pattern, 0~3
        '''
        # print "#" + sentence + "#"
        if(len(sentence) == 5):
            #1
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze):
                return 0
            #2
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ze):
                return 0
            #3
            if(sentence[0] in self.__ze and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze):
                return 0
            #4
            if(sentence[0] in self.__ze and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 0
            #5
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 0
            #6
            if(sentence[0] in self.__ze and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping):
                return 1
            #7
            if(sentence[0] in self.__ping and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping):
                return 1
            #8
            if(sentence[0] in self.__ping and sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #9
            if(sentence[0] in self.__ping and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #10
            if(sentence[0] in self.__ze and sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #11
            if(sentence[0] in self.__ze and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #12
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ping):
                return 2
            #13
            if(sentence[0] in self.__ze and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ping):
                return 2
            #14
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ping):
                return 2


        elif (len(sentence) == 7):
            #1
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            #2
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            #3
            if(sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            #4
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 0
            #5
            if(sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 0
            #6
            if(sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ping):
                return 1
            #7
            if(sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ping):
                return 1
            #8
            if(sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ping and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #9
            if(sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #10
            if(sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ping and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #11
            if(sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #12
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
            #13
            if(sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
            #14
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
        else:
            return -2
        return -1
        
