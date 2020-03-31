#coding=utf-8
import pickle
import argparse
import random

parser = argparse.ArgumentParser(description="""Preprocess the corpus.""")
parser.add_argument("-i", "--input", required=True, help="the input file")
parser.add_argument("-b", "--binarized", required=True,
        help="the name of the pickled binarized text file.")
parser.add_argument("-d", "--dic", required=True, help="the dictionary file.")


# NOTE: These two parameters should be set  in accord 
#    with those in preprocess.py
MAX_SEN_LEN = 9
MAX_SENS_NUM = 4

def chars2idxes(chars, dic):
    idxes = []
    for c in chars:
        if c in dic:
            idx = dic[c]
        else:
            idx = dic['UNK']
        idxes.append(idx)

    return idxes


def main():
    args = parser.parse_args()
    dic = pickle.load(open(args.dic,'rb'))

    poems = []
    fin = open(args.input, 'r')
    lines = fin.readlines()
    fin.close()


    max_len = 0
    pad_sens_num = 0
    print ("building...")
    N = len(lines)
    for i, line in enumerate(lines):
        if i % 50000 == 0:
            info = float(i) / N * 100
            print ("%.2f%%" % (info))

        line = line.strip()
        para = line.split("#")
        wstr = para[0]
        senstr = para[1]
        glstr = para[2]
        
        sensvec = []
        sens = senstr.split("|")
        for sen in sens:
            chars = [c for c in sen]
            idxes = chars2idxes(chars, dic)
            yan = len(idxes)
            assert yan == 5 or yan == 7
            if yan > max_len:
                max_len = yan
            sensvec.append(idxes)

        keyvec = []
        words = wstr.split(" ")
        for word in words:
            chars = [c for c in word]
            idxes = chars2idxes(chars, dic)
            assert len(idxes) <= 2
            keyvec.append(idxes)

        glvec = []
        gls = glstr.split("|")
        for gl in gls:
            gl_idxes = gl.split(" ")
            gl_idxes = [int(gid) for gid in gl_idxes]
            glvec.append(gl_idxes)
        
        assert len(sensvec) == len(glvec)
        rn = MAX_SENS_NUM-len(sensvec)
        if rn > 0:
            pad_sens_num += 1
        sensvec = sensvec + [[] for _ in range(0, rn)]
        glvec = glvec + [[0]*MAX_SEN_LEN for _ in range(0, rn)]
        assert len(sensvec) == len(glvec) == MAX_SENS_NUM
        poems.append((keyvec, sensvec, glvec))


    random.shuffle(poems)
    print ("max length:%d, pad sens num:%d, final instances:%d" \
        % (max_len, pad_sens_num, len(poems)))
    print ("saving dev poems to %s" % (args.binarized))
    output = open(args.binarized, 'wb')
    pickle.dump(poems, output, -1)
    output.close()


if __name__ == "__main__":
    main()
