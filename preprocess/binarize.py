#coding=utf-8
import cPickle
import argparse


parser = argparse.ArgumentParser(description="""Preprocess the corpus.""")
parser.add_argument("-b", "--binarized", required=True, help="the name of the pickled binarized text file.")
parser.add_argument("-i", "--input", required=True, help="the input file.")
parser.add_argument("-d", "--dic", required=True, help="the dictionary file.")


def sentence2idx(sentence, dic):
    idxes = []
    for w in sentence:
        if dic.has_key(w):
            idx = dic[w]
        else:
            idx = dic['UNK']
        idxes.append(idx)

    return idxes

def lineSplit(line):
    line = line.decode("utf-8")
    chars = []
    for c in line:
        chars.append(c.encode("utf-8"))
    return chars

def main():
    args = parser.parse_args()
    dic = cPickle.load(open(args.dic,'rb'))

    poems = []
    fin = open(args.input, 'rb')
    lines = fin.readlines()
    fin.close()
    max_len = 0
    for line in lines:
        line = line.strip()
        para = line.split("#")
        wstr = para[0]
        senstr = para[1]
        glstr = para[2]

        words = wstr.split(" ")
        sensvec = []
        sens = senstr.split("|")
        for sen in sens:
            sensplit = lineSplit(sen)
            idxes = sentence2idx(sensplit, dic)
            if len(idxes) > max_len:
                max_len = len(idxes)
            sensvec.append(idxes)

        keyvec = []
        for word in words:
            chars = lineSplit(word)
            idxes = sentence2idx(chars, dic)
            keyvec.append(idxes)

        gls = glstr.split("|")
        glvec = []
        for gl in gls:
            glidxes = gl.split(" ")
            glidxes = [int(gid) for gid in glidxes]
            glvec.append(glidxes)
        
        assert len(sensvec) == len(glvec)
        poems.append((keyvec, sensvec, glvec))

    print ("max sentence length: %d" % (max_len))
    print ("dats size: %d" % (len(poems)))
    print "saving dev poems to %s" % (args.binarized)
    output = open(args.binarized, 'wb')
    cPickle.dump(poems, output, -1)
    output.close()


if __name__ == "__main__":
    main()
