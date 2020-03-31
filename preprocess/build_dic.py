#coding=utf-8
import pickle
import argparse

parser = argparse.ArgumentParser(description="""Build dictionary.""")
parser.add_argument("-i", "--input", required=True, help="the input file")
parser.add_argument("-m", "--minnum", required=True, help="the min count")

def create_dict():
    print ("creatie dictionary and inverting dictionary...")
    args = parser.parse_args()
    fin = open(args.input, 'r')
    lines = fin.readlines()
    fin.close()

    # -------------------------
    print ("input lines: %d" % (len(lines)))
    count_dic = {}
    for line in lines:
        line = line.strip()
        line = line.replace(" ", "")
        for c in line:
            if c in count_dic:
                count_dic[c] += 1
            else:
                count_dic[c] = 1

    vec = sorted(count_dic.items(), key=lambda d:d[1], reverse=True)
    print ("original char num:%d" % (len(vec)))

    # add special symbols
    # --------------------------------------
    dic = {}
    idic = {}
    dic['PAD'] = 0
    idic[0] = 'PAD'

    dic['UNK'] = 1
    idic[1] = 'UNK'

    dic['<E>'] = 2
    idic[2] = '<E>'

    dic['<B>'] = 3
    idic[3] = '<B>'

    # separator of different paragraph
    dic['<M>'] = 4
    idic[4] = '<M>'
    # --------------------------------------

    idx = 5
    min_freq = int(args.minnum)
    print ("min freq:%d" % (min_freq))

    for c, v in vec:
        if v < min_freq:
            continue
        if not c in dic:
            dic[c] = idx
            idic[idx] = c
            idx += 1

    print ("total char num: %s" % (len(dic)))

    dic_file = "vocab.pickle"
    idic_file = "ivocab.pickle"

    print ("saving dictionary to %s" % (dic_file))
    output = open(dic_file, 'wb')
    pickle.dump(dic, output, -1)
    output.close()

    print ("saving inverting dictionary to %s" % (idic_file))
    output = open(idic_file, 'wb')
    pickle.dump(idic, output, -1)
    output.close()


def main():

    create_dict()

if __name__ == "__main__":
    main()