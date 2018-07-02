#coding=utf-8
import cPickle
import argparse

parser = argparse.ArgumentParser(description="""Build dictionary.""")
parser.add_argument("-i", "--input", required=True, help="the input file")

def lineSplit(line):
    line = line.decode("utf-8")
    chars = []
    for c in line:
        chars.append(c.encode("utf-8"))
    return chars

def create_dict():
    print ("creatie dictionary and inverting dictionary...")
    args = parser.parse_args()
    fin = open(args.input, 'r')
    lines = fin.readlines()
    fin.close()

    dic = {}
    idic = {}
    idx = 0

    for line in lines:
        line = line.strip()
        para = line.split("#")
        sens = para[1].split("|")
        for sen in sens:
            words = lineSplit(sen)
            for word in words:
                if word not in dic:
                    dic[word] = idx
                    idic[idx] = word
                    idx += 1


    # add EOS
    dic['</S>'] = idx
    idic[idx] = '</S>'
    idx += 1

    # add UNK
    dic['UNK'] = idx
    idic[idx] = 'UNK'
    idx += 1

    # add GO
    dic['GO'] = idx
    idic[idx] = 'GO'
    idx += 1

    # add PAD
    dic['PAD'] = idx
    idic[idx] = 'PAD'
    idx += 1

    print "Total char num: %s" % (len(dic))

    print "saving dictionary to %s" % ("vocab.pkl")
    output = open("vocab.pkl", 'wb')
    cPickle.dump(dic, output, -1)
    output.close()

    print "saving inverting dictionary to %s" % ("i" + "vocab.pkl")
    output = open("i" + "vocab.pkl", 'wb')
    cPickle.dump(idic, output, -1)
    output.close()

def main():

    create_dict()

if __name__ == "__main__":
    main()