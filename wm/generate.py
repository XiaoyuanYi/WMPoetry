from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from generate_base import Generator
import argparse
import copy
from DataTool import data_tool

def parse_args():
    parser = argparse.ArgumentParser(description="The parametrs for the generator.")
    parser.add_argument("--inp", type=str, help="The input file path, for file generation.")
    parser.add_argument("--out", type=str, help="The output file path, for file generation.")
    parser.add_argument("--type",  required=True, type=str, choices=['one', 'file'], 
        help='The mode. one: just generate one poem; file: generate by the input file.')
    parser.add_argument("--model", type=str, help="The checkpoint path. If none, just use the newest checkpoint.")
    parser.add_argument("--bsize",  required=True, type=int, help="The beam size.")
    return parser.parse_args()


class GeneratorUI(object):

    def __init__(self, beam_size, modelfile):
        self.generator = Generator(beam_size, modelfile)
        self.dtool = data_tool

    def __setYun2Pattern(self, ori_pattern, yun):
        pattern = copy.deepcopy(ori_pattern)
        for i in xrange(0, len(pattern)):
            if pattern[i][-1] == 36 or pattern[i][-1] == 37:
                pattern[i][-1] = yun
        return pattern

    def generate_one(self):
        while True:
            keys = raw_input("please input keywords (with whitespace split) > ")
            pattern_id = input("please select genre pattern > ")
            yun = input("please input yun type> ")
            ori_pattern = self.dtool.getPattern(pattern_id)
            pattern = ori_pattern[1]
            name = ori_pattern[0]
            pattern = self.__setYun2Pattern(pattern, yun)
            print ("select pattern: %s" % (name))
            ans, info = self.generator.generate_one(keys, pattern)
            if len(ans) == 0:
                print("generation failed!")
                print(info)
                continue

            print ("\n".join(ans))

    def modiPattern(self, patternStr):
        #print (patternStr)
        patterns = []
        for pstr in patternStr:
            pas = pstr.split(" ")
            pas = [int(pa) for pa in pas]
            patterns.append(pas)

        #print (patterns)
        return patterns

    def generate_file(self, infile, outfile):

        fin = open(infile, 'r')
        lines = fin.readlines()
        fin.close()

        fout = open(outfile, 'w')
        for i, line in enumerate(lines):
            line = line.strip()
            para = line.split("#")
            wstr = para[0].strip()
            print ("%d  keys: %s" % (i, wstr))
            pattern_str = para[2].split("|")
            pattern = self.modiPattern(pattern_str)

            sens, info = self.generator.generate_one(wstr, pattern)
            if len(sens) == 0:
                fout.write(info + "\n")
            else:
                fout.write("|".join(sens) + "\n")
            fout.flush()

        fout.close()

def main():
    args = parse_args()
    if args.model:
        modefile = args.model
    else:
        modefile=None
    ui = GeneratorUI(int(args.bsize), modefile)
    if args.type == 'one':
        ui.generate_one()
    elif args.type == 'file':
        ui.generate_file(args.inp, args.out)
    
if __name__ == "__main__":
    main()
