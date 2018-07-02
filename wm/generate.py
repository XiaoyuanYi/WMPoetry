from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from generate_base import Generator
import argparse
import copy
from DataTool import data_tool

def parse_args():
    parser = argparse.ArgumentParser(description="The parametrs for the generator.")
    parser.add_argument("--inp", '-o', type=str, help="The input file path, for file generation.")
    parser.add_argument("--out", '-o', type=str, help="The output file path, for file generation.")
    parser.add_argument("--type", type=str, choices=['one', 'file'], 
        help='The mode. one: just generate one poem; file: generate by the input file.')
    parser.add_argument("--model", '-o', type=str, help="The checkpoint path. If none, just use the newest checkpoint.")
    parser.add_argument("--bsize", type=int, help="The beam size.")
    return parser.parse_args()


class GeneratorUI(object):

    def __init__(self, beam_size, modelfile):
        self.generator = Generator(beam_size, modelfile)
        self.dtool = data_tool


    def generate_one(self):
        while True:
            keys = raw_input("please input keywords (with whitespace split) > ")
            pattern_id = input("please select  genre pattern > ")
            yun = input("please input yun type> ")
            pattern = self.dtool.getPattern(pattern_id)
            pattern = pattern[1]
            name = pattern[0]
            print ("select pattern: %s" % (name))
            ans, info = self.generator.generate_one(keys, pattern, yun)
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

    
    def generate_specify_file(self, infile, outfile):

        fin = open(infile, 'r')
        lines = fin.readlines()
        fin.close()

        fout = open(outfile, 'w')
        for i, line in enumerate(lines):
            line = line.strip()
            para = line.split("|")
            wstr = para[0].strip()
            print ("%d  keys: %s" % (i, wstr))
            pattern = self.modiPattern(para[1:])

            sens, info = self.generator.generate_specify(wstr, pattern, beam_size)
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
        ui.generate_specify_file(args.infile, args.outfile)
    
if __name__ == "__main__":
    main()
