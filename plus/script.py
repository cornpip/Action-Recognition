import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c')
args = parser.parse_args()

print(args.c)
print(1/0)