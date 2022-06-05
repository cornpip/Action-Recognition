import argparse

parser = argparse.ArgumentParser()
parser.add_argument('first')
parser.add_argument('--pose-config')
args = parser.parse_args()

print(args.first)
print(args.pose_config)