import torch

path2 = "C:\\Users\\choi\\PycharmProjects\\total-action\\학습결과\\latest.pth"

check = torch.load(path2)
epoch = check['epoch']
print(epoch)