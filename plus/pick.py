import pickle

path = "C:\\Users\\choi\\PycharmProjects\\total-action\\mmaction2\\work_dirs\\taxi_keypoint\\epoch_100.pth"
path2 = "C:\\Users\\choi\\PycharmProjects\\total-action\\학습결과\\latest.pth"
result_data = []

a = ["test_dataset_train.pkl","test_dataset_val.pkl"]
with open(path2, "rb") as fr:
    while True:
        try:
            data = pickle.load(fr)
        except EOFError:
            break
        result_data.append(data)

print(result_data)
print(len(result_data))

# for i in a:
#     with open(i,"rb") as fr:
#         while True:
#             try:
#                 data = pickle.load(fr)
#             except EOFError:
#                 break
#             result_data.append(data)


# for i, val in enumerate(result_data):
#     a = val["keypoint_score"]
#     print(len(a))
    # if i==1:
    #     print(len(a[1]))
    #     continue
    # print(a)

# for i, val in enumerate(result_data):
#     a = val[0]
#     print(len(val))