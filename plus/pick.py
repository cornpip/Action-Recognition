import pickle

a = ["test_dataset_train.pkl","test_dataset_val.pkl"]
result_data = []
for i in a:
    with open(i,"rb") as fr:
        while True:
            try:
                data = pickle.load(fr)
            except EOFError:
                break
            result_data.append(data)


# for i, val in enumerate(result_data):
#     a = val["keypoint_score"]
#     print(len(a))
    # if i==1:
    #     print(len(a[1]))
    #     continue
    # print(a)

for i, val in enumerate(result_data):
    a = val[0]
    print(len(val))