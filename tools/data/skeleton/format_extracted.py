import pickle
import os
pkl_dir = "C:\\Users\\choi\\PycharmProjects\\total-action\\train_video_pkl"
pkl_dir_list = os.listdir(pkl_dir)

train_data = []
val_data = []
val_count, train_count = 0, 0

for label in pkl_dir_list:
    pkl_dir_path = os.path.join(pkl_dir, label)
    pkl_list = os.listdir(pkl_dir_path)
    for pkl in pkl_list:
        pkl_path = os.path.join(pkl_dir_path, pkl)
        if pkl[0] != "v":
            train_count += 1
            with open(pkl_path, "rb") as fr:
                while 1:
                    try:
                        data = pickle.load(fr)
                    except EOFError:
                        break
                    train_data.append(data)
        else:
            val_count += 1
            with open(pkl_path, "rb") as fr:
                while 1:
                    try:
                      data = pickle.load(fr)
                    except EOFError:
                      break
                    val_data.append(data)

print(f"train 개 수 {train_count}, val 개 수 {val_count}")
with open('./res_pkl/taxi_dataset_train.pkl', 'wb') as fr:
    pickle.dump(train_data, fr)

with open('./res_pkl/taxi_dataset_val.pkl', 'wb') as fr:
    pickle.dump(val_data, fr)