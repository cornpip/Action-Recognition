import subprocess
import os
path_dir = "C:\\Users\\choi\\PycharmProjects\\total-action\\train_video"
label_files = os.listdir(path_dir)
count = 0
print(label_files)
basic_path = "C:\\Users\\choi\\PycharmProjects\\total-action\\train_video_pkl"
for i in label_files:
  label_path = os.path.join(path_dir, i)
  extract_files = os.listdir(label_path)

  print(extract_files)

  for j in extract_files:
    vid = os.path.join(label_path, j)
    name, ext = os.path.splitext(j)
    res = os.path.join(basic_path, name + ".pkl")
    print(f'\n{i}_____{j}')
    proc = subprocess.Popen(
        ['python', 'ntu_pose_extraction.py', vid, res],
        stdout=subprocess.PIPE
    )
    out, err = proc.communicate()
    print(out.decode('utf-8'), end="")
    count += 1
    # if err: print(err) 프로세스 멈춰도 err에 안걸림

print(f"들어온 영상 개수: {count}")