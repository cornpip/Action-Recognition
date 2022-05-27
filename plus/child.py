import subprocess

label = ["P", "N", "R", "L"]
filelist = ["punch","nikick", "right-kick", "left-kick"]
# for i in range(4):
#     name1 = label[i]
#     name2 = filelist[i]
#     try:
#         num = 1
#         while 1:
#             res_path = "./result/" + name1 + "00" + str(num) + "A00" + str(i+1)
#             vidpath = "/content/drive/MyDrive/action/test/" + name2 + "/" + name1 + "00" + str(num) + "A00" + str(i+1)
#             proc = subprocess.Popen(
#                 ['python', 'ntu_pose_extraction.py', vidpath, res_path],
#                 stdout=subprocess.PIPE
#             )
#             out, err = proc.communicate()
#             print("child 예욤", out.decode('utf-8'), end="")
#             num += 1
#             print(vidpath, "----", res_path)
#     except:
#         print(f'except:{vidpath}++{res_path}')
#         continue

proc = subprocess.Popen(
    ['python', 'script.py', '-c', '3'],
    stdout=subprocess.PIPE
)
out, err = proc.communicate()
print(out.decode('utf-8'), end="")
print("-------------")
print(f"aaa")