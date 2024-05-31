import os
import time

time_start = time.time()
# 文件路径
path_images = './CUB_200_2011/images.txt'
path_split = './CUB_200_2011/train_test_split.txt'
path_class = "./CUB_200_2011/image_class_labels.txt"
path = "./CUB_200_2011/images/"  # 图片所在路径
# 新建txt文件
f_test = open("./CUB_200_2011/test.txt", 'w')
f_train = open("./CUB_200_2011/train.txt", 'w')
# 读取images.txt文件
images = []
with open(path_images, 'r') as f:
    for line in f:
        images.append(list(line.strip('\n').split()))
# 读取train_test_split.txt文件
split = []
with open(path_split, 'r') as f_:
    for line in f_:
        split.append(list(line.strip('\n').split()))
# 读取
clss = []
with open(path_class,'r') as f_c:
     for cls in f_c:
        clss.append(list(cls.strip('\n').split()))
# 根据train_test_split.txt文件信息1、0划分
num = len(images)  # c
print(num)
for k in range(num):
    file_name = images[k][-1]
    label = split[k][-1]
    Class = clss[k][-1]
    if int(label) == 1:  # 划分到训练集
        f_train.write(file_name + " " + Class + "\n")
    else:
        f_test.write(file_name +" " + Class + "\n")
time_end = time.time()
print('CUB200训练集和测试集划分完毕, 耗时%s!!' % (time_end - time_start))
f_test.close()
f_train.close()