import os
import shutil
import random

src_path = r"./handpose_x_gesture_v1"
dst_path = r"./data"

if not os.path.exists(dst_path + "/train"):
    os.makedirs(dst_path + "/train")
if not os.path.exists(dst_path + "/val"):
    os.makedirs(dst_path + "/val")

select_list = os.listdir(src_path)

split_ratio = 0.8
min_cls_nums = 1

max_cls_nums = 20000

select_list_last = []
for cls_name in select_list:
    if len(os.listdir(os.path.join(src_path, cls_name))) > min_cls_nums:
        select_list_last.append(cls_name)

for cls_name in select_list_last:
    cls_path = os.path.join(src_path, cls_name)
    cls_imgs = os.listdir(cls_path)
    cls_nums = len(cls_imgs)
    print(cls_name, ": ", cls_nums)

    random.shuffle(cls_imgs)

    cls_imgs_tmp = []

    if cls_nums > max_cls_nums:
        cls_nums = max_cls_nums
        cls_imgs = cls_imgs[:max_cls_nums]
        # for im in cls_imgs_tmp:
        #     if im not in cls_imgs:
        #         cls_imgs.append(im)
        #         print("----------------------------")
    random.shuffle(cls_imgs)
    # train
    for im in cls_imgs[:int(cls_nums * split_ratio)]:
        if not os.path.exists(dst_path + "/train/" + cls_name):
            os.makedirs(dst_path + "/train/" + cls_name)

        src = os.path.join(cls_path, im)
        dst = dst_path + "/train/" + cls_name + "/" + im
        print(src, " copy to ", dst)
        shutil.copyfile(src, dst)
    # val
    for im in cls_imgs[int(cls_nums * split_ratio):]:
        if not os.path.exists(dst_path + "/val/" + cls_name):
            os.makedirs(dst_path + "/val/" + cls_name)
        src = os.path.join(cls_path, im)
        dst = dst_path + "/val/" + cls_name + "/" + im
        print(src, " copy to ", dst)
        shutil.copyfile(src, dst)







