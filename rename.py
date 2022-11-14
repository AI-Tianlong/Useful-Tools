# 将训练集图像和标签重新命名

import os

img_dir = '../dz_data/clear_dz_train/images'
label_dir = '../dz_data/clear_dz_train/labels'
img_list = os.listdir(img_dir)

# print(img_list[0])
# images_original_0.png_62d06f05-bca4-4071-b965-a2b8bcaddccf.png
# _groundtruth_(1)_images_0.png_62d06f05-bca4-4071-b965-a2b8bcaddccf.png

for i in range(len(img_list)):

    img_name = img_list[i]
    label_name = str('_groundtruth_(1)_') + img_name.replace('_original_', '_')

    # rename img
    src_img = os.path.join(img_dir, img_name)
    dst_img = os.path.join(img_dir, str(i) + '.png')
    os.rename(src_img, dst_img)

    # rename label
    src_label = os.path.join(label_dir, label_name)
    dst_label = os.path.join(label_dir, str(i) + '.png')
    os.rename(src_label, dst_label)

    print("已完成{}张图像的改名".format(i+1))


    # break