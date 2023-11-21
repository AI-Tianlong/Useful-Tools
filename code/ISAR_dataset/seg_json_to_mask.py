import argparse
import json
import os
import os.path as osp
import warnings
import copy
import numpy as np
import PIL.Image
from skimage import io
import yaml
from labelme import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file',default='/media/atl/Data1/ATL/AI_work/ISAR/空对空的RGB图片')   # 标注文件json所在的文件夹
    parser.add_argument('-o', '--out', default='/media/atl/Data1/ATL/AI_work/ISAR/空对空的RGB图片/正面')
    args = parser.parse_args()

    json_file = args.json_file

    list = os.listdir(json_file)   # 获取json文件列表
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])  # 获取每个json文件的绝对路径
        filename = list[i][:-5]       # 提取出.json前的字符作为文件名，以便后续保存Label图片的时候使用
        extension = list[i][-4:]
        if extension == 'json':
            if os.path.isfile(path):
                data = json.load(open(path))
                img = utils.image.img_b64_to_arr(data['imageData'])  # 根据'imageData'字段的字符可以得到原图像
                # lbl为label图片（标注的地方用类别名对应的数字来标，其他为0）lbl_names为label名和数字的对应关系字典
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])   # data['shapes']是json文件中记录着标注的位置及label等信息的字段

                #captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
                #lbl_viz = utils.draw.draw_label(lbl, img, captions)
                out_dir = osp.basename(list[i])[:-5]+'_json'
                out_dir = osp.join(osp.dirname(list[i]), out_dir)
                if not osp.exists(out_dir):
                    os.mkdir(out_dir)

                PIL.Image.fromarray(img).save(osp.join(out_dir, '{}_source.png'.format(filename)))
                PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}_mask.png'.format(filename)))
                #PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.jpg'.format(filename)))

                with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                    for lbl_name in lbl_names:
                        f.write(lbl_name + '\n')

                warnings.warn('info.yaml is being replaced by label_names.txt')
                info = dict(label_names=lbl_names)
                with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                    yaml.safe_dump(info, f, default_flow_style=False)

                print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()

