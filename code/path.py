# Copyright (c) OpenMMLab. All rights reserved.

"""
用法：
from path import scandir

for label_path in scandir(dataset_path, suffix='.png', recursive=True):
    if 'labels' in label_path and 'labels_mask' not in label_path:
        rgb_label_path = osp.join(dataset_path, label_path)
        RGB_labels_path.append(rgb_label_path)
        if 'v1.2' in label_path:
            RGB_labels_v1_2_path.append(rgb_label_path)
        elif 'v2.0' in label_path:
            RGB_labels_v2_0_path.append(rgb_label_path)

"""

import os
import os.path as osp
from pathlib import Path
from typing import List

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Default: True.

    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = suffix.lower() if isinstance(suffix, str) else tuple(
            item.lower() for item in suffix)

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive,
                                    case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def find_data_list(img_root_path: str, suffix: str ='.jpg') -> List:
    
    """根据给定的数据集根目录，找出其子文件夹下
    的所有符合后缀的数据集图片完整路径

    """
    print('\n==============================================================')
    print('-- 正在读取数据集列表...')

    img_list = []
    for img_name in scandir(img_root_path, suffix=suffix, recursive=True):
        if suffix in img_name:
            img_path = os.path.join(img_root_path, img_name)
            img_list.append(img_path)
    print(f'-- 共在 {img_root_path} 下寻找到图片 {len(img_list)} 张')

    return img_list


