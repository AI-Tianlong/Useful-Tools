import subprocess
import os
from ATL_Tools import find_data_list, mkdir_or_exist
from tqdm import tqdm
import shutil

def unzip_with_terminal(zip_file_path, output_folder):
    """
    使用终端命令 unzip 解压 ZIP 文件
    :param zip_file_path: ZIP 文件路径
    :param output_folder: 解压目标文件夹
    """
    # 确保目标文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    try:
        # 调用 unzip 命令
        result = subprocess.run(
            ["unzip", "-o", zip_file_path, "-d", output_folder],
            check=True,
            text=True,
            capture_output=True
        )
        print(f"成功解压: {zip_file_path} 到 {output_folder}")
        print(result.stdout)  # 打印解压过程中的输出
    except subprocess.CalledProcessError as e:
        print(f"解压失败: {zip_file_path}")
        print(e.stderr)  # 打印错误信息


def delete_folder_with_terminal(folder_path):
    """
    使用终端命令删除文件夹
    :param folder_path: 要删除的文件夹路径
    """
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    try:
        # 判断操作系统
        if os.name == 'nt':  # Windows 系统
            # 使用 rmdir /s /q 删除文件夹
            # subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", folder_path], check=True)
            subprocess.run(["rm", "-rf", folder_path], check=True)
        else:  # Linux 或 macOS
            # 使用 rm -rf 删除文件夹
            subprocess.run(["rm", "-rf", folder_path], check=True)
        print(f"成功删除文件夹: {folder_path}")
    except subprocess.CalledProcessError as e:
        print(f"删除文件夹失败: {folder_path}, 错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")


def find_unzip_folders(directory):
    """
    找到指定目录中以 .SAFE 结尾的文件夹
    :param directory: 要搜索的目录路径
    :return: 以 .SAFE 结尾的文件夹列表
    """
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []

    # 遍历目录，找到以 .SAFE 结尾的文件夹
    uzip_folders = [f for f in os.listdir(directory) if f.endswith('.SAFE') and os.path.isdir(os.path.join(directory, f))]
    return uzip_folders



if __name__ == "__main__":
    # 示例：指定 ZIP 文件路径和解压目标文件夹
    zip_file_path = r'../../0-数据集-下载同时相的S2数据-zip压缩包/'  # 替换为你的 ZIP 文件路径
    output_folder = r'../../0-数据集-下载同时相的S2数据-大图文件/'  # 替换为解压目标文件夹路径
    

    zip_file_path = os.path.abspath(zip_file_path)
    output_folder = os.path.abspath(output_folder)

    unzip = True  # 是否解压文件
    transfer = True  # 是否转换文件格式

    # 解压文件
    if unzip:
        zip_list = find_data_list(zip_file_path, suffix='.zip', recursive=False)
        for zip_file in tqdm(zip_list):
            # import pdb; pdb.set_trace()  # 调试用
            if os.path.exists(os.path.join(output_folder, os.path.basename(zip_file).replace('.zip', '')+'-大图')):
                print(f'-- 文件已存在: {os.path.join(output_folder, os.path.basename(zip_file).replace(".zip", ""))}')
                continue
            else:
                unzip_with_terminal(zip_file, output_folder)
            
            # 调用函数找到 .SAFE 文件夹
            # S2_dir_list = find_unzip_folders(output_folder)
            S2_dir = os.path.join(output_folder, os.path.basename(zip_file).replace('.zip', ''))
            
            S2_img_suffix = ['_B02_10m.jp2', '_B03_10m.jp2', '_B04_10m.jp2', '_B05_20m.jp2', '_B06_20m.jp2', '_B07_20m.jp2', '_B08_10m.jp2', '_B8A_20m.jp2', '_B11_20m.jp2', '_B12_20m.jp2']
            save_img_dir = os.path.join(output_folder, os.path.basename(S2_dir)+'-大图')
            mkdir_or_exist(save_img_dir)  # 创建新文件夹
            # import pdb; pdb.set_trace()  # 调试用
            img_list = []
            for img_suffix in S2_img_suffix:
                img_ = find_data_list(S2_dir, suffix=img_suffix, recursive=True)[0]
                img_list.append(img_) # 存的符合上述后缀的文件路径

            for img_ in img_list:
                img_name = os.path.basename(img_) # T50RLS_20151020T025912_AOT_10m
                img_dir = os.path.dirname(img_)   # 文件所在的文件夹路径
                new_img_dir = os.path.join(output_folder, save_img_dir) # 新文件夹路径
                mkdir_or_exist(new_img_dir) # 创建新文件夹
                
                # import pdb; pdb.set_trace() # 调试用
                new_img_name = img_name
                new_img_path = os.path.join(new_img_dir, new_img_name)
                # 在windows中处理文件的时候，判断当前系统为windows系统
                # 处理路径长度问题, SB windows系统对路径长度有限制，超过260个字符会报错
                if os.name == 'nt':
                    # 如果是windows系统，处理路径长度问题
                    img_ = r"\\?\{}".format(os.path.abspath(img_))
                    new_img_path = r"\\?\{}".format(os.path.abspath(os.path.join(new_img_dir, img_name)))

                # 转移文件
                print(f'-- 正在复制文件: {img_} ----> {new_img_path}')
                # shutil.copy(img_, new_img_path)  # 复制文件
                if os.path.exists(new_img_path):
                    print(f'-- 文件已存在: {new_img_path}')
                    os.remove(new_img_path)  # 删除已存在的文件
                else:
                    os.rename(img_, new_img_path)
                # 删除当前解压后的文件夹
                # print('\n \n \n \n \n \n ')
            # import pdb; pdb.set_trace()  # 调试用    
            delete_folder_with_terminal(S2_dir)  # 删除文件夹

                