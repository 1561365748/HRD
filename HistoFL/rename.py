import os
import sys

folder_name = "DATA_ROOT_DIR/classification_features_dir/h5_files/"  # 获取文件夹的名字，即路径
new_str = 'slide_'
# 增加字符串
def add_str():
    file_names = os.listdir(folder_name)  # 获取文件夹内所有文件的名字
    for name in file_names:  # 如果某个文件名在file_names内
        old_name = folder_name + '/' + name  # 获取旧文件的名字，注意名字要带路径名
        new_name = folder_name + '/' + new_str + name  # 定义新文件的名字，这里给每个文件名前加了前缀 a_
        os.rename(old_name, new_name)  # 用rename()函数重命名
        print(old_name, '======>', new_name)  # 打印新的文件名字

def sub_str():
    # 获取该目录下所有文件，存入列表中
    fileList = os.listdir(folder_name)
    for file in fileList:
        old = folder_name + '/' + file
        new = folder_name + '/' + ''.join(file.split(new_str))  # join和split方法
        os.rename(old, new)  # rename方法
        print(old, '======>', new)

if __name__ == "__main__":
    # sub_str()
    add_str()



