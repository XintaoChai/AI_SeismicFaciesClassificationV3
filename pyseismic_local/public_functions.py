import os
import glob  # glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合。


# glob模块用来查找文件目录和文件，glob支持*?[]这三种通配符
def findFileNumber(path_dir, file_suffix):
    # glob.glob返回所有匹配的文件路径列表。
    # 它只有一个参数pathname，定义了文件路径匹配规则，
    # 这里可以是绝对路径，也可以是相对路径。
    file_list = glob.glob(os.path.join(path_dir, '*.' + file_suffix))
    File_Number = len(file_list)
    return File_Number
