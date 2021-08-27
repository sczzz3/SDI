import os
import linecache


def combine_file():
    # 读取指定路径下的所有文件并放入到列表中
    root = input('Directory: ')
    file_names = os.listdir(root)
    print(file_names)
    file_ob_list = []
    for file_name in file_names:
        fileob = root + '/' + file_name
        file_ob_list.append(fileob)

    # 对每个文件，按行读取文件内容并放入同一个列表data中
    data = []
    for file_ob in file_ob_list:
        line_num = 1
        length_file = len(open(file_ob, encoding='utf-8').readlines())
        print(length_file)
        while line_num <= length_file:
            line = linecache.getline(file_ob, line_num)
            line = line.strip()
            data.append(line)
            line_num = line_num + 1

    # 将data内容写入到生成的txt文件中，注意编码问题
    f = open('E:/data/combine_data.txt', 'w', encoding='utf-8')
    for i, p in enumerate(data):
        f.write(p + '\n')
    f.close()


combine_file()
