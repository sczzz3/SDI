## Extract_Part

首先使用 Polyman 分别导出 data 和 annotation文件

通过此程序得到的是每 3001个数据为一个整体的文本

（3000 个数据 -- 30s（100Hz）+ 1个数据 -- 标签）

后续模型在 Dataset部分会直接对这一步生成的文件进行处理



## Combination

将多个 *** Extract_part***提取后的数据拼接起来

（相当于把多个病人的数据合起来，增大数据量并直接分析，比较方便）

