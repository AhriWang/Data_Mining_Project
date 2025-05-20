

import pandas as pd

file_path = "D:\\Datamining\\test.csv"  # 你的 CSV 文件路径
df = pd.read_csv(file_path)

# 查看所有列名
print("\n📌 数据列名：")
print(df.columns)
