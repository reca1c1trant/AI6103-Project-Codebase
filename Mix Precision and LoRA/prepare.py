import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据
df = pd.read_csv('SQuAD-v1.1.csv', keep_default_na=True)
df = df.dropna(subset=['context', 'question', 'answer'])

# 划分训练集和测试集（90% train, 10% test）
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# 保存
train_df.to_csv('SQuAD-v1.1.csv', index=False)  # 覆盖原文件作为训练集
test_df.to_csv('test.csv', index=False)

print(f"训练集: {len(train_df)} 行 -> SQuAD-v1.1.csv")
print(f"测试集: {len(test_df)} 行 -> test.csv")