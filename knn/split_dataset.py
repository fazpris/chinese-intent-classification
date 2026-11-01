
import pandas as pd
from config import Config

conf = Config()

# 读取原始数据
df = pd.read_json(conf.train_path, lines=True, encoding="utf-8")
print(df.head(50))

# 划分数据集
train_df = df[df["partition"] == "train"][["utt", "intent"]]
dev_df = df[df["partition"] == "dev"][["utt", "intent"]]
test_df = df[df["partition"] == "test"][["utt", "intent"]]

# 保存数据集
train_df.to_csv("data/train.csv", index=False, encoding="utf-8-sig")
dev_df.to_csv("data/dev.csv", index=False, encoding="utf-8-sig")
test_df.to_csv("data/test.csv", index=False, encoding="utf-8-sig")

print("训练集数量：", len(train_df))
print("验证集数量：", len(dev_df))
print("测试集数量：", len(test_df))
