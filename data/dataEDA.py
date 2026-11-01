import pandas as pd
from collections import Counter
from config import Config


conf = Config()
# 1. 读取数据
df = pd.read_json(
    conf.train_path,
    lines=True,
    encoding="utf-8"
)

# 2. 数据基本查看
print("========== 数据基本信息 ==========")
# print(df.head(10))
print("总样本数量：", len(df))
print("数据字段：", df.columns.tolist())

# 3. 统计数据集划分
print("\n========== 数据集划分 ==========")
partition_count = Counter(df["partition"])

for name, count in partition_count.items():
    print(f"{name}: {count}条，占比：{count / len(df):.2%}")

# 4. 统计意图类别
print("\n========== 意图类别统计 ==========")
intent_count = Counter(df["intent"])
print("意图类别数量：", len(intent_count))

for label, count in intent_count.items():
    print(
        f"意图：{label}，"
        f"数量：{count}，"
        f"占比：{count / len(df):.2%}"
    )

# 5. 各数据集的标签分布
print("\n========== 各数据集标签分布 ==========")
for partition in ["train", "dev", "test"]:
    subset = df[df["partition"] == partition]
    print(
        f"{partition}: "
        f"样本数={len(subset)}, "
        f"意图类别数={subset['intent'].nunique()}"
    )

# 6. 检查缺失值
print("\n========== 缺失值检查 ==========")
print(df[["utt", "intent", "partition"]].isnull().sum())

# 7. 检查重复文本
print("\n========== 重复文本检查 ==========")
print("重复文本数量：", df["utt"].duplicated().sum())

# 8. 检查跨数据集重复文本
print("\n========== 跨数据集重复检查 ==========")

duplicate_data = df[
    df.duplicated(subset=["utt"], keep=False)
].sort_values("utt")

# 查看重复文本涉及哪些数据集
cross_partition = duplicate_data.groupby("utt")["partition"].nunique()

cross_data = cross_partition[cross_partition > 1]

print("跨数据集重复文本数量：", len(cross_data))

print("\n跨数据集重复样本示例：")

if len(cross_data) > 0:
    print(
        duplicate_data[
            duplicate_data["utt"].isin(cross_data.index)
        ][["utt", "intent", "partition"]].head(20)
    )

# 9. 检查相同文本是否对应不同意图
print("\n========== 标签冲突检查 ==========")

conflict = df.groupby("utt")["intent"].nunique()

conflict_data = conflict[conflict > 1]

print("相同文本对应不同意图的数量：", len(conflict_data))

if len(conflict_data) > 0:
    print(
        df[df["utt"].isin(conflict_data.index)]
        [["utt", "intent", "partition"]]
        .head(20)
    )

# 10. 检查各数据集缺失的类别
print("\n========== 意图类别覆盖检查 ==========")

train_labels = set(df[df["partition"] == "train"]["intent"])
dev_labels = set(df[df["partition"] == "dev"]["intent"])
test_labels = set(df[df["partition"] == "test"]["intent"])

print("训练集类别数：", len(train_labels))
print("验证集类别数：", len(dev_labels))
print("测试集类别数：", len(test_labels))

print("验证集缺失类别：", train_labels - dev_labels)
print("测试集缺失类别：", train_labels - test_labels)