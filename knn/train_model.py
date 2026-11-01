import pandas as pd
import joblib
from config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

conf = Config()

# ==================== 1. 数据加载 ====================

train_df = pd.read_csv(conf.train_data_path)
dev_df = pd.read_csv(conf.dev_data_path)
test_df = pd.read_csv(conf.test_data_path)

print("训练集数量：", len(train_df))
print("验证集数量：", len(dev_df))
print("测试集数量：", len(test_df))


# ==================== 2. 数据预处理 ====================

# 提取文本和标签
X_train = train_df["utt"].astype(str)
y_train = train_df["intent"]

X_dev = dev_df["utt"].astype(str)
y_dev = dev_df["intent"]

X_test = test_df["utt"].astype(str)
y_test = test_df["intent"]


# ==================== 3. 特征工程 ====================

# TF-IDF 文本向量化
tfidf = TfidfVectorizer(
    max_features=conf.max_features,
    ngram_range=conf.ngram_range,
    analyzer="char"
)

# 训练集拟合并转换
X_train_tfidf = tfidf.fit_transform(X_train)

# 验证集和测试集只进行转换
X_dev_tfidf = tfidf.transform(X_dev)
X_test_tfidf = tfidf.transform(X_test)

print("训练集特征维度：", X_train_tfidf.shape)
print("验证集特征维度：", X_dev_tfidf.shape)
print("测试集特征维度：", X_test_tfidf.shape)


# ==================== 4. 模型训练 ====================

knn = KNeighborsClassifier(
    weights="distance",
    n_neighbors=conf.n_neighbors,
    metric=conf.metric,
    algorithm="brute"
)

knn.fit(X_train_tfidf, y_train)

print("KNN 模型训练完成！")

# 保存模型和 TF-IDF 向量化器
joblib.dump(knn, conf.model_path)
joblib.dump(tfidf, conf.tfidf_path)

print("模型保存成功！")


# ==================== 5. 模型预测 ====================

# 使用验证集进行预测
dev_pred = knn.predict(X_dev_tfidf)

# 使用测试集进行预测
test_pred = knn.predict(X_test_tfidf)

print("验证集预测完成！")
print("测试集预测完成！")


# ==================== 6. 模型评估 ====================

def evaluate_model(y_true, y_pred, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\n========== {dataset_name}评估结果 ==========")
    print(f"Accuracy：{accuracy:.4f}")
    print(f"Precision：{precision:.4f}")
    print(f"Recall：{recall:.4f}")
    print(f"Macro-F1：{f1:.4f}")


# # 分类报告
# print("\n测试集分类报告：")
# print(classification_report(y_test, test_pred, zero_division=0))
#
# # 混淆矩阵
# cm = confusion_matrix(y_test, test_pred, labels=knn.classes_)
#
# plt.figure(figsize=(20, 16))
# sns.heatmap(cm, cmap="Blues", xticklabels=knn.classes_,
#             yticklabels=knn.classes_)
#
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("KNN Confusion Matrix")
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()

# 验证集评估
evaluate_model(y_dev, dev_pred, "验证集")

# 测试集评估
evaluate_model(y_test, test_pred, "测试集")
