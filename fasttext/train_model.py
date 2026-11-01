
import fasttext
import pandas as pd
import jieba
from config import Config

conf = Config()

# 1. 数据加载
train_df = pd.read_csv(conf.train_data_path)
dev_df = pd.read_csv(conf.dev_data_path)
test_df = pd.read_csv(conf.test_data_path)

print("训练集数量：", len(train_df))
print("验证集数量：", len(dev_df))
print("测试集数量：", len(test_df))


# 2. 数据预处理
def predict_data(data):
    predictions = []

    for text in data:
        text = ' '.join(jieba.lcut(str(text)))
        result = model.predict(text)
        predictions.append(result[0][0].replace("__label__", ""))

    return predictions


# 3. 特征工程
# FastText 在训练过程中自动完成文本特征处理


# 4. 模型训练
model = fasttext.train_supervised(
    input=conf.process_train_path,
    lr=conf.lr,
    epoch=conf.epoch,
    wordNgrams=conf.wordNgrams,
    dim=conf.dim,
    loss="softmax"
)

# 模型保存
model.save_model(conf.model_path)
print("FastText 模型保存成功！")


# 5. 模型预测
dev_pred = predict_data(dev_df["utt"])
test_pred = predict_data(test_df["utt"])

print("验证集预测完成！")
print("测试集预测完成！")


# 6. 模型评估
print("\n验证集评估结果：")
dev_result = model.test(conf.process_dev_path)
print(dev_result)

print("\n测试集评估结果：")
test_result = model.test(conf.process_test_path)
print(test_result)
