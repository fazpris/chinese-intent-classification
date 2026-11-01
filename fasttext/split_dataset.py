
import jieba
import pandas as pd
from config import Config

conf = Config()


# 数据预处理
def process_data(in_path, out_path):
    # 读取数据
    data = pd.read_csv(in_path)

    # 按词分词
    words = [' '.join(jieba.lcut(str(utt))) for utt in data["utt"]]

    # 标签转换
    targets = ['__label__' + str(intent) for intent in data["intent"]]

    # 标签和文本组合
    lines = zip(targets, words)

    # 保存处理后的数据
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line[0] + ' ' + line[1] + '\n')


# 数据集路径
path_dict = {
    conf.train_data_path: conf.process_train_path,
    conf.dev_data_path: conf.process_dev_path,
    conf.test_data_path: conf.process_test_path
}

# 分别处理训练集、验证集和测试集
for key, value in path_dict.items():
    process_data(key, value)