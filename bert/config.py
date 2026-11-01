
import torch
import pandas as pd
from transformers import BertModel, BertTokenizer, BertConfig


# 模型相关信息
model_path = "../models/chinese-roberta-wwm-ext"


# BERT 分类模型
class MyModel(torch.nn.Module):
    def __init__(self, num_hidden, num_class):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.out = torch.nn.Linear(num_hidden, num_class)

    def forward(self, x, mask):
        bert_pred = self.bert(x, attention_mask=mask)
        y_pred = self.out(bert_pred.pooler_output)
        return y_pred


class Config:
    def __init__(self):
        # 数据来源：训练集、测试集、验证集
        self.train_path = "data/train.csv"
        self.test_path = "data/test.csv"
        self.dev_path = "data/dev.csv"

        # 模型保存地址
        self.model_path = "save-model/"

        # 设备
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 模型训练参数
        self.epochs = 5
        self.lr = 0.00005
        self.batch_size = 32
        self.seq_len = 64

        # BERT 相关参数
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.config = BertConfig.from_pretrained(model_path)

        # 获取意图类别
        data = pd.read_csv(self.train_path)
        self.class_list = sorted(data["intent"].unique().tolist())
        self.num_class = len(self.class_list)

        # 初始化模型
        self.model = MyModel(
            self.config.hidden_size,
            self.num_class
        )