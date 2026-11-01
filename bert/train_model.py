"""
核心：
1. 准备数据
2. 模型训练
3. 模型评估
"""
import time
import random
import numpy as np
import torch
import pandas as pd
from config import Config
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from tqdm import tqdm

# 固定随机种子，尽量保证实验结果可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 固定随机种子后再初始化模型
conf = Config()

# 标签与数字的映射
label_dict = {label: index for index, label in enumerate(conf.class_list)}

# 加载数据
def create_loader(file_path, shuffle=True):
    data = pd.read_csv(file_path)

    # 原数据是中文，需要转换成张量
    tokens = conf.tokenizer(
        text=data["utt"].astype(str).tolist(),
        max_length=conf.seq_len,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )

    features = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # 意图标签转换成数字
    targets = torch.tensor([label_dict[label] for label in data["intent"]])
    # 张量 -> 数据集
    dataset = TensorDataset(features, attention_mask, targets)
    # 数据集 -> 加载器
    loader = DataLoader(dataset=dataset,batch_size=conf.batch_size,shuffle=shuffle)

    return loader

# 模型训练
def model_train():
    # 加载数据
    train_loader = create_loader(conf.train_path)

    # 加载模型
    model = conf.model.to(conf.device)

    # 统计训练集中各类别的样本数量
    train_data = pd.read_csv(conf.train_path)
    class_counts = train_data["intent"].value_counts()

    counts = torch.tensor(
        [class_counts[label] for label in conf.class_list],
        dtype=torch.float
    )

    # 计算类别权重
    class_weights = torch.rsqrt(counts)
    class_weights = class_weights / class_weights.mean()

    # 加权交叉熵
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(conf.device))

    # 优化器参数分组
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        }
    ]

    # 优化器
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=conf.lr
    )

    # 学习率调度器
    total_steps = len(train_loader) * conf.epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_f1 = -1.0

    # 多轮训练
    for epoch in range(conf.epochs):
        model.train()
        total_loss, count, start = 0.0, 0, time.time()

        for batch in tqdm(train_loader, desc=f"第 {epoch + 1} 轮训练"):
            x, mask, y = batch

            x = x.to(conf.device)
            mask = mask.to(conf.device)
            y = y.to(conf.device)

            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(x, mask)
            # 计算加权交叉熵损失
            loss = loss_fn(y_pred, y)
            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()

            total_loss += loss.item()
            count += 1

        print(
            f"第 {epoch + 1} 轮，"
            f"训练损失：{total_loss / count:.4f}，"
            f"当前学习率：{scheduler.get_last_lr()[0]:.8f}，"
            f"耗时：{time.time() - start:.2f}秒"
        )

        # 验证模型
        f1 = model_valid(model)

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            model_save(model)
            print("发现更优模型，已保存！")

    print("模型训练完成！")
    print("最佳验证集 Macro-F1：", best_f1)

# 模型验证
def model_valid(model):
    dev_loader = create_loader(conf.dev_path, shuffle=False)

    model.eval()
    pred_list, true_list = [], []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="开始验证模型..."):
            x, mask, y = batch

            x = x.to(conf.device)
            mask = mask.to(conf.device)
            y = y.to(conf.device)

            y_pred = model(x, mask)
            predict = torch.argmax(y_pred, dim=1)

            pred_list.extend(predict.cpu().tolist())
            true_list.extend(y.cpu().tolist())

    # 模型评估
    labels = list(range(len(conf.class_list)))

    accuracy = accuracy_score(true_list, pred_list)
    precision = precision_score(
        true_list, pred_list,labels=labels,average="macro",zero_division=0
    )
    recall = recall_score(
        true_list, pred_list,labels=labels,average="macro",zero_division=0
    )
    f1 = f1_score(
        true_list, pred_list,labels=labels,average="macro",zero_division=0
    )

    print("\n验证集评估结果：")
    print(f"Accuracy：{accuracy:.4f}")
    print(f"Macro Precision：{precision:.4f}")
    print(f"Macro Recall：{recall:.4f}")
    print(f"Macro-F1：{f1:.4f}")

    return f1


# 模型保存
def model_save(model):
    torch.save(
        model.state_dict(),
        conf.model_path + "bert_model.bin"
    )

    # 保存分词器和配置文件
    conf.tokenizer.save_pretrained(conf.model_path)
    conf.config.save_pretrained(conf.model_path)

    print("最佳模型保存成功！")


# 模型测试
def model_test():
    test_loader = create_loader(conf.test_path, shuffle=False)

    model = conf.model.to(conf.device)

    # 加载最佳模型
    model.load_state_dict(
        torch.load(
            conf.model_path + "bert_model.bin",
            map_location=conf.device
        )
    )

    model.eval()
    pred_list, true_list = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="开始测试模型..."):
            x, mask, y = batch

            x = x.to(conf.device)
            mask = mask.to(conf.device)
            y = y.to(conf.device)

            y_pred = model(x, mask)
            predict = torch.argmax(y_pred, dim=1)

            pred_list.extend(predict.cpu().tolist())
            true_list.extend(y.cpu().tolist())

    # 模型评估
    accuracy = accuracy_score(true_list, pred_list)
    labels = list(range(len(conf.class_list)))

    precision = precision_score(
        true_list, pred_list,
        labels=labels,average="macro",zero_division=0
    )

    recall = recall_score(
        true_list, pred_list,
        labels=labels,average="macro",zero_division=0
    )

    f1 = f1_score(
        true_list, pred_list,
        labels=labels,average="macro",zero_division=0
    )

    print("\n测试集评估结果：")
    print(f"准确率：{accuracy}")
    print(f"精确率：{precision}")
    print(f"召回率：{recall}")
    print(f"F1值：{f1}")

    # 分类报告
    print("\n分类报告：")
    print(classification_report(
        true_list,
        pred_list,
        labels=list(range(len(conf.class_list))),
        target_names=conf.class_list,zero_division=0,digits=4
    ))


if __name__ == "__main__":
    model_train()
    model_test()