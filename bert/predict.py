
import torch
from config import Config

conf = Config()


def predict(data):
    # 加载模型
    model = conf.model.to(conf.device)

    model.load_state_dict(
        torch.load(
            conf.model_path + "bert_model.bin",
            map_location=conf.device
        )
    )
    model.eval()

    # 文本转换成张量
    tokens = conf.tokenizer(
        text=data["utt"],
        return_tensors="pt",
        max_length=conf.seq_len,
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        x = tokens["input_ids"].to(conf.device)
        mask = tokens["attention_mask"].to(conf.device)

        y_pred = model(x, mask)

    # 获取预测类别
    index = torch.argmax(y_pred, dim=1).item()
    data["pred_intent"] = conf.class_list[index]

    return data


# 测试
if __name__ == "__main__":
    data = {
        "utt": "帮我设置一个明天早上八点的闹钟"
    }

    result = predict(data)
    print(result)