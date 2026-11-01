
import jieba
import fasttext


def predict(data):
    # 处理要预测的文本（词模式）
    text = ' '.join(jieba.lcut(data['utt']))

    # 加载模型
    model = fasttext.load_model('save-model/fasttext_model.bin')

    # 模型预测
    pred = model.predict(text)

    # 返回预测结果
    data['pred_intent'] = pred[0][0].replace('__label__', '')

    return data


# 测试
if __name__ == '__main__':
    data = {'utt': '帮我设置一个明天早上八点的闹钟'}
    result = predict(data)
    print(result)
