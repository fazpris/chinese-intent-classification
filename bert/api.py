
from flask import Flask, request, jsonify, render_template
from predict import predict

# 创建 Flask 应用
app = Flask(__name__)


# 首页：返回前端页面
@app.route("/", methods=["GET"])
def get_html():
    return render_template("index.html")


# 意图识别接口
@app.route("/predict", methods=["POST"])
def get_predict():

    # 获取 JSON 数据
    data = request.get_json(silent=True)

    # 判断请求数据是否有效
    if not isinstance(data, dict):
        return jsonify({
            "code": 400,
            "message": "请求数据格式错误，请传入 JSON"
        }), 400

    # 获取用户输入文本
    utt = data.get("utt")

    # 检查输入字段
    if not isinstance(utt, str) or not utt.strip():
        return jsonify({
            "code": 400,
            "message": "请输入有效的文本，字段名为 utt"
        }), 400

    try:
        # 调用 BERT 模型进行意图预测
        result = predict({
            "utt": utt.strip()
        })

        # 统一返回格式
        if isinstance(result, dict):
            response = {
                "code": 200,
                "message": "预测成功",
                "utt": utt.strip(),
                **result
            }
        else:
            response = {
                "code": 200,
                "message": "预测成功",
                "utt": utt.strip(),
                "pred_intent": result
            }

        return jsonify(response), 200

    except Exception:
        # 记录异常信息
        app.logger.exception("意图识别预测失败")

        return jsonify({
            "code": 500,
            "message": "模型预测失败，请检查服务端日志"
        }), 500

# 启动服务
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=10086,debug=True)