class Config:
    def __init__(self):
        # 数据集路径
        self.train_data_path = "data/train.csv"
        self.dev_data_path = "data/dev.csv"
        self.test_data_path = "data/test.csv"

        # 预处理数据保存路径
        self.process_train_path = "data/train.txt"
        self.process_dev_path = "data/dev.txt"
        self.process_test_path = "data/test.txt"

        # FastText 参数
        self.lr = 0.1
        self.epoch = 50
        self.wordNgrams = 2
        self.dim = 100

        # 模型保存路径
        self.model_path = "save-model/fasttext_model.bin"