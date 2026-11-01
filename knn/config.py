class Config:
    def __init__(self):
        # 原始数据路径
        self.train_path = "../data/zh-CN.jsonl"

        # 划分后的数据集路径
        self.train_data_path = "data/train.csv"
        self.dev_data_path = "data/dev.csv"
        self.test_data_path = "data/test.csv"

        # TF-IDF 参数
        self.max_features = 15000
        self.ngram_range = (1, 2)

        # KNN 参数
        self.n_neighbors = 11
        self.metric = "cosine"
        # self.metric = "euclidean"

        # 模型保存路径
        self.model_path = "save-model/knn_model.pkl"
        self.tfidf_path = "save-model/tfidf.pkl"