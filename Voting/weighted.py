import pickle
import numpy as np

# 模型路径配置
models = {
    "gcn": {
        "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
        "ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
        "ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
        "ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
        "ctrgcn_j_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample_rotate.pkl",
        "ctrgcn_b_2d": "../scores/Mix_GCN/ctrgcn_V1_B_2d.pkl",
        "ctrgcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_J_2d.pkl",
        "ctrgcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
        "ctrgcn_jm_2d": "../scores/Mix_GCN/ctrgcn_V1_JM_2d.pkl",
        "tdgcn_j_2d": "../scores/Mix_GCN/tdgcn_V1_J_2d.pkl",
        "blockgcn_j_3d": "../scores/Mix_GCN/blockgcn_J_3d.pkl",
        "blockgcn_jm_3d": "../scores/Mix_GCN/blockgcn_JM_3d.pkl",
        "blockgcn_b_3d": "../scores/Mix_GCN/blockgcn_B_3d.pkl",
        "blockgcn_bm_3d": "../scores/Mix_GCN/blockgcn_BM_3d.pkl",
        "ctrgcn_b_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_B_3d_resample_rotate.pkl",
        "degcn_J_3d": "../scores/Mix_GCN/degcn_J_3d.pkl"
    },
    "former": {
        "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
        "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
        "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
        "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
        "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
        "former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
        "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl"
    }
}

# 加载数据函数
def load_data(use_gcn=False, use_former=False, num_samples=2000):
    data_list = []
    if use_gcn:
        data_list.extend(load_model_data(models["gcn"], num_samples))
    if use_former:
        data_list.extend(load_model_data(models["former"], num_samples))
    
    data_np = np.array(data_list)
    print(f"Loaded data shape: {data_np.shape}")
    
    X = data_np.transpose(1, 0, 2)  # 重塑数据形状为 (samples, models, features)
    y = np.load("../labels/test_label_A.npy")  # 加载标签路径
    return X, y

# 加载并转换模型数据
def load_model_data(paths, num_samples):
    data = []
    for path in paths.values():
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        data.append(np.array([model_data[f"test_{i}"] for i in range(num_samples)]))
    return data

# softmax 函数
def softmax(X):
    exp_X = np.exp(X)
    return exp_X / np.sum(exp_X, axis=0)

# 加权预测
def weighted_vote(X, weights):
    final_pred = []
    for sample in X:
        weighted_sum = np.sum([w * softmax(pred) for w, pred in zip(weights, sample)], axis=0)
        final_pred.append(np.argmax(weighted_sum))
    return np.array(final_pred)

# 硬投票
def hard_vote(X):
    final_pred = []
    for sample in X:
        votes = [np.argmax(pred) for pred in sample]
        final_pred.append(max(set(votes), key=votes.count))
    return np.array(final_pred)

# 计算准确率
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

# 主函数
if __name__ == "__main__":
    # 定义模型权重
    weights = [0.367, 0.049, 0.49, 0.963, 0.479, 1.15, 0.385, 0.368, 0.334, 0.739, 0.906, 
               0.576, 1.183, 0.922, 0.248, 0.55, 0.012, 0.571, 0.335, 0.460, -0.391, 
               0.344, 0.248]

    # 加载数据并进行加权预测
    X, y = load_data(use_gcn=True, use_former=True)
    predictions = weighted_vote(X, weights)
    acc = accuracy(y, predictions)
    print(f"Weighted Voting Accuracy: {acc}%")
