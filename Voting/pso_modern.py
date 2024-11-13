import pickle
import numpy as np
import pyswarms as ps
import time

# 模型路径定义
model_paths = {
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
        "degcn_J_3d": "../scores/Mix_GCN/degcn_J_3d.pkl",
        "degcn_B_3d": "../scores/Mix_GCN/degcn_B_3d.pkl",
        "tegcn_V1_J_3d": "../scores/Mix_GCN/tegcn_V1_J_3d.pkl"
    },
    "former": {
        "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
        "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
        "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
        "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
        "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
        "former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
        "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
        "former_b_3d_resample_rotate": "../scores/Mix_Former/mixformer_B_3d_resample_rotate.pkl",
        "skateformer_j_3d": "../scores/Mix_Former/skateformer_B_3d.pkl"
    }
}

# 加载数据
def load_data(gcn=False, former=False):
    data = []
    if gcn:
        data.extend(load_model_scores(model_paths["gcn"]))
    if former:
        data.extend(load_model_scores(model_paths["former"]))
    X = np.transpose(np.array(data), (1, 0, 2))
    y = np.load("test_label_A.npy")  # 实际标签文件路径
    return X, y

# 加载模型分数
def load_model_scores(paths):
    scores = []
    for path in paths.values():
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        scores.append([data_dict[f"test_{i}"] for i in range(2000)])
    return scores

# Softmax 函数
def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=0)

# 计算加权预测
def weighted_prediction(X, weights):
    weighted_sum = np.tensordot(weights, np.apply_along_axis(softmax, 2, X), axes=([0], [1]))
    return np.argmax(weighted_sum, axis=-1)

# 损失函数
def loss(weights, X, y):
    return -np.mean(weighted_prediction(X, weights) == y)

# 使用PSO优化权重
def optimize_weights(X, y, init_pos=None):
    bounds = ([-1] * X.shape[1], [4] * X.shape[1])
    options = {'c1': 0.5, 'c2': 0.5, 'w': 5.0}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=50,
        dimensions=X.shape[1],
        options=options,
        bounds=bounds,
        init_pos=init_pos
    )
    cost, pos = optimizer.optimize(lambda w: [loss(w, X, y)], iters=40)
    return pos

# 准确率计算
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

if __name__ == "__main__":
    X, y = load_data(gcn=True, former=True)
    start_time = time.time()

    # 假设初始权重是均匀分布的
    init_pos = np.random.uniform(-1, 4, size=(50, X.shape[1]))

    # 优化权重
    optimized_weights = optimize_weights(X, y, init_pos=init_pos)
    print(f"优化完成，耗时：{time.time() - start_time:.2f} 秒")
    print(f"优化权重 (PSO): {optimized_weights}")

    # 预测并计算准确率
    predictions = weighted_prediction(X, optimized_weights)
    acc = accuracy(y, predictions)
    print(f"优化后准确率 (PSO): {acc}%")
