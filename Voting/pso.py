import pickle
import numpy as np
from pyswarm import pso
import time

# 完整的模型路径
model_paths = {
    "gcn": {
        "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
        "ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
        "ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
        "ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
    },
    "former": {
        "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
        "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl"
    }
}

# 加载模型数据
def load_model_data(paths, num_samples=2000):
    data_list = []
    for path in paths.values():
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        data = np.array([data_dict[f"test_{i}"] for i in range(num_samples)])
        data_list.append(data)
    return np.array(data_list)

# 加载并准备数据集
def load_data(use_gcn=False, use_former=False):
    data_list = []
    if use_gcn:
        data_list.extend(load_model_data(model_paths["gcn"]))
    if use_former:
        data_list.extend(load_model_data(model_paths["former"]))
    X = np.transpose(np.array(data_list), (1, 0, 2))
    y = np.load("../labels/test_label_A.npy")  # 确保路径正确
    return X, y

# Softmax计算
def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=0)

# 计算加权预测
def weighted_prediction(X, weights):
    weighted_sum = np.tensordot(weights, np.apply_along_axis(softmax, 2, X), axes=([0], [1]))
    return np.argmax(weighted_sum, axis=-1)

# 损失函数 (负准确率)
def loss_function(weights, X, y):
    predictions = weighted_prediction(X, weights)
    accuracy = np.mean(predictions == y)
    return -accuracy

# 使用PSO优化权重
def optimize_weights(X, y):
    lb = [0] * X.shape[1]  # 下限
    ub = [2] * X.shape[1]  # 上限
    weights, _ = pso(loss_function, lb, ub, args=(X, y), swarmsize=50, maxiter=20)
    return weights

# 准确率计算
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

# 主程序
if __name__ == "__main__":
    X, y = load_data(use_gcn=True, use_former=True)
    start_time = time.time()
    
    # 优化权重
    optimized_weights = optimize_weights(X, y)
    print(f"优化完成，用时：{time.time() - start_time:.2f}秒")
    print(f"优化权重（PSO）：{optimized_weights}")

    # 加权预测并计算准确率
    predictions = weighted_prediction(X, optimized_weights)
    acc = accuracy_score(y, predictions)
    print(f"使用优化权重的准确率（PSO）：{acc}%")
