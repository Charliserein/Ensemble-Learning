import pickle
import numpy as np
from scipy.optimize import minimize

# 模型路径定义
model_paths = {
    "former": {
        "former_b_m_r_w": "../scores/Mix_Former/mixformer_BM_r_w.pkl",
        "former_b_m": "../scores/Mix_Former/mixformer_BM_r_w.pkl",
        "former_j": "../scores/Mix_Former/mixformer_J.pkl",
    },
    "gcn": {
        "gcn_b_m": "../scores/Mix_GCN/ctrgcn_V1_J_3d_bone_vel.pkl",
        "gcn_j": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
        "gcn_b": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
        "gcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
        "gcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
    }
}

def load_data(use_gcn=False, use_former=False):
    data_list = []
    for model_type, paths in model_paths.items():
        if (model_type == "gcn" and use_gcn) or (model_type == "former" and use_former):
            for path in paths.values():
                with open(path, 'rb') as f:
                    data_dict = pickle.load(f)
                data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
                data_list.append(data)
    
    X = np.stack(data_list, axis=1)
    y = np.load("test_label_A.npy")
    return X, y

def softmax(X):
    e_x = np.exp(X - np.max(X, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def weighted_predictions(X, weights):
    predictions = []
    for sample in X:
        weighted_sum = sum(w * softmax(model) for w, model in zip(weights, sample))
        predictions.append(np.argmax(weighted_sum))
    return np.array(predictions)

def loss_function(weights, X, y):
    return -np.mean(weighted_predictions(X, weights) == y)

def optimize_weights(X, y):
    initial_weights = np.random.uniform(0, 1, X.shape[1])
    bounds = [(0, 3)] * X.shape[1]
    result = minimize(loss_function, initial_weights, args=(X, y), bounds=bounds, method='SLSQP')
    return result.x

def hard_voting(X):
    return np.array([np.argmax(np.bincount([np.argmax(model) for model in sample])) for sample in X])

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

if __name__ == "__main__":
    X, y = load_data(use_gcn=True, use_former=True)
    opt_weights = optimize_weights(X, y)
    print(f"Optimized Weights: {opt_weights}")

    weighted_acc = accuracy(y, weighted_predictions(X, opt_weights))
    print(f"Weighted Accuracy: {weighted_acc}%")

    hard_vote_acc = accuracy(y, hard_voting(X))
    print(f"Hard Voting Accuracy: {hard_vote_acc}%")
