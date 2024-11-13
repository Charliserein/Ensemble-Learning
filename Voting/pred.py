import pickle
import numpy as np


model_paths = {
    "gcn": {
        "ctrgcn_jm_3d": "../scores/Mix_GCN/test/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/test/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d_resample.pkl",
    "ctrgcn_j_3d_resample_rotate": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d_resample_rotate.pkl",
    "ctrgcn_b_2d": "../scores/Mix_GCN/test/ctrgcn_V1_B_2d.pkl",
    "ctrgcn_j_2d": "../scores/Mix_GCN/test/ctrgcn_V1_J_2d.pkl",
    "ctrgcn_bm_2d": "../scores/Mix_GCN/test/ctrgcn_V1_BM_2d.pkl",
    "ctrgcn_jm_2d": "../scores/Mix_GCN/test/ctrgcn_V1_JM_2d.pkl",
    "tdgcn_j_2d": "../scores/Mix_GCN/test/tdgcn_V1_J_2d.pkl",
    "blockgcn_j_3d": "../scores/Mix_GCN/test/blockgcn_J_3d.pkl",
    "blockgcn_jm_3d": "../scores/Mix_GCN/test/blockgcn_JM_3d.pkl",
    "blockgcn_b_3d": "../scores/Mix_GCN/test/blockgcn_B_3d.pkl",
    "blockgcn_bm_3d": "../scores/Mix_GCN/test/blockgcn_BM_3d.pkl",
    "ctrgcn_b_3d_resample_rotate": "../scores/Mix_GCN/test/ctrgcn_V1_B_3d_resample_rotate.pkl",
    "degcn_J_3d": "../scores/Mix_GCN/test/degcn_J_3d.pkl",
    "degcn_B_3d": "../scores/Mix_GCN/test/degcn_B_3d.pkl",
    "tegcn_V1_J_3d": "../scores/Mix_GCN/test/tegcn_V1_J_3d.pkl"
    },
    "former": {
        "former_bm_r_w_2d": "../scores/Mix_Former/test/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/test/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/test/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/test/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/test/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/test/mixformer_J_3d_resample_rotate.pkl",
    "former_jm_2d": "../scores/Mix_Former/test/mixformer_JM_2d.pkl",
    "former_b_3d_resample_rotate": "../scores/Mix_Former/test/mixformer_B_3d_resample_rotate.pkl",
    "skateformer_j_3d": "../scores/Mix_Former/test/skateformer_B_3d.pkl",
}
    }
}

weights = [0.293, 0.016, 0.617, 0.695, 0.720, 0.785, 1.057, -0.143, 1.387, -0.777, 2.403, -0.072, 2.503, 1.656, 1.622, 1.570, 1.371, 3.231, 0.991, 1.522, 0.718, 0.231, 1.027, 0.310, 0.017, 0.641, 1.932]

def load_data(use_gcn=False, use_former=False):
    data = []
    for model_type, paths in model_paths.items():
        if (model_type == "gcn" and use_gcn) or (model_type == "former" and use_former):
            for path in paths.values():
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                data.append(np.array([model_data[f"test_{i}"] for i in range(4599)]))
    return np.array(data).transpose(1, 0, 2)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# 生成置信度
def generate_confidence(data, weights):
    confidences = []
    for sample in data:
        weighted_sum = sum(w * softmax(model) for w, model in zip(weights, sample))
        confidences.append(weighted_sum)
    return np.array(confidences)

if __name__ == "__main__":
    X = load_data(use_gcn=True, use_former=True)
    confidences = generate_confidence(X, weights)
    np.save("pred.npy", confidences)
    print("Confidence scores have been saved to 'pred.npy'")
