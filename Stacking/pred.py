import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pickle

# 文件路径配置
file_paths = {
    "former": {
        "former_b_m_r_w": "",
        "former_b_m": "",
        "former_j": "",
    },
    "gcn": {
        "gcn_b_m": "Mix_GCN/test/ctrgcn_V1_J_3d_bone_vel.pkl",
        "gcn_j": "Mix_GCN/test/ctrgcn_V1_J_3d.pkl"
    }
}

DATA_SIZE = 4599

# 加载数据
def load_data(gcn=False, former=False):
    data_list = []

    def load_files(path_dict):
        for _, path in path_dict.items():
            with open(path, 'rb') as f:
                data_dict = pickle.load(f)
                data = np.array([data_dict[f"test_{i}"] for i in range(DATA_SIZE)])
                data_list.append(data)

    if former:
        load_files(file_paths["former"])
    if gcn:
        load_files(file_paths["gcn"])

    return np.concatenate(data_list, axis=1)

# 定义元学习模型
class MetaLearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 生成置信度
def generate_confidences(model, dataloader, output_file="pred.npy"):
    model.eval()
    all_confidences = []
    with torch.no_grad():
        for X_batch, in dataloader:
            probabilities = torch.softmax(model(X_batch.float()), dim=1)
            all_confidences.append(probabilities.numpy())

    np.save(output_file, np.vstack(all_confidences))
    print(f"置信度保存到 {output_file}")

if __name__ == "__main__":
    # 加载数据并创建 DataLoader
    X = load_data(gcn=True, former=False)
    dataloader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=32, shuffle=False)

    # 初始化模型并加载权重
    model = MetaLearner(input_dim=X.shape[1], output_dim=155)
    model.load_state_dict(torch.load("meta_learner_weights.pth"))

    # 生成并保存置信度
    generate_confidences(model, dataloader)
