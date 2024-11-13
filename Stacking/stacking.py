import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import logging

# 日志配置
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 模型路径
model_paths = {
    "GCN": {
        "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
        # 其他 GCN 模型路径
    },
    "Former": {
        "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
        # 其他 Former 模型路径
    }
}

# 加权损失计算
def calculate_class_weights(labels, num_classes=155):
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 - (counts / len(labels))
    logging.info(f"Class distribution: {counts}")
    return weights.astype(np.float32)

# 数据加载
def load_model_data(use_gcn=False, use_former=False):
    data_list = []
    for model_type, paths in model_paths.items():
        if (model_type == "GCN" and use_gcn) or (model_type == "Former" and use_former):
            for path in paths.values():
                with open(path, 'rb') as f:
                    data = np.array([pickle.load(f)[f"test_{i}"] for i in range(2000)])
                    data_list.append(data)
    X = np.stack(data_list, axis=1).sum(axis=1)
    y = np.load("test_label_A.npy")
    return X, y

# 数据分割
def prepare_datasets(X, y, train_size=0.8):
    return train_test_split(X, y, train_size=train_size, random_state=42)

# 元学习模型
class MetaLearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# 训练函数
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=50):
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        for features, labels in train_loader:
            features, labels = features.float(), labels.long()
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        accuracy = correct / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        log_msg = f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        print(log_msg)
        logging.info(log_msg)
        evaluate(model, test_loader)

# 评估模型
def evaluate(model, data_loader):
    model.eval()
    total_loss, correct = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.float(), labels.long()
            outputs = model(features)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    accuracy = correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    log_msg = f"Eval Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
    print(log_msg)
    logging.info(log_msg)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.weight = alpha, gamma, weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * CE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

if __name__ == "__main__":
    X, y = load_model_data(use_gcn=True, use_former=True)
    X_train, X_test, y_train, y_test = prepare_datasets(X, y)
    
    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    model = MetaLearner(input_dim=X.shape[1], output_dim=155)
    class_weights = torch.tensor(calculate_class_weights(y_train))
    criterion = FocalLoss(alpha=1, gamma=2, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=80)

    torch.save(model.state_dict(), "meta_learner_weights.pth")
    logging.info("模型权重已保存为 meta_learner_weights.pth")
    print("模型权重已保存为 meta_learner_weights.pth")
