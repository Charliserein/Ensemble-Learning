import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt

gcn_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
    #"ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
    #"ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
    #"ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
    #"ctrgcn_j_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample_rotate.pkl",
    #"ctrgcn_b_2d": "../scores/Mix_GCN/ctrgcn_V1_B_2d.pkl",
    #"ctrgcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_J_2d.pkl",
    #"ctrgcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
    #"ctrgcn_jm_2d": "../scores/Mix_GCN/ctrgcn_V1_JM_2d.pkl",
    #"tdgcn_j_2d": "../scores/Mix_GCN/tdgcn_V1_J_2d.pkl",
    #"blockgcn_j_3d": "../scores/Mix_GCN/blockgcn_J_3d.pkl",
    #"blockgcn_jm_3d": "../scores/Mix_GCN/blockgcn_JM_3d.pkl",
    #"blockgcn_b_3d": "../scores/Mix_GCN/blockgcn_B_3d.pkl",
    #"blockgcn_bm_3d": "../scores/Mix_GCN/blockgcn_BM_3d.pkl",
    #"ctrgcn_b_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_B_3d_resample_rotate.pkl",
    #"degcn_J_3d": "../scores/Mix_GCN/degcn_J_3d.pkl",
    #"degcn_B_3d": "../scores/Mix_GCN/degcn_B_3d.pkl",
    #"tegcn_V1_J_3d": "../scores/Mix_GCN/tegcn_V1_J_3d.pkl"
}

former_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    #"former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    #"former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    #"former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    #"former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    #"former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
    #"former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
    #"former_b_3d_resample_rotate": "../scores/Mix_Former/mixformer_B_3d_resample_rotate.pkl",
    #"skateformer_j_3d": "../scores/Mix_Former/skateformer_B_3d.pkl",
}


def load_data(gcn: bool = False, former: bool = False):
    data_list = []
    if gcn:
        for name in gcn_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if former:
        for name in former_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    
    data_np = np.array(data_list)
    
    X = data_np.transpose(1, 0, 2) 
    y = np.load("test_label_A.npy")  

    return X, y

if __name__ == "__main__":

    X, y = load_data(gcn=True, former=True)

  
    X_train = X.reshape(X.shape[0], -1)  # 将 (samples, models, features) 展平为 (samples, models * features)
    y_train = y

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("随机森林拟合完成。");
    print("开始运行解释器...");

    explainer = shap.TreeExplainer(model)

    print("解释器运行完成, 开始计算SHAP值...");

    shap_values = explainer.shap_values(X_train);
    print("计算完毕, 正在保存图片...");

  
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.savefig("shap_summary_plot.png")  

    print("保存成功。");