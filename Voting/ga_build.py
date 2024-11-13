import pickle
import numpy as np
from deap import base, creator, tools, algorithms
import random
import multiprocessing
from functools import partial
import time

# 定义模型路径
partial_model_paths = {
    "former_2d": "./partial/partial_former_2d.pkl",
    "former_3d": "./partial/partial_former_3d.pkl",
    "gcn_2d": "./partial/partial_gcn_2d.pkl",
    "gcn_3d": "./partial/partial_gcn_3d.pkl"
}

# 加载数据
def load_data(former_2d=False, former_3d=False, gcn_2d=False, gcn_3d=False):
    selected_models = {
        "former_2d": former_2d,
        "former_3d": former_3d,
        "gcn_2d": gcn_2d,
        "gcn_3d": gcn_3d
    }
    
    data_list = []
    for model_name, load_flag in selected_models.items():
        if load_flag:
            with open(partial_model_paths[model_name], 'rb') as f:
                model_data = pickle.load(f)
            data_list.append(np.array([model_data[f"test_{i}"] for i in range(2000)]))
    
    X = np.transpose(data_list, (1, 0, 2))  # X shape: (samples, models, features)
    y = np.load("test_label_A.npy")
    return X, y

# Softmax处理
def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=0, keepdims=True))
    return exp_X / exp_X.sum(axis=0, keepdims=True)

# 加权预测
def weighted_prediction(X, weights):
    predictions = [
        np.argmax(sum(w * softmax(model) for w, model in zip(weights, sample)))
        for sample in X
    ]
    return np.array(predictions)

# 损失函数
def loss_function(weights, X, y):
    predictions = weighted_prediction(X, weights)
    return -np.mean(predictions == y)

# 遗传算法优化
def optimize_weights_ga(X, y, n_generations=30, population_size=50):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    evaluate = partial(lambda ind, X, y: (-loss_function(ind, X, y),), X=X, y=y)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        population = toolbox.population(n=population_size)
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=True)

    return tools.selBest(population, k=1)[0]

# 主程序
if __name__ == "__main__":
    X, y = load_data(gcn_2d=True, gcn_3d=True, former_2d=True, former_3d=True)
    
    start_time = time.time()
    optimized_weights = optimize_weights_ga(X, y, n_generations=30)
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Optimized Weights: {optimized_weights}")

    # 使用优化权重进行预测
    predictions = weighted_prediction(X, optimized_weights)
    accuracy = np.mean(predictions == y) * 100
    print(f"Accuracy with Optimized Weights: {accuracy}%")
