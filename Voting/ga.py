
# 模型路径
model_paths = {
    "gcn_3d": {
         "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
    "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
    "former_b_3d_resample_rotate": "../scores/Mix_Former/mixformer_B_3d_resample_rotate.pkl",
    "skateformer_j_3d": "../scores/Mix_Former/skateformer_B_3d.pkl",
    },

    former_names : {
    "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
    "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
    "former_b_3d_resample_rotate": "../scores/Mix_Former/mixformer_B_3d_resample_rotate.pkl",
    "skateformer_j_3d": "../scores/Mix_Former/skateformer_B_3d.pkl",
}
}

# 数据加载
def load_data(**model_flags):
    data = []
    for model_type, is_enabled in model_flags.items():
        if is_enabled:
            for path in model_paths[model_type].values():
                with open(path, 'rb') as f:
                    data_dict = pickle.load(f)
                data.append(np.array([data_dict[f"test_{i}"] for i in range(2000)]))
    X = np.transpose(data, (1, 0, 2))
    y = np.load("test_label_A.npy")
    return data, X, y

# Softmax处理
def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=0, keepdims=True))
    return exp_X / exp_X.sum(axis=0, keepdims=True)

# 加权预测
def weighted_prediction(X, weights):
    predictions = [
        np.argmax(np.sum(w * softmax(model) for w, model in zip(weights, sample)))
        for sample in X
    ]
    return np.array(predictions)

# 损失函数
def loss(weights, X, y):
    predictions = weighted_prediction(X, weights)
    return -np.mean(predictions == y)

# 遗传算法优化
def optimize_weights_ga(X, y, n_generations=25, population_size=50):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", partial(lambda ind, X, y: (-loss(ind, X, y),), X=X, y=y))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        population = toolbox.population(n=population_size)
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=True)

    return tools.selBest(population, k=1)[0]

# 局部模型保存
def save_partial_results(data, weights, model_type):
    weighted_data = np.sum([d * w for d, w in zip(data, weights)], axis=0)
    result_dict = {f"test_{i}": sample for i, sample in enumerate(weighted_data)}
    with open(f"./partial/partial_{model_type}.pkl", "wb") as f:
        pickle.dump(result_dict, f)

# 主程序
if __name__ == "__main__":
    model_flags = {"gcn_2d": False, "gcn_3d": True, "former_2d": True, "former_3d": False}
    data_list, X, y = load_data(**model_flags)
    
    start_time = time.time()
    optimized_weights = optimize_weights_ga(X, y)
    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")

    accuracy = np.mean(weighted_prediction(X, optimized_weights) == y) * 100
    print(f"Accuracy with Optimized Weights (GA): {accuracy}%")
    
    # 保存局部集成结果
    save_partial_results(data_list, optimized_weights, model_type="gcn_3d")
