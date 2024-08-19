from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,adjusted_rand_score
import os
import torch
import random


def save_tsne_figure(X,Y,name):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    X_embedded = tsne.fit_transform(X)
    min_x, max_x = min(X_embedded[:, 0]), max(X_embedded[:, 0])
    min_y, max_y = min(X_embedded[:, 1]), max(X_embedded[:, 1])
    print(min_x, max_x, min_y, max_y)
    plt.figure()
    plt.xlim(min_x-1, max_x+1)
    plt.ylim(min_y-1, max_y+1)
    
    cmap = plt.get_cmap('viridis')
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y, cmap=cmap, s=1)
    plt.colorbar()

    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(name + '.png')
    
def get_full_cluster(data, dec):
    full_x = []
    for i in range(len(data)):
        d = data[i]
        q = dec(d).detach().cpu().numpy()
        full_x.append(q)
    return np.array(full_x)

def get_full_encode(data, dec):
    full_x = []
    for i in range(len(data)):
        d = data[i]
        q = dec.encoder(d).detach().cpu().numpy()
        full_x.append(q)
    return np.array(full_x)

def monitor_loss(loss_list, threshold = 0.015):
    if len(loss_list) < 21:
        return False
    
    average_loss = sum(loss_list[-10:]) / len(loss_list[-10:])
    previous_average_loss = sum(loss_list[-20:-10]) / len(loss_list[-20:-10])
    
    if abs(average_loss - previous_average_loss) < average_loss * threshold:
        return True
    
    return False

def calcMaes(test_y,result_y):
    ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    sorted_indices = np.argsort(test_y)[::-1]
    y_real_sorted = test_y[sorted_indices]
    y_pred_sorted = result_y[sorted_indices]
    maes = []
    for ratio in ratios:
        k = int(len(test_y) * ratio)
        y_real_top_k = y_real_sorted[:k]
        y_pred_top_k = y_pred_sorted[:k]
        abs_diff = np.abs(y_real_top_k - y_pred_top_k)
        mae = np.mean(abs_diff)
        maes.append(mae)
    return maes

def calcF1(test_y,result_y):
    ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    y = []
    for ratio in ratios:
        sorted_y = sorted(test_y)
        stand = sorted_y[int(ratio * len(test_y))]
        tp,tn,fn,fp = 0,0,0,0
        for i in range(len(test_y)):
            if test_y[i] > stand and result_y[i] > stand:
                tp+=1
            elif test_y[i] < stand and result_y[i] < stand:
                tn+=1
            elif test_y[i] > stand and result_y[i] < stand:
                fp+=1
            elif test_y[i] < stand and result_y[i] > stand:
                fn+=1
        y.append([tp,fp,fn,fp])
    return y

def RF_Predict(train_x, test_x, train_y, test_y):
    regression_model = RandomForestRegressor()
    regression_model.fit(train_x, train_y)
    regression_result = regression_model.predict(test_x)
    regression_r2 = r2_score(test_y,regression_result)
    regression_mae = mean_absolute_error(test_y, regression_result)
    regression_f1 = calcF1(test_y, regression_result)
    regression_rmse = mean_squared_error(test_y, regression_result) ** 0.5
    regression_maes = calcMaes(test_y,regression_result)
    return regression_r2, regression_mae, regression_f1, regression_rmse, regression_maes, regression_result

def test_represention(data, full_y, special_id =False, dec = None, test_id = None, train_id = None, cluster = True, test_size = 0.95, model = 'RF', device = torch.device("cpu")):
    if dec == None:
        full_x = data.cpu()
    elif cluster:
        dec.update()
        full_x = get_full_cluster(data, dec)
    else:
        full_x = get_full_encode(data, dec)

    if not special_id:
        train_x, test_x, train_y, test_y = train_test_split(full_x, full_y, test_size = test_size, shuffle = False)
    else:
        train_x = full_x[train_id]
        test_x = full_x[test_id]
        train_y = full_y[train_id]
        test_y = full_y[test_id]

    if model == 'RF':
        return RF_Predict(train_x, test_x, train_y, test_y)
    
def get_dataset(dataset, dataset_dir, representation=None, representation_dim=None, split_mode=None):
    if representation is not None:
        dataset_file_predix = os.path.join(dataset_dir, dataset)
        if split_mode is not None:
            dataset_file_predix = os.path.join(dataset_file_predix, "split_" + str(split_mode))
        dataset_file_predix = os.path.join(dataset_file_predix, dataset + "_" + representation,
                                        dataset + "_" + representation if representation_dim is None else
                                        dataset + "_" + representation + "_" + str(representation_dim))
    data_train = np.load(dataset_file_predix + "_train.npz")
    data_test = np.load(dataset_file_predix + "_test.npz")
    x_train, y_train = data_train["train_data"], data_train["train_labels"]
    x_test, y_test = data_test["test_data"], data_test["test_labels"]
    return (x_train, y_train), (x_test, y_test)

def get_split_id(N, test_ratio = 0.95, test_num = 0):
    if test_num == 0:
        random_numbers = random.sample(range(N), int(test_ratio * N))
    else:
        random_numbers = random.sample(range(N), N - test_num)
    train_id = []
    test_id =[]
    for i in range(N):
        if i in random_numbers:
            test_id.append(i)
        else:
            train_id.append(i)
    return train_id

def save_tsne_hex(X,Y,name):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure()
    cmap = plt.get_cmap('viridis')
    plt.hexbin(X_embedded[:, 0], X_embedded[:, 1], C=Y, gridsize=50, cmap=cmap)
    plt.colorbar()

    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(name + '.png')
    

def save_both_tsne(X, Y, name, noise=False):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    X_embedded = tsne.fit_transform(X)
    if noise == True:
        noise = np.random.uniform(-0.5, 0.5, len(X_embedded))
        X_embedded[:, 0] += noise
        noise = np.random.uniform(-0.5, 0.5, len(X_embedded))
        X_embedded[:, 1] += noise
    
    min_x, max_x = min(X_embedded[:, 0]), max(X_embedded[:, 0])
    min_y, max_y = min(X_embedded[:, 1]), max(X_embedded[:, 1])
    print(min_x, max_x, min_y, max_y)
    plt.figure()
    plt.xlim(min_x-1, max_x+1)
    plt.ylim(min_y-1, max_y+1)
    
    cmap = plt.get_cmap('viridis')
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y, cmap=cmap, s=1)
    plt.colorbar()

    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(name + '.png')
    
    plt.figure()
    cmap = plt.get_cmap('viridis')
    plt.hexbin(X_embedded[:, 0], X_embedded[:, 1], C=Y, gridsize=50, cmap=cmap)
    plt.colorbar()

    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(name + '_h.png')