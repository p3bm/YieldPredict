import matplotlib.pyplot as plt
import datetime
import argparse
import random
import os

import numpy as np
from sklearn.decomposition import TruncatedSVD

from dataset_container import *
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.manifold import TSNE

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.metrics import adjusted_rand_score

from sklearn.cluster import KMeans, BisectingKMeans, DBSCAN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process.kernels import RBF, Matern


round = 1


class DEC(nn.Module):
    def __init__(
        self,
        ori_cluster_centers: torch.Tensor,
        input_dim = 256,
        hidden_dim = 128, 
        output_dim = 64
    ):
        super(DEC, self).__init__()
        self.ori_cluster_centers = ori_cluster_centers
        self.cluster_centers = None
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def new_center(self, ori_cluster_centers):
        self.ori_cluster_centers = ori_cluster_centers
    
    def update(self):
        self.cluster_centers = self.encoder(self.ori_cluster_centers)
            
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.clusterAssignment(self.encoder(batch))
    
    def clusterAssignment(self, batch: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum((batch - self.cluster_centers) ** 2, 1)
        numerator = 1.0 / (1.0 + (norm_squared))
        return numerator / torch.sum(numerator)

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        weight = (batch ** 2) / torch.sum(batch)
        return (weight.t() / torch.sum(weight)).t()

def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str, choices={"real_1", "real_2", "real_3"},
                        help="Name of the dataset.")
    parser.add_argument("model", type=str, choices={"svm"},
                        help="Name of the model")
    parser.add_argument("active_learning", type=str, choices={"dist_to_boundary", "margin", "qbc", "uniform"},
                        help="Name of the active learning algorithm")

    parser.add_argument("--dataset_path", type=str, default="./datasets/synthetic",
                        help="Path where datasets will be loaded/saved.")

    parser.add_argument("--init_num_labeled", type=float, default=0.05,
                        help="Initial number of labeled data in active learning. Float values less than 1.0 indicate a"
                             " percentage.")
    parser.add_argument("--budget_round", type=float, default=0.05,
                        help="Budget in each active learning round. Float values less than 1.0 indicate a percentage.")
    parser.add_argument("--active_epoch", type=int, default=9,
                        help="Number of rounds to run the active learning algorithm.")

    parser.add_argument("--boundaries", nargs='*',
                        help="The sequence of values of decision boundaries that a model is expected to learn.")

    parser.add_argument("--random_seed", type=int, default=0)

    # arguments for QBC
    parser.add_argument("--num_models", type=int, default=10)
    parser.add_argument("--num_samples", type=float, default=0.8)
    
    # arguments for retrain
    parser.add_argument("--retrain",type=bool,default=0)

    args = parser.parse_args()
    return args

class Arguments:
    def __init__(self):
        self.random_seed = 500

        self.dataset = "real_4"
        self.split_mode = 0
        self.dataset_path = "./datasets/real"

        self.model = "logistic_regression"  # {"logistic_regression", "svm"}

        self.if_hybrid = 1  # {1,0} 1 refers to two types of descriptors 0 otherwise
        self.representationA = "pka_bde01"
        self.representationB = "morgan_fp"  # {"morgan_fp", "one_hot", "morgan_pka","ohe_pka"}

        self.representation = "Mordred"  # {"morgan_fp", "one_hot", "Mordred", "morgan_pka", "ohe_pka"}
        self.representation_dim = 2048  # used for morgan fingerprint or morgan_pka
        self.reduce_dim = 'pca' # {pca, vae}

        # self.representations = ['rdkit', 'morgan_fp']
        self.representations = ['Mordred', 'morgan_fp']
        self.reduce_method = ['pca','pca']
        # self.pca_components = [512*4,512*4]
        self.pca_components = [4096, 4096]

        self.active_learning = "qbc"  # {"dist_to_boundary", "least_confidence", "margin", "qbc", "uniform"}
        self.qbc_strategy = "margin"  # {"dist_to_boundary", "least_confidence", "margin"}
        self.active_epoch = 10
        self.budget_round = 5  # budget in each active learning round, real values indicate a percentage
        self.init_num_labeled = 5  # number of initial labeled data, real values indicate a percentage

        self.num_models = 25
        self.num_samples = 0.7
        
        self.retrain = False
        self.no_label_pred = True
        self.random_test = True

def PCA_reduce(xs, n_components = 72, random_state = 500):
    trunc_svd = TruncatedSVD(n_components = n_components, random_state = random_state)
    xs_reduced = trunc_svd.fit_transform(xs)
    pca_explained_variance_ratio = trunc_svd.explained_variance_ratio_.sum()

    print("xs_reduced.shape: {}".format(xs_reduced.shape))
    print("explained_variance_ratio_: {}".format(pca_explained_variance_ratio))
    return xs_reduced, trunc_svd

def main():
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
        print("train on gpu")
    else:
        device = torch.device("cpu")
        print("train on cpu")

    args = Arguments()
    
    dataset_kwargs = dict()
    dataset_kwargs["pred"] = False
    dataset_kwargs["split_mode"] = args.split_mode

    origx_trains = []
    for i in range(len(args.representations)):
        representation = args.representations[i]
        reduce_method = args.reduce_method[i]
        pca_component = args.pca_components[i]
        print(representation + ',' + reduce_method)
        if representation == "morgan_fp" or representation == "morgan_pka" or representation == "morgan_pka01":
            dataset_kwargs["representation_dim"] = args.representation_dim
        else:
            dataset_kwargs["representation_dim"] = None
        original_dataset = get_dataset(args.dataset, args.dataset_path, representation, **dataset_kwargs)
        (orig_x_train, orig_y_train_unnormalized), (orig_x_test, orig_y_test_unnormalized) = original_dataset

        num_orig_train = orig_x_train.shape[0]
        if reduce_method is not None:
            xs = np.concatenate((orig_x_train, orig_x_test), axis=0)
            xs_reduced = xs.copy()
            print("xs.shape: {}".format(xs.shape))
            if reduce_method == "pca":
                xs_reduced, trunc_svd = PCA_reduce(xs, n_components = pca_component, random_state=args.random_seed)
            orig_x_train = xs_reduced[:num_orig_train, :]
        print(orig_x_train.shape)
        origx_trains.append(orig_x_train)

    full_y = orig_y_train_unnormalized
    
    low_id = []
    for i in range(len(orig_x_train)):
        low_id.append(i)
    
    low_id = np.array(low_id)
    all_low0 = origx_trains[0][low_id, ...]
    all_low1 = origx_trains[1][low_id, ...]
    

    pred_array = dict()
    # encode_array = dict()
    def double_cluster(dec0, dec1, data0, data1, start_id, step_size, n_clusters = 30, epoch_size = 20, dir_name = 'figs', const_epoch = False, al_method = 'prob', lam=1):
        def monitor_loss(loss_list, threshold = 0.015):
            if len(loss_list) < 120:
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
                        fn+=1
                    elif test_y[i] < stand and result_y[i] > stand:
                        fp+=1
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
                f1sc = 2 * precision * recall / (precision + recall)
                y.append(f1sc)
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

        def get_full_cluster(data, dec, device = torch.device("cpu")):
            full_x = []
            for i in range(len(data)):
                d = data[i]
                q = dec(d).detach().cpu().numpy()
                full_x.append(q)
            return np.array(full_x)

        def get_full_encode(data, dec, device = torch.device("cpu")):
            full_x = []
            for i in range(len(data)):
                d = data[i]
                q = dec.encoder(d).detach().cpu().numpy()
                full_x.append(q)
            return np.array(full_x)
        
        def test_represention(data, full_y, special_id =False, dec = None, test_id = None, train_id = None, cluster = True, test_size = 0.95, model = 'RF', device = torch.device("cpu")):
            if dec == None:
                full_x = data.cpu()
            elif cluster:
                dec.update()
                full_x = get_full_cluster(data, dec, device)
            else:
                full_x = get_full_encode(data, dec, device)

            if not special_id:
                train_x, test_x, train_y, test_y = train_test_split(full_x, full_y, test_size = test_size, shuffle = False)
            else:
                train_x = full_x[train_id]
                test_x = full_x[test_id]
                train_y = full_y[train_id]
                test_y = full_y[test_id]

            if model == 'RF':
                return RF_Predict(train_x, test_x, train_y, test_y)
        
        print(len(start_id))

        train_idx = start_id
        test_idx = [i for i in range(len(orig_y_train_unnormalized)) if i not in start_id]
        
        train_id = np.array(start_id)
        test_id = np.array(test_idx)
        labeled_y = full_y[train_id].reshape(-1,1)

        def get_center(n_clusters):
            while True:
                kmeans = KMeans(n_clusters)
                kmeans.fit(labeled_y)
                cluster_centers = kmeans.cluster_centers_
                cluster_labels = kmeans.labels_
                
                closest_points, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, labeled_y)
                closest_points = train_id[closest_points]

                max_dis = []
                for j in range(n_clusters):
                    cluster_0_indices = [i for i, label in enumerate(cluster_labels) if label == j]
                    cluster_0_data = labeled_y[cluster_0_indices]
                    max_dis.append(max(cluster_0_data) - min(cluster_0_data))
                judge_dis0 = max(max_dis)
                judge_dis1 = min(max_dis)
                print(judge_dis0, judge_dis1, n_clusters)
                if judge_dis0 > 5 or judge_dis1 > 1:
                    n_clusters += 2
                elif judge_dis0 < 3:
                    n_clusters -= 2
                else:
                    break
            return closest_points, judge_dis0 + judge_dis1, n_clusters, cluster_centers

        closest_points, judge_dis, n_clusters, cluster_centers = get_center(n_clusters)
        
        def get_const_q():
            const_q = {}
            for id in train_id:
                level = []
                for center in closest_points:
                    y = full_y [id]
                    c_y = full_y[center]
                    # if abs(c_y-y) < 1 * judge_dis:
                    #     level.append(2)
                    # elif abs(c_y-y) < 2 * judge_dis:
                    #     level.append(1)
                    # elif abs(c_y-y) < 3 * judge_dis:
                    #     level.append(0.1)
                    # else:
                    #     level.append(0.0001)
                    w = max(2 - abs(y-c_y)**2 / 100, 0.0001)
                    level.append(w)
                level = torch.tensor(level, dtype = torch.double).to(device)
                level = level/torch.sum(level)
                const_q[id] = level
            return const_q
        
        const_q = get_const_q()

        cluster_centers0 = torch.tensor(all_low0[closest_points], dtype=torch.double, requires_grad=True).to(device)
        cluster_centers1 = torch.tensor(all_low1[closest_points], dtype=torch.double, requires_grad=True).to(device)
        
        dec0.new_center(cluster_centers0)
        dec1.new_center(cluster_centers1)
        
        optimizer0 = torch.optim.Adam(dec0.parameters(), lr=0.001)
        optimizer1 = torch.optim.Adam(dec1.parameters(), lr=0.001)
        
        loss_function = nn.KLDivLoss(reduction = 'sum')
        loss_list1 = []
        loss_list0 = []
        for epoch in range(epoch_size):
            dec0.update()
            dec1.update()

            total_loss0 = 0
            total_loss1 = 0
            for i in range(len(data0)):
                d0 = data0[i]
                d1 = data1[i]
                q0 = dec0(d0)
                q1 = dec1(d1)
                loss_local0 = 0
                loss_local1 = 0
                if i in train_id:
                    loss_local0 = loss_function(q0.log(), const_q[i])
                    loss_local1 = loss_function(q1.log(), const_q[i])
                loss01 = loss_function(q0.log(), q1.detach())
                loss10 = loss_function(q1.log(), q0.detach())
                total_loss0 +=  1 * loss_local0 + lam * loss01
                total_loss1 +=  1 * loss_local1 + lam * loss10
            
            optimizer0.zero_grad()
            total_loss0.backward()
            optimizer0.step()
            optimizer1.zero_grad()
            total_loss1.backward()
            optimizer1.step()

            print(total_loss0,total_loss1,epoch)
            loss_list0.append(total_loss0.item())
            loss_list1.append(total_loss1.item())

            if not const_epoch and ((monitor_loss(loss_list0) and monitor_loss(loss_list1))):
                # epoch_size = epoch
                break

        def save_test():
            orig_0_r2, orig_0_mae, orig_0_f1, orig_0_rmse, orig_0_maes, y_orig_0 = test_represention(data = data0, dec = None, full_y = full_y, train_id = train_id, test_id = test_id, cluster = True, special_id = True, device = device)
            orig_1_r2, orig_1_mae, orig_1_f1, orig_1_rmse, orig_1_maes, y_orig_1 = test_represention(data = data1, dec = None, full_y = full_y, train_id = train_id, test_id = test_id, cluster = True, special_id = True, device = device)
            encode_0_r2, encode_0_mae, encode_0_f1, encode_0_rmse, encode_0_maes, y_encode_0 = test_represention(data = data0, dec = dec0, full_y = full_y, train_id = train_id,
                            test_id = test_id, cluster = False, special_id = True, device = device)
            encode_1_r2, encode_1_mae, encode_1_f1, encode_1_rmse, encode_1_maes, y_encode_1 = test_represention(data = data1, dec = dec1, full_y = full_y, train_id = train_id,
                            test_id = test_id, cluster = False, special_id = True, device = device)
            cluster_0_r2, cluster_0_mae, cluster_0_f1, cluster_0_rmse, cluster_0_maes, y_cluster_0 = test_represention(data = data0, dec = dec0, full_y = full_y, train_id = train_id,
                            test_id = test_id, cluster = True, special_id = True, device = device)
            cluster_1_r2, cluster_1_mae, cluster_1_f1, cluster_1_rmse, cluster_1_maes, y_cluster_1 = test_represention(data = data1, dec = dec1, full_y = full_y, train_id = train_id,
                            test_id = test_id, cluster = True, special_id = True, device = device)
            print(orig_0_r2, orig_0_mae, '\n',
                  orig_1_r2, orig_1_mae, '\n',
                  cluster_0_r2, cluster_0_mae, '\n',
                  cluster_1_r2, cluster_1_mae, '\n',
                  encode_0_r2, encode_0_mae, '\n',
                  encode_1_r2, encode_1_mae, '\n',
                  )

            with open("./results/reborn.txt","a+") as f:
                f.write(str(n_clusters) + ',')
                f.write(str(len(start_id)) + ',')

                f.write(str(encode_0_r2) + ',')
                f.write(str(encode_0_mae) + ',')
                f.write(str(encode_0_rmse) + ',')

                f.write(str(encode_1_r2) + ',')
                f.write(str(encode_1_mae) + ',')
                f.write(str(encode_1_rmse) + ',')

                f.write(str(epoch) + ',')
                f.write(al_method + '\n')

            global round
            round += 1

        save_test()

        def prob_cover_mix():
            add_id = []
            X_encoder0 = get_full_encode(data0, dec0)
            X_encoder1 = get_full_encode(data1, dec1)
            X_cluster0 = get_full_cluster(data0, dec0)
            X_cluster1 = get_full_cluster(data1, dec1)
            X_encoder = np.concatenate((X_encoder0, X_encoder1), axis=1)
            X_cluster = X_cluster0 + X_cluster1

            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X_encoder)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            distances = pairwise_distances(X_encoder, centroids, metric='euclidean')
            avg_distances = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_points = X_encoder[labels == i]
                cluster_distances = distances[labels == i, i]
                avg_distances[i] = np.mean(cluster_distances)
            judge_dis = np.mean(avg_distances)
            print(judge_dis)

            coverd_set = set([])
            for train_id in train_idx:
                for j in range(len(X_encoder)):
                    X = X_encoder[train_id]
                    Xx = X_encoder[j]
                    l2_distance = np.linalg.norm(X - Xx)
                    if l2_distance < judge_dis:
                        coverd_set.add(j)

            degree = [0 for i in range(len(X_encoder))]
            for test_id in test_idx:
                cluster0 = X_cluster0[i]
                cluster1 = X_cluster1[i]
                entropy0 = -sum(cluster0 * np.log(cluster0))
                entropy1 = -sum(cluster1 * np.log(cluster1))
                degree[test_id] = entropy0 + entropy1
                for j in range(len(X_encoder)):
                    if j not in coverd_set:
                        X = X_encoder[test_id]
                        Xx = X_encoder[j]
                        l2_distance = np.linalg.norm(X - Xx)
                        if l2_distance < judge_dis:
                            degree[test_id] += 1
            
            for idx in range(step_size):
                query_idx = degree.index(max(degree))
                add_id.append(query_idx)
                new_covered = set([])
                for j in range(len(X_encoder)):
                    if j not in coverd_set:
                        X = X_encoder[query_idx]
                        Xx = X_encoder[j]
                        l2_distance = np.linalg.norm(X - Xx)
                        if l2_distance < judge_dis:
                            new_covered.add(j)
                            coverd_set.add(j)
                for test_id in test_idx:
                    for newc in new_covered:
                        X = X_encoder[test_id]
                        Xx = X_encoder[newc]
                        l2_distance = np.linalg.norm(X - Xx)
                        if l2_distance < judge_dis:
                            degree[test_id] -= 1
                degree[query_idx] = 0

            print(add_id)
            return add_id

        def prob_cover():
            add_id = []
            X_encoder0 = get_full_encode(data0, dec0)
            X_encoder1 = get_full_encode(data1, dec1)
            X_cluster0 = get_full_cluster(data0, dec0)
            X_cluster1 = get_full_cluster(data1, dec1)
            X_encoder = np.concatenate((X_encoder0, X_encoder1), axis=1)
            X_cluster = X_cluster0 + X_cluster1

            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X_encoder)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            distances = pairwise_distances(X_encoder, centroids, metric='euclidean')
            avg_distances = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_points = X_encoder[labels == i]
                cluster_distances = distances[labels == i, i]
                avg_distances[i] = np.mean(cluster_distances)
            judge_dis = np.mean(avg_distances)
            print(judge_dis)

            coverd_set = set([])
            for train_id in train_idx:
                for j in range(len(X_encoder)):
                    X = X_encoder[train_id]
                    Xx = X_encoder[j]
                    l2_distance = np.linalg.norm(X - Xx)
                    if l2_distance < judge_dis:
                        coverd_set.add(j)

            degree = [0 for i in range(len(X_encoder))]
            for test_id in test_idx:
                for j in range(len(X_encoder)):
                    if j not in coverd_set:
                        X = X_encoder[test_id]
                        Xx = X_encoder[j]
                        l2_distance = np.linalg.norm(X - Xx)
                        if l2_distance < judge_dis:
                            degree[test_id] += 1
            
            for idx in range(step_size):
                query_idx = degree.index(max(degree))
                add_id.append(query_idx)
                new_covered = set([])
                for j in range(len(X_encoder)):
                    if j not in coverd_set:
                        X = X_encoder[query_idx]
                        Xx = X_encoder[j]
                        l2_distance = np.linalg.norm(X - Xx)
                        if l2_distance < judge_dis:
                            new_covered.add(j)
                            coverd_set.add(j)
                for test_id in test_idx:
                    for newc in new_covered:
                        X = X_encoder[test_id]
                        Xx = X_encoder[newc]
                        l2_distance = np.linalg.norm(X - Xx)
                        if l2_distance < judge_dis:
                            degree[test_id] -= 1
                degree[query_idx] = 0

            print(add_id)
            return add_id

        def entropy():
            X_cluster0 = get_full_cluster(data0, dec0)
            X_cluster1 = get_full_cluster(data1, dec1)
            possible_id = []
            for i in range(len(X_cluster0)):
                if i not in start_id:
                    cluster0 = X_cluster0[i]
                    cluster1 = X_cluster1[i]
                    entropy0 = -sum(cluster0 * np.log(cluster0))
                    entropy1 = -sum(cluster1 * np.log(cluster1))
                    entropyf = entropy0 + entropy1
                    possible_id.append((entropyf, i))
            possible_id.sort(key=lambda x:x[0], reverse=True)
            add_id = [x[1] for x in possible_id[:step_size]]
            print(add_id)
            return add_id

        if al_method =='prob' or al_method == '20prob':
            add_id = prob_cover()
        elif al_method == 'mix':
            add_id = prob_cover_mix()
        elif al_method == 'entropy':
            add_id = entropy()

        return add_id
    
    def activate_learning(active_method='prob'):
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

        data0 = torch.tensor(all_low0, dtype=torch.double).to(device)
        data1 = torch.tensor(all_low1, dtype=torch.double).to(device)
        dec0 = DEC(ori_cluster_centers = None,input_dim = data0.shape[1], hidden_dim = 2048, output_dim = 1024)
        dec0.to(torch.double)
        dec0.to(device)
        dec1 = DEC(ori_cluster_centers = None,input_dim = data1.shape[1], hidden_dim = 2048, output_dim = 1024)
        dec1.to(torch.double)
        dec1.to(device)

        lam = 1
        train_id = get_split_id(len(orig_y_train_unnormalized), 0.95, 99)
        total_size = 199
        
        step_size = 25
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        dir_name = f"{timestamp}"
        while len(train_id) < total_size:
           add_id = double_cluster(dec0, dec1, data0, data1, train_id, step_size, 24, 300, dir_name, al_method=active_method, lam=lam)
           train_id = train_id + add_id
        double_cluster(dec0, dec1, data0, data1, train_id, step_size, 30, 300, dir_name, al_method=active_method, lam=lam)

    active_method = 'prob'
    activate_learning(active_method=active_method)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    with open("./results/reborn.txt","a+") as f:
        f.write("start")
    main()
