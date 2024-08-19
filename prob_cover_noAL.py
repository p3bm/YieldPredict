import matplotlib.pyplot as plt
import datetime
import random
import numpy as np
from sklearn.decomposition import TruncatedSVD
from myutil import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,adjusted_rand_score
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances

from sklearn.cluster import KMeans, BisectingKMeans, DBSCAN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process.kernels import RBF, Matern

import scipy.stats as stats

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

class Arguments:
    def __init__(self):
        self.dataset = "real_4"        #dataset name of your reaction space
        self.split_mode = 0
        self.dataset_path = "./datasets/"

        self.representations = ['Mordred', 'morgan_fp']    #Molecular descriptors you want to use
        self.reduce_method = ['pca','pca']    #Arguments for PCA
        self.pca_components = [4096,4096]    

def PCA_reduce(xs, n_components = 72):
    trunc_svd = TruncatedSVD(n_components = n_components)
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
    dataset_kwargs["split_mode"] = args.split_mode
    
    origx_trains = []
    for i in range(len(args.representations)):
        representation = args.representations[i]
        reduce_method = args.reduce_method[i]
        pca_component = args.pca_components[i]
        print(representation + ',' + reduce_method)
        if representation == "morgan_fp" or representation == "morgan_pka" or representation == "morgan_pka01":
            dataset_kwargs["representation_dim"] = 2048
        else:
            dataset_kwargs["representation_dim"] = None
        original_dataset = get_dataset(args.dataset, args.dataset_path, representation, **dataset_kwargs)
        (orig_x_train, orig_y_train_unnormalized), (orig_x_test, orig_y_test_unnormalized) = original_dataset

        num_orig_train = orig_x_train.shape[0]
        if reduce_method is not None:
            xs = orig_x_train
            xs_reduced = xs.copy()
            print("xs.shape: {}".format(xs.shape))
            if reduce_method == "pca":
                xs_reduced, trunc_svd = PCA_reduce(xs, n_components = pca_component)
            orig_x_train = xs_reduced[:num_orig_train, :]
        print(orig_x_train.shape)
        origx_trains.append(orig_x_train)

    full_y = orig_y_train_unnormalized
    all_low0 = origx_trains[0]
    all_low1 = origx_trains[1]
    
    def double_cluster(dec0, dec1, data0, data1, start_id, step_size, n_clusters = 30, al_method = 'prob'):        
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
                total_loss0 +=  loss_local0 + loss01
                total_loss1 +=  loss_local1 + loss10
            
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
                epoch_size = epoch
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
            with open("./results/r2_noAL.csv","a+") as f:
                f.write(str(n_clusters) + ',')
                f.write(str(len(start_id)) + ',')
                f.write(str(orig_0_r2) + ',')
                f.write(str(orig_1_r2) + ',')
                f.write(str(cluster_0_r2) + ',')
                f.write(str(cluster_1_r2) + ',')
                f.write(str(encode_0_r2) + ',')
                f.write(str(encode_1_r2) + ',')
                f.write(str(epoch_size) + ',')
                f.write(al_method + '\n')

        save_test()
        return None
    
    data0 = torch.tensor(all_low0, dtype=torch.double).to(device)
    data1 = torch.tensor(all_low1, dtype=torch.double).to(device)
    dec0 = DEC(ori_cluster_centers = None,input_dim = data0.shape[1], hidden_dim = 2048, output_dim = 1024)
    dec0.to(torch.double)
    dec0.to(device)
    dec1 = DEC(ori_cluster_centers = None,input_dim = data1.shape[1], hidden_dim = 2048, output_dim = 1024)
    dec1.to(torch.double)
    dec1.to(device)
    train_id = []
    for i in range(len(full_y)):
        if full_y[i] > 0:
            train_id.append(i)
    step_size = 25
    recommend_reactions = double_cluster(dec0, dec1,data0, data1, train_id, step_size, 30, 100, dir_name)

    print(recommend_reactions)

if __name__ == "__main__":
    main()
