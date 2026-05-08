from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import torch
import os
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler, Data
from utils.load_configs import get_link_prediction_args
import pickle
from collections import defaultdict, deque
import torch.nn.functional as F
import gc
from scipy.sparse import lil_matrix, vstack
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.stats import percentileofscore

class SimpleCounterfactualLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=0.1):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions, labels, z_pos, z_cf, z_neg):
        pred_loss = self.bce_loss(predictions, labels)
        
        pos_cf_sim = F.cosine_similarity(z_pos, z_cf, dim=-1) / self.temperature
        pos_neg_sim = F.cosine_similarity(z_pos, z_neg, dim=-1) / self.temperature
        
        logits = torch.stack([pos_cf_sim, pos_neg_sim], dim=1)
        targets = torch.zeros(z_pos.size(0), dtype=torch.long, device=z_pos.device)
        
        contrastive_loss = F.cross_entropy(logits, targets)
        
        total_loss = self.alpha * pred_loss + (1 - self.alpha) * contrastive_loss
        
        return total_loss, pred_loss, contrastive_loss

class JointLoss(nn.Module):
    def __init__(self, alpha=0.8, temp=0.3):
        super().__init__()
        self.alpha = alpha
        self.temp = temp
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def contrastive_loss(self, z_cf, z_pos, z_neg):
        sim_pos = torch.cosine_similarity(z_pos, z_cf, dim=-1) / self.temp
        sim_neg = torch.cosine_similarity(z_pos, z_neg, dim=-1) / self.temp
        
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg.unsqueeze(1)], dim=1)
        labels = torch.zeros(z_cf.size(0), dtype=torch.long).to(z_cf.device)
        return self.ce(logits, labels)
    
    def forward(self, y_pred_fact, y_true, z_cf, z_pos, z_neg):
        loss_fact = self.bce(y_pred_fact, y_true)
        
        loss_contrast = self.contrastive_loss(z_cf, z_pos, z_neg)
        
        total_loss = self.alpha * loss_fact + (1-self.alpha) * loss_contrast
        return total_loss, loss_fact, loss_contrast

class EnhancedNodeEncoder():
    def __init__(self, data: Data, node_features=None, delta=0, theta_percentile=20, 
                 phi_type='cumulative', lambda_=0.1, max_hop=3):
        super().__init__()
        
        self.data=data
        self.time_records = defaultdict(list)
        self.degree_history = defaultdict(int)

        max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
        self.num_nodes = max_node_id + 1
        
        self.idx2node = list(range(self.num_nodes))
        self.node2idx = {node: idx for idx, node in enumerate(self.idx2node)}
        
        self.max_hop = max_hop
        
        self.delta = delta
        self.theta_percentile = theta_percentile
        self.phi_type = phi_type
        self.lambda_ = lambda_
        
        self.theta_cache = defaultdict(float)

        self.has_features = node_features is not None
        if self.has_features:
            self.node_features = node_features
        else:
            self.node_features = None
            torch.manual_seed(42)
            self.feature_dim = min(128, self.num_nodes)
            self._random_features = torch.randn(self.num_nodes, self.feature_dim) * 0.1

        self.nodes_neighbor_ids = [[] for _ in range(self.num_nodes)]
        self.nodes_neighbor_times = [[] for _ in range(self.num_nodes)]
        
        for src, dst, t in zip(data.src_node_ids, data.dst_node_ids, data.node_interact_times):
            self.nodes_neighbor_ids[src].append(dst)
            self.nodes_neighbor_ids[dst].append(src)
            self.nodes_neighbor_times[src].append(t)
            self.nodes_neighbor_times[dst].append(t)
    
    def get_k_hop_neighbors(self, node_id, current_time, k_max=None):
        if k_max is None:
            k_max = self.max_hop
            
        k_hop_neighbors = defaultdict(set)
        visited = set()
        queue = deque([(node_id, 0)])
        visited.add(node_id)
        
        while queue:
            current_node, current_hop = queue.popleft()
            
            if current_hop >= k_max:
                continue
                
            neighbors = self.get_all_first_hop_neighbors(current_node, current_time)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    k_hop_neighbors[current_hop + 1].add(neighbor)
                    queue.append((neighbor, current_hop + 1))
        
        result = {}
        for k, neighbors in k_hop_neighbors.items():
            result[k] = list(neighbors)
            
        return result
    
    def _get_node_feature(self, node_id):
        if self.has_features:
            return self.node_features[node_id]
        else:
            return self._random_features[node_id]
        
    def get_all_first_hop_neighbors(self, node_id, current_time=None):
        if current_time is None:
            return self.nodes_neighbor_ids[node_id]
            
        temporal_neighbors = []
        for neighbor, time in zip(self.nodes_neighbor_ids[node_id], self.nodes_neighbor_times[node_id]):
            if time < current_time:
                temporal_neighbors.append(neighbor)
        return temporal_neighbors
    
    def get_node_embedding(self, node_id, current_time):
        self_feature = self._get_node_feature(node_id)
        
        neighbors = self.get_all_first_hop_neighbors(node_id, current_time)
        
        if not neighbors:
            return self_feature
        
        if len(neighbors) > 50:
            neighbors = np.random.choice(neighbors, 50, replace=False).tolist()
        
        neighbor_features = torch.stack([self._get_node_feature(n) for n in neighbors])
        neighbor_mean = torch.mean(neighbor_features, dim=0)
        
        return self_feature + neighbor_mean
    
    def compute_Ti(self, src, dst, query_time):
        src_neighbors = self.nodes_neighbor_ids[src]
        src_neighbor_times = self.nodes_neighbor_times[src]

        dst_neighbors = self.nodes_neighbor_ids[dst]
        dst_neighbor_times = self.nodes_neighbor_times[dst]

        if self.delta == 0:
            valid_src_neighbors = {neighbor for neighbor, t in zip(src_neighbors, src_neighbor_times) if t <= query_time}
            valid_dst_neighbors = {neighbor for neighbor, t in zip(dst_neighbors, dst_neighbor_times) if t <= query_time}
        else:
            valid_src_neighbors = {
                neighbor for neighbor, t in zip(src_neighbors, src_neighbor_times) 
                if query_time - self.delta <= t <= query_time
            }
            valid_dst_neighbors = {
                neighbor for neighbor, t in zip(dst_neighbors, dst_neighbor_times) 
                if query_time - self.delta <= t <= query_time
            }

        common_neighbors = valid_src_neighbors & valid_dst_neighbors

        return len(common_neighbors)
    
    def _get_theta(self):
        if "global" in self.theta_cache:
            return self.theta_cache["global"]

        integrals = []
        
        max_sample_pairs = min(10000, self.num_nodes * 10)
        sampled_pairs = 0
        
        for src in range(min(self.num_nodes, 1000)):
            for dst in self.nodes_neighbor_ids[src]:
                if sampled_pairs >= max_sample_pairs:
                    break
                    
                window_records = [(t, 1) for t in self.nodes_neighbor_times[src] if dst in self.nodes_neighbor_ids[src]]

                if not window_records:
                    continue

                if self.phi_type == "cumulative":
                    integral = sum(weight for _, weight in window_records)
                elif self.phi_type == "exponential":
                    integral = sum(weight * np.exp(-self.lambda_ * (max(self.nodes_neighbor_times[src]) - t))
                                for t, weight in window_records)

                integrals.append(integral)
                sampled_pairs += 1
            
            if sampled_pairs >= max_sample_pairs:
                break

        theta = np.percentile(integrals, self.theta_percentile) if integrals else 0

        self.theta_cache["global"] = theta  
        return theta

    def classify_interaction(self, src, dst, query_time):
        Ti = self.compute_Ti(src, dst, query_time)
        theta = self._get_theta()
        return 1 if Ti >= theta else 0
    
    def find_counterfactual_with_breadth_search(self, src, dst, query_time, original_Ti):
        src_k_hop = self.get_k_hop_neighbors(src, query_time)
        dst_k_hop = self.get_k_hop_neighbors(dst, query_time)
        
        for k in range(1, self.max_hop + 1):
            print(f"搜索第{k}跳邻域...")
            
            src_candidates = src_k_hop.get(k, [])
            dst_candidates = dst_k_hop.get(k, [])
            
            if not src_candidates and not dst_candidates:
                continue
            
            candidate_pairs = []
            
            for candidate_src in src_candidates:
                if candidate_src != dst:
                    candidate_pairs.append((candidate_src, dst))
            
            for candidate_dst in dst_candidates:
                if candidate_dst != src:
                    candidate_pairs.append((src, candidate_dst))
            
            for candidate_src in src_candidates:
                for candidate_dst in dst_candidates:
                    if candidate_src != candidate_dst:
                        candidate_pairs.append((candidate_src, candidate_dst))
            
            if len(candidate_pairs) > 100:
                candidate_pairs = random.sample(candidate_pairs, 100)
            
            valid_candidates = []
            for cf_src, cf_dst in candidate_pairs:
                try:
                    cf_Ti = self.compute_Ti(cf_src, cf_dst, query_time)
                    
                    if cf_Ti != original_Ti:
                        src_embedding = self.get_node_embedding(src, query_time)
                        dst_embedding = self.get_node_embedding(dst, query_time)
                        cf_src_embedding = self.get_node_embedding(cf_src, query_time)
                        cf_dst_embedding = self.get_node_embedding(cf_dst, query_time)
                        
                        src_sim = F.cosine_similarity(src_embedding.unsqueeze(0), 
                                                    cf_src_embedding.unsqueeze(0), dim=1).item()
                        dst_sim = F.cosine_similarity(dst_embedding.unsqueeze(0), 
                                                    cf_dst_embedding.unsqueeze(0), dim=1).item()
                        
                        hop_weight = 1.0 / k
                        combined_score = hop_weight * (src_sim + dst_sim) / 2
                        
                        valid_candidates.append((cf_src, cf_dst, cf_Ti, combined_score, k))
                        
                except Exception as e:
                    continue
            
            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[3], reverse=True)
                best_candidate = valid_candidates[0]
                print(f"在第{best_candidate[4]}跳找到反事实节点对")
                return best_candidate[0], best_candidate[1], best_candidate[2]
        
        print(f"在{self.max_hop}跳邻域内未找到合适的反事实节点对")
        return None
    
    def find_similar_pairs(self, src_nodes, dst_nodes, node_interact_times, time_threshold=3600, max_candidates=5000):
        
        similar_src_list = []
        similar_dst_list = []
        interaction_times = []
        Ti_values = []
        src_Ti_values = []
        total_samples = len(src_nodes)
        
        batch_size = min(500, total_samples // 10)
        if batch_size < 100:
            batch_size = 100
            
        print(f"开始使用breadth search处理 {total_samples} 个样本，批次大小: {batch_size}")

        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_src = src_nodes[batch_start:batch_end]
            batch_dst = dst_nodes[batch_start:batch_end]
            batch_times = node_interact_times[batch_start:batch_end]
            
            print(f"处理批次 {batch_start//batch_size + 1}/{(total_samples-1)//batch_size + 1}")
            
            for i, (src, dst, t) in enumerate(zip(batch_src, batch_dst, batch_times)):
                try:
                    original_Ti = self.compute_Ti(src, dst, t)
                    src_Ti_values.append(original_Ti)
                    
                    cf_result = self.find_counterfactual_with_breadth_search(src, dst, t, original_Ti)
                    
                    if cf_result:
                        cf_src, cf_dst, cf_Ti = cf_result
                        similar_src_list.append(cf_src)
                        similar_dst_list.append(cf_dst)
                        interaction_times.append(t)
                        Ti_values.append(cf_Ti)
                    else:
                        alt_src = (src + 1) % self.num_nodes
                        alt_dst = (dst + 1) % self.num_nodes
                        alt_Ti = 1 - original_Ti if original_Ti in [0, 1] else np.random.choice([0, 1])
                        
                        similar_src_list.append(alt_src)
                        similar_dst_list.append(alt_dst)
                        interaction_times.append(t)
                        Ti_values.append(alt_Ti)
                        
                except Exception as e:
                    print(f"处理样本 {i} 时出错: {e}")
                    if len(src_Ti_values) <= i:
                        src_Ti_values.append(0)
                    
                    alt_src = (src + 1) % self.num_nodes
                    alt_dst = (dst + 1) % self.num_nodes
                    alt_Ti = np.random.choice([0, 1])
                    
                    similar_src_list.append(alt_src)
                    similar_dst_list.append(alt_dst)
                    interaction_times.append(t)
                    Ti_values.append(alt_Ti)
            
            if (batch_start // batch_size) % 3 == 0:
                gc.collect()

        print(f"Breadth search处理完成: 共 {total_samples} 个样本")

        return (np.array(similar_src_list, dtype=np.int64),
                np.array(similar_dst_list, dtype=np.int64), 
                np.array(interaction_times),
                np.array(Ti_values, dtype=np.int64),
                np.array(src_Ti_values, dtype=np.int64))

    def find_nosimilar_pairs(self, src_nodes, dst_nodes, node_interact_times, random_ratio=1, epsilon=0.3, time_threshold=3600):
        total_samples = len(src_nodes)
        batch_size = min(500, total_samples // 10)
        if batch_size < 100:
            batch_size = 100
            
        print(f"开始使用breadth search生成负样本，批次大小: {batch_size}")
        
        nosimilar_src_list = []
        nosimilar_dst_list = []
        interaction_times = []
        nosimilar_Ti_values = []
        src_Ti_values = []

        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_src = src_nodes[batch_start:batch_end]
            batch_dst = dst_nodes[batch_start:batch_end]
            batch_times = node_interact_times[batch_start:batch_end]
            
            if batch_start % (batch_size * 5) == 0:
                print(f"负样本生成进度: {batch_start}/{total_samples} ({batch_start/total_samples:.1%})")
            
            for src, dst, t in zip(batch_src, batch_dst, batch_times):
                try:
                    original_Ti = self.compute_Ti(src, dst, t)
                    src_Ti_values.append(original_Ti)
                    
                    src_k_hop = self.get_k_hop_neighbors(src, t)
                    
                    all_candidates = []
                    for k in range(1, self.max_hop + 1):
                        candidates = src_k_hop.get(k, [])
                        all_candidates.extend(candidates)
                    
                    all_candidates = [n for n in all_candidates if n != dst]
                    
                    if all_candidates:
                        neg_dst = random.choice(all_candidates)
                        neg_Ti = self.compute_Ti(src, neg_dst, t)
                        
                        nosimilar_src_list.append(src)
                        nosimilar_dst_list.append(neg_dst)
                        interaction_times.append(t)
                        nosimilar_Ti_values.append(neg_Ti)
                    else:
                        neg_src = (src + 1) % self.num_nodes
                        neg_dst = (dst + 1) % self.num_nodes
                        neg_Ti = np.random.choice([0, 1])
                        
                        nosimilar_src_list.append(neg_src)
                        nosimilar_dst_list.append(neg_dst)
                        interaction_times.append(t)
                        nosimilar_Ti_values.append(neg_Ti)
                        
                except Exception as e:
                    if len(src_Ti_values) <= len(nosimilar_src_list):
                        src_Ti_values.append(0)
                    
                    neg_src = (src + 1) % self.num_nodes
                    neg_dst = (dst + 1) % self.num_nodes
                    neg_Ti = np.random.choice([0, 1])
                    
                    nosimilar_src_list.append(neg_src)
                    nosimilar_dst_list.append(neg_dst)
                    interaction_times.append(t)
                    nosimilar_Ti_values.append(neg_Ti)
            
            if (batch_start // batch_size) % 5 == 0:
                gc.collect()

        print(f"负样本生成完成: 共 {total_samples} 个样本")

        return nosimilar_src_list, nosimilar_dst_list, interaction_times, nosimilar_Ti_values, src_Ti_values

class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)

def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader

class SmallData:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)

def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'

    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    random.seed(2020)

    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * num_total_unique_node_ids)))

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)
    
    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                                node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, max_ids_list=None,max_dst_list=None,
                      max_src_label_list=None,max_dst_label_list=None)

    setmax=EnhancedNodeEncoder(full_data, max_hop=3)
    max_ids_list,max_dst_list,t_list,cf_Ti, src_Ti=setmax.find_similar_pairs(src_node_ids, dst_node_ids, node_interact_times)

    max_ids_list = np.array(max_ids_list)
    max_dst_list = np.array(max_dst_list)
    print("cf_Ti type:", type(cf_Ti))
    print("src_Ti type:", type(src_Ti))
    cf_Ti = np.array(cf_Ti)
    src_Ti = np.array(src_Ti)

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, max_ids_list=max_ids_list,max_dst_list=max_dst_list,
                      max_src_label_list=cf_Ti,max_dst_label_list=src_Ti)
    
    real_train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[real_train_mask], dst_node_ids=dst_node_ids[real_train_mask],node_interact_times=node_interact_times[real_train_mask],
                      edge_ids=edge_ids[real_train_mask], labels=labels[real_train_mask],max_ids_list=max_ids_list[real_train_mask],max_dst_list=max_dst_list[real_train_mask],
                      max_src_label_list=cf_Ti[real_train_mask],max_dst_label_list=src_Ti[real_train_mask]) 

    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time,observed_edges_mask)

    test_mask = node_interact_times > test_time
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    val_data = Data(src_node_ids=full_data.src_node_ids[val_mask], dst_node_ids=full_data.dst_node_ids[val_mask],
                    node_interact_times=full_data.node_interact_times[val_mask], edge_ids=full_data.edge_ids[val_mask], labels=full_data.labels[val_mask],max_ids_list=max_ids_list[val_mask],max_dst_list=max_dst_list[val_mask],max_src_label_list=cf_Ti[val_mask],max_dst_label_list=src_Ti[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],node_interact_times=node_interact_times[test_mask], 
                     edge_ids=edge_ids[test_mask],labels=labels[test_mask],
                     max_ids_list=max_ids_list[test_mask],max_dst_list=max_dst_list[test_mask],max_src_label_list=cf_Ti[test_mask],max_dst_label_list=src_Ti[test_mask])

    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask],max_ids_list=max_ids_list[new_node_val_mask],max_dst_list=max_dst_list[new_node_val_mask],
                      max_src_label_list=cf_Ti[new_node_val_mask],max_dst_label_list=src_Ti[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask],max_ids_list=max_ids_list[new_node_test_mask],max_dst_list=max_dst_list[new_node_test_mask],
                      max_src_label_list=cf_Ti[new_node_test_mask],max_dst_label_list=src_Ti[new_node_test_mask])
    
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data

def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_or_create_data(folder_path, dataset_name, val_ratio, test_ratio):
    files_exist = all(os.path.exists(os.path.join(folder_path, f"{name}.pkl")) for name in [
        'node_raw_features', 'edge_raw_features', 'full_data', 'train_data', 'val_data', 'test_data', 'new_node_val_data', 'new_node_test_data'])

    if files_exist:
        print(f"Loading data from {folder_path}...")
        node_raw_features = load_data(os.path.join(folder_path, 'node_raw_features.pkl'))
        edge_raw_features = load_data(os.path.join(folder_path, 'edge_raw_features.pkl'))
        full_data = load_data(os.path.join(folder_path, 'full_data.pkl'))
        train_data = load_data(os.path.join(folder_path, 'train_data.pkl'))
        val_data = load_data(os.path.join(folder_path, 'val_data.pkl'))
        test_data = load_data(os.path.join(folder_path, 'test_data.pkl'))
        new_node_val_data = load_data(os.path.join(folder_path, 'new_node_val_data.pkl'))
        new_node_test_data = load_data(os.path.join(folder_path, 'new_node_test_data.pkl'))
    else:
        print(f"Data not found in {folder_path}. Creating data...")
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
            get_link_prediction_data(dataset_name=dataset_name, val_ratio=val_ratio, test_ratio=test_ratio)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        save_data(node_raw_features, os.path.join(folder_path, 'node_raw_features.pkl'))
        save_data(edge_raw_features, os.path.join(folder_path, 'edge_raw_features.pkl'))
        save_data(full_data, os.path.join(folder_path, 'full_data.pkl'))
        save_data(train_data, os.path.join(folder_path, 'train_data.pkl'))
        save_data(val_data, os.path.join(folder_path, 'val_data.pkl'))
        save_data(test_data, os.path.join(folder_path, 'test_data.pkl'))
        save_data(new_node_val_data, os.path.join(folder_path, 'new_node_val_data.pkl'))
        save_data(new_node_test_data, os.path.join(folder_path, 'new_node_test_data.pkl'))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data

def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data