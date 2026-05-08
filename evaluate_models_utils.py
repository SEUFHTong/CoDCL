import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from models.EdgeBank import edge_bank_link_prediction
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data

class CausalEvaluationWrapper:
    def __init__(self, model, use_intervention=True):
        self.model = model
        self.use_intervention = use_intervention
        
    def predict(self, src_embeddings, dst_embeddings):
        if hasattr(self.model[1], 'forward') and len(self.model[1].__class__.__bases__) > 0:
            if 'InterventionalPredictor' in str(type(self.model[1])):
                input_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
                predictions, _ = self.model[1](input_features, use_intervention=self.use_intervention)
                return predictions.squeeze(dim=-1).sigmoid()
            else:
                return self.model[1](input_1=src_embeddings, input_2=dst_embeddings).squeeze(dim=-1).sigmoid()
        else:
            return self.model[1](input_1=src_embeddings, input_2=dst_embeddings).squeeze(dim=-1).sigmoid()


def compute_causal_evaluation_loss(loss_func, predicts, labels, 
                                  factual_embeddings=None, 
                                  counterfactual_embeddings=None, 
                                  negative_embeddings=None):
    try:
        if hasattr(loss_func, 'forward') and len(loss_func.forward.__code__.co_varnames) > 3:
            if factual_embeddings is not None and counterfactual_embeddings is not None:
                total_loss, factual_loss, treatment_loss, propensity_loss = loss_func(
                    predicts, labels, factual_embeddings, counterfactual_embeddings, negative_embeddings
                )
                return total_loss, factual_loss, treatment_loss, propensity_loss
            else:
                if hasattr(loss_func, 'bce_loss'):
                    basic_loss = loss_func.bce_loss(predicts, labels)
                else:
                    basic_loss = nn.BCELoss()(predicts, labels)
                return basic_loss, basic_loss, torch.tensor(0.0), torch.tensor(0.0)
        else:
            if callable(loss_func):
                try:
                    loss, loss_fact, loss_contrast = loss_func(predicts, labels, 
                                                              counterfactual_embeddings, 
                                                              factual_embeddings, 
                                                              negative_embeddings)
                    return loss, loss_fact, loss_contrast, torch.tensor(0.0)
                except:
                    loss = loss_func(predicts, labels)
                    return loss, loss, torch.tensor(0.0), torch.tensor(0.0)
            else:
                loss = nn.BCELoss()(predicts, labels)
                return loss, loss, torch.tensor(0.0), torch.tensor(0.0)
    except Exception as e:
        loss = nn.BCELoss()(predicts, labels)
        return loss, loss, torch.tensor(0.0), torch.tensor(0.0)


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task with causal inference support
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()
    
    eval_wrapper = CausalEvaluationWrapper(model, use_intervention=True)

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_factual_losses, evaluate_treatment_losses, evaluate_propensity_losses = [], [], []
        
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_max_ids, batch_max_dst = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                evaluate_data.max_ids_list[evaluate_data_indices], evaluate_data.max_dst_list[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            batch_cf_src_embeddings = None
            batch_cf_dst_embeddings = None

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
                
                if hasattr(evaluate_data, 'max_ids_list') and batch_max_ids is not None:
                    batch_cf_src_embeddings, batch_cf_dst_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_max_ids,
                                                                          dst_node_ids=batch_max_dst,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors)

            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)

                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
                
                if hasattr(evaluate_data, 'max_ids_list') and batch_max_ids is not None:
                    batch_cf_src_embeddings, batch_cf_dst_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_max_ids,
                                                                          dst_node_ids=batch_max_dst,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=num_neighbors)

            elif model_name in ['GraphMixer']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
                
                if hasattr(evaluate_data, 'max_ids_list') and batch_max_ids is not None:
                    batch_cf_src_embeddings, batch_cf_dst_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_max_ids,
                                                                          dst_node_ids=batch_max_dst,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors,
                                                                          time_gap=time_gap)

            elif model_name in ['DyGFormer']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
                
                if hasattr(evaluate_data, 'max_ids_list') and batch_max_ids is not None:
                    batch_cf_src_embeddings, batch_cf_dst_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_max_ids,
                                                                          dst_node_ids=batch_max_dst,
                                                                          node_interact_times=batch_node_interact_times)

            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            positive_probabilities = eval_wrapper.predict(batch_src_node_embeddings, batch_dst_node_embeddings)
            negative_probabilities = eval_wrapper.predict(batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings)

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            factual_embeddings = torch.cat([batch_src_node_embeddings, batch_dst_node_embeddings], dim=1)
            negative_embeddings = torch.cat([batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings], dim=1)
            
            counterfactual_embeddings = None
            if batch_cf_src_embeddings is not None and batch_cf_dst_embeddings is not None:
                counterfactual_embeddings = torch.cat([batch_cf_src_embeddings, batch_cf_dst_embeddings], dim=1)

            try:
                loss, factual_loss, treatment_loss, propensity_loss = compute_causal_evaluation_loss(
                    loss_func, predicts, labels, factual_embeddings, counterfactual_embeddings, negative_embeddings
                )
                
                evaluate_losses.append(loss.item())
                evaluate_factual_losses.append(factual_loss.item() if isinstance(factual_loss, torch.Tensor) else factual_loss)
                evaluate_treatment_losses.append(treatment_loss.item() if isinstance(treatment_loss, torch.Tensor) else treatment_loss)
                evaluate_propensity_losses.append(propensity_loss.item() if isinstance(propensity_loss, torch.Tensor) else propensity_loss)
                
            except Exception as e:
                loss = nn.BCELoss()(predicts, labels)
                evaluate_losses.append(loss.item())
                evaluate_factual_losses.append(loss.item())
                evaluate_treatment_losses.append(0.0)
                evaluate_propensity_losses.append(0.0)

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            if len(evaluate_treatment_losses) > 0 and evaluate_treatment_losses[-1] > 0:
                evaluate_idx_data_loader_tqdm.set_description(
                    f'evaluate batch {batch_idx + 1}, '
                    f'total_loss: {evaluate_losses[-1]:.4f}, '
                    f'factual_loss: {evaluate_factual_losses[-1]:.4f}, '
                    f'treatment_loss: {evaluate_treatment_losses[-1]:.4f}'
                )
            else:
                evaluate_idx_data_loader_tqdm.set_description(
                    f'evaluate batch {batch_idx + 1}, loss: {evaluate_losses[-1]:.4f}'
                )

    return evaluate_losses, evaluate_metrics


def evaluate_model_link_prediction_with_causal_analysis(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, 
                                                        evaluate_idx_data_loader: DataLoader, evaluate_neg_edge_sampler: NegativeEdgeSampler, 
                                                        evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000,
                                                        return_detailed_results: bool = False):
    evaluate_losses, evaluate_metrics = evaluate_model_link_prediction(
        model_name, model, neighbor_sampler, evaluate_idx_data_loader, 
        evaluate_neg_edge_sampler, evaluate_data, loss_func, num_neighbors, time_gap
    )
    
    if return_detailed_results:
        detailed_results = {
            'basic_metrics': evaluate_metrics,
            'losses': evaluate_losses,
            'average_loss': np.mean(evaluate_losses),
            'loss_std': np.std(evaluate_losses)
        }
        
        if hasattr(loss_func, 'compute_treatment_effect'):
            detailed_results['has_causal_components'] = True
        else:
            detailed_results['has_causal_components'] = False
            
        return evaluate_losses, evaluate_metrics, detailed_results
    
    return evaluate_losses, evaluate_metrics


def evaluate_model_link_prediction_legacy(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                          evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                          num_neighbors: int = 20, time_gap: int = 2000):
    return evaluate_model_link_prediction(model_name, model, neighbor_sampler, evaluate_idx_data_loader,
                                          evaluate_neg_edge_sampler, evaluate_data, loss_func, num_neighbors, time_gap)

def CF_evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    # args = get_link_prediction_args(is_evaluation=False)
    # loss_func = nn.BCELoss()
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_max_ids,batch_max_dst = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices],evaluate_data.max_ids_list[evaluate_data_indices],evaluate_data.max_dst_list[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
                batch_src_node_embeddings2, batch_dst_node_embeddings2 = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_max_ids,
                                                                        dst_node_ids=batch_max_dst,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
                batch_src_node_embeddings2, batch_dst_node_embeddings2 = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_max_ids,
                                                                    dst_node_ids=batch_max_dst,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)                                                      
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings2, batch_dst_node_embeddings2 = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_max_ids,
                                                                    dst_node_ids=batch_max_dst,
                                                                    node_interact_times=batch_node_interact_times)

                # batch_neg_src_node_embeddings2, batch_neg_dst_node_embeddings2 = \
                #     model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_nomax_ids,
                #                                                     dst_node_ids=batch_nomax_dst,
                #                                                     node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            
            z_pos=batch_src_node_embeddings * batch_dst_node_embeddings
            z_cf=batch_src_node_embeddings2 * batch_dst_node_embeddings2
            z_neg=batch_neg_src_node_embeddings*batch_neg_dst_node_embeddings
            #loss = loss_func(input=predicts, target=labels)
            loss, loss_fact, loss_contrast = loss_func(predicts, labels, z_cf, z_pos, z_neg)

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics


def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get predicted probabilities, shape (batch_size, )
            predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
            labels = torch.from_numpy(batch_labels).float().to(predicts.device)

            loss = loss_func(input=predicts, target=labels)

            evaluate_total_loss += loss.item()

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)

        evaluate_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics


def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_idx_data_loader: DataLoader,
                                       test_neg_edge_sampler: NegativeEdgeSampler, test_data: Data):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_idx_data_loader: DataLoader, test index data loader
    :param test_neg_edge_sampler: NegativeEdgeSampler, test negative edge sampler
    :param test_data: Data, test data
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids]),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids]),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times]),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids]),
                          labels=np.concatenate([train_data.labels, val_data.labels]))

    test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        loss_func = nn.BCELoss()

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
        assert test_neg_edge_sampler.seed is not None
        test_neg_edge_sampler.reset_random_state()

        test_losses, test_metrics = [], []
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_max_ids,batch_max_dst,batch_nomax_ids, batch_nomax_dst= \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices],test_data.max_ids_list[test_data_indices],test_data.max_dst_list[test_data_indices],\
                    test_data.max_src_label_list[test_data_indices],test_data.max_dst_label_list[test_data_indices]

            if test_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                              batch_src_node_ids=batch_src_node_ids,
                                                                                              batch_dst_node_ids=batch_dst_node_ids,
                                                                                              current_batch_start_time=batch_node_interact_times[0],
                                                                                              current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (batch_neg_src_node_ids, batch_neg_dst_node_ids)

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]]),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]]),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]]),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]]),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]]))

            # perform link prediction for EdgeBank
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=args.test_ratio)

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, negative_probabilities])).float()
            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(negative_probabilities))], dim=0)

            loss = loss_func(input=predicts, target=labels)

            test_losses.append(loss.item())

            test_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch, test loss: {loss.item()}')

        # store the evaluation metrics at the current run
        test_metric_dict = {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}'for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
