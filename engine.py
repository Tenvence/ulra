import time

import numpy as np
import torch
import torch.cuda.amp as amp

from metrics import quadratic_weighted_kappa


def train(model, optimizer, data_loader):
    st = time.time()
    losses = []
    model.train()
    scaler = amp.GradScaler()

    for input_ids, attention_mask, features, labels in data_loader:
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)

        batch_size, num_features = features.shape

        optimizer.zero_grad()
        with amp.autocast():
            pred_score, weight_memory = model(input_ids, attention_mask)

            total_mask = torch.ones((batch_size, batch_size))
            idx_pairs = torch.nonzero(total_mask).cuda()

            features_a = features[idx_pairs[:, 0], :]  # [num_pairs, num_features]
            features_b = features[idx_pairs[:, 1], :]  # [num_pairs, num_features]
            ge_mask = torch.where(features_a >= features_b, weight_memory[None, :], 1. - weight_memory[None, :])

            pred_score_a = pred_score[idx_pairs[:, 0], :]
            pred_score_b = pred_score[idx_pairs[:, 1], :]
            term = pred_score_a.exp() / (pred_score_a.exp() + pred_score_b.exp())
            loss = -torch.log(ge_mask * term + (1 - ge_mask) * (1 - term)).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.detach())

    return torch.stack(losses).mean(), time.time() - st


@torch.no_grad()
def evaluate(model, data_loader, num_classes):
    st = time.time()
    model.eval()
    pred_scores_list, gt_labels_list = [], []
    for input_ids, attention_mask, _, cls_labels in data_loader:
        input_ids = input_ids.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        cls_labels = cls_labels.cuda(non_blocking=True)
        with amp.autocast():
            pred_scores, weight_memory = model(input_ids, attention_mask)
        pred_scores_list.append(pred_scores)
        gt_labels_list.append(cls_labels)
    pred_scores_list = torch.cat(pred_scores_list, dim=0).cpu().squeeze()
    gt_labels_list = torch.cat(gt_labels_list, dim=0).cpu()

    reg_scores = reg_scoring(pred_scores_list, num_classes)
    reg_qwk = quadratic_weighted_kappa(gt_labels_list, reg_scores)

    gt_scores = gt_scoring(pred_scores_list, gt_labels_list)
    gt_qwk = quadratic_weighted_kappa(gt_labels_list, gt_scores)

    u_scores = uniform_scoring(pred_scores_list, num_classes)
    u_qwk = quadratic_weighted_kappa(gt_labels_list, u_scores)

    t_scores = tri_scoring(pred_scores_list, num_classes)
    t_qwk = quadratic_weighted_kappa(gt_labels_list, t_scores)

    n_scores = normal_scoring(pred_scores_list, num_classes)
    n_qwk = quadratic_weighted_kappa(gt_labels_list, n_scores)

    return gt_qwk, u_qwk, reg_qwk, t_qwk, n_qwk, time.time() - st


def gt_scoring(pred_scores_list, gt_labels_list):
    gt_labels_sorted, gt_indices_sorted = torch.sort(gt_labels_list)
    gt_labels_set = sorted(set(gt_labels_sorted.numpy()))
    pred_scores_sorted, pred_indices_sorted = torch.sort(pred_scores_list)
    scores = torch.zeros(len(pred_scores_sorted))
    for label in gt_labels_set:
        ids = torch.nonzero(gt_labels_sorted == label).squeeze()
        scores[pred_indices_sorted[ids]] = int(label)
    return scores


def normal_scoring(pred_scores_list, num_classes):
    pred_scores_sorted, pred_indices_sorted = torch.sort(pred_scores_list)
    k = 1 / np.sqrt(2 * np.pi) * torch.exp(-(torch.arange(num_classes) - (num_classes - 1) / 2) ** 2 / 2)
    k = k / k.sum()

    num_list = [int(np.floor(len(pred_scores_list) * i)) for i in k]
    count = len(pred_scores_list) - sum(num_list)
    for i in range(num_classes):
        num_list[i] += 1
        count -= 1
        if count == 0:
            break

    scores = torch.zeros(len(pred_scores_list))
    sorted_labels = []
    for i, a in enumerate(num_list):
        sorted_labels.extend([i for _ in range(a)])
    sorted_labels = torch.tensor(sorted_labels, dtype=torch.float)
    scores[pred_indices_sorted] = sorted_labels

    return scores


def tri_scoring(pred_scores_list, num_classes):
    pred_scores_sorted, pred_indices_sorted = torch.sort(pred_scores_list)
    num_samples = len(pred_scores_list)
    # print(num_samples)

    k = -torch.abs(torch.arange(num_classes) - 1 - (num_classes - 1) / 2) + (num_classes + 1) / 2
    k /= torch.sum(k)
    # print(k)
    # k = np.floor(num_classes / 2)
    # if num_classes % 2 == 0:
    #     x = 1 / (k * (k + 1))
    #     num_list = [-np.abs(i + 1 - (k + 0.5)) + k + 0.5 for i in range(num_classes)]
    # else:
    #     x = 1 / ((k + 1) * (k + 1))
    #     num_list = [-np.abs(i + 1 - (k + 1)) + k + 1 for i in range(num_classes)]
    num_list = [int(np.floor(a * num_samples)) for a in k]
    # print(len(num_list))
    # print(num_list)

    count = num_samples - sum(num_list)
    for i in range(num_classes):
        if count <= 0:
            break
        num_list[i] += 1
        count -= 1
    # print(num_list)
    scores = torch.zeros(num_samples)
    sorted_labels = []
    for i, a in enumerate(num_list):
        sorted_labels.extend([i for _ in range(a)])
    sorted_labels = torch.tensor(sorted_labels, dtype=torch.float)
    # print(sorted_labels.shape)
    scores[pred_indices_sorted] = sorted_labels

    return scores


def uniform_scoring(pred_scores_list, num_classes):
    pred_scores_sorted, pred_indices_sorted = torch.sort(pred_scores_list)
    scores = torch.zeros(len(pred_scores_sorted))
    scores[pred_indices_sorted] = torch.arange(len(pred_scores_sorted)).float()
    scores = scores / (len(pred_scores_sorted)) * num_classes
    scores = torch.round(scores).long()
    return scores


def reg_scoring(pred_scores_list, num_classes):
    pred_scores_list -= min(pred_scores_list.clone())
    pred_scores_list /= max(pred_scores_list.clone())
    pred_scores_list *= (num_classes - 1)
    scores = torch.round(pred_scores_list.float()).long()
    return scores
