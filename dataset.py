import os

import pandas as pd
import torch
import torch.utils.data as data
import transformers

SCORE_RANGES = {1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60)}
NUM_SCORES = {key: max_score - min_score + 1 for key, (min_score, max_score) in SCORE_RANGES.items()}


def load_samples(root, prompt_idx, fold_idx, name='train'):
    samples = pd.read_csv(os.path.join(root, 'split_data', f'fold_{fold_idx}', f'{name}.tsv'), sep='\t', header=None).loc[lambda x: x[1] == prompt_idx]
    data_ids = torch.tensor(samples[0].tolist())
    essays = samples[2].tolist()
    cls_labels = torch.tensor(samples[6].tolist()) - SCORE_RANGES[prompt_idx][0]
    return data_ids, essays, cls_labels


def encode_to_bert(essays):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(essays, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask


def load_data(root, prompt_idx, fold_idx, name):
    print(f'Loading BERT input of prompt {prompt_idx} (fold {fold_idx}, {name})...')
    sample_data_ids, essays, cls_labels = load_samples(root, prompt_idx, fold_idx, name)
    feature_data_ids, features = load_features(root, prompt_idx)

    mask = torch.eq(sample_data_ids[None, :], feature_data_ids[:, None])
    features = features[torch.argmax(mask.float(), dim=0), :]
    features = features[:, [4, 5, 6, 7, 10, 12, 14, 15, 19, 21, 95, 97, 98, 99, 100, 101, 103, 104, 105, 118]]
    # features = cls_labels[:, None]
    # features = features[:, [10, 12, 14, 15, 19, 21, 95, 97, 98, 99, 100, 101, 103, 104, 105, 118]]
    # features = torch.mean(features, dim=-1, keepdim=True)

    return essays, features, cls_labels


def load_features(root, prompt_idx):
    object_ids, data_ids = [], []
    with open(os.path.join(root, 'features', f'ASAP_PAES_feature_129_essay_order.txt')) as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        if line.startswith(f'ASAP/{prompt_idx}_'):
            object_ids.append(idx)
            data_ids.append(int(line[7:-4]))
    frame = pd.read_csv(os.path.join(root, 'features', f'ASAP_PAES_feature_129.csv'), header=None)
    frame = frame.loc[object_ids, :]

    data_ids = torch.tensor(data_ids)
    features = torch.tensor(frame.to_numpy()).float()

    return data_ids, features


def load_datasets(root, prompt_idx, fold_idx=0):
    train_essays, train_features, train_cls_labels = load_data(root, prompt_idx, fold_idx, 'train')
    dev_essays, dev_features, dev_cls_labels = load_data(root, prompt_idx, fold_idx, 'dev')
    test_essays, test_features, test_cls_labels = load_data(root, prompt_idx, fold_idx, 'test')

    input_ids, attention_mask = encode_to_bert(train_essays + dev_essays + test_essays)
    train_input_ids = input_ids[:len(train_essays), ...]
    train_attention_mask = attention_mask[:len(train_essays), ...]
    dev_input_ids = input_ids[len(train_essays):len(train_essays) + len(dev_essays), ...]
    dev_attention_mask = attention_mask[len(train_essays):len(train_essays) + len(dev_essays), ...]
    test_input_ids = input_ids[len(train_essays) + len(dev_essays):, ...]
    test_attention_mask = attention_mask[len(train_essays) + len(dev_essays):, ...]

    train_dataset = data.TensorDataset(train_input_ids, train_attention_mask, train_features, train_cls_labels)
    dev_dataset = data.TensorDataset(dev_input_ids, dev_attention_mask, dev_features, dev_cls_labels)
    test_dataset = data.TensorDataset(test_input_ids, test_attention_mask, test_features, test_cls_labels)

    return train_dataset, dev_dataset, test_dataset
