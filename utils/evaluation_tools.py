import torch
import numpy as np


def mean_average_precision(args, database_hash, test_hash, database_labels, test_labels):
    # binary the hash code
    R = args.R
    T = args.T
    database_hash[database_hash < T] = -1
    database_hash[database_hash >= T] = 1
    test_hash[test_hash < T] = -1
    test_hash[test_hash >= T] = 1

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)

    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        if i % 100 == 0:
            print(str(i) + "/" + str(query_num))
        label = test_labels[i, :]  # the first test labels
        # label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0

        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx


def predict_hash_code(args, model, data_loader, device):
    model.eval()
    is_start = True

    for batch_idx, (data, label, image_root) in enumerate(data_loader):
        data, label = data.to(device), label.to(device)
        pred, feature = model(data)
        if is_start:
            all_output = pred.cpu().detach().float()
            all_feature = feature.cpu().detach().float()
            all_label = label.unsqueeze(-1).cpu().detach().float()
            all_root = image_root
            is_start = False
        else:
            all_output = torch.cat((all_output, pred.cpu().detach().float()), 0)
            all_feature = torch.cat((all_feature, feature.cpu().detach().float()), 0)
            all_label = torch.cat((all_label, label.unsqueeze(-1).cpu().detach().float()), 0)
            all_root = all_root + image_root

    return all_output.numpy().astype("float32"), all_feature.numpy().astype("float32"), all_label.numpy().astype("float32"), all_root


@torch.no_grad()
def test_MAP(args, model, database_loader, test_loader, device, vis=False):
    print('Waiting for generate the hash code from database')
    database_hash, database_feature, database_labels, base_root = predict_hash_code(args, model, database_loader, device)
    print('Waiting for generate the hash code from test set')
    test_hash, test_feature, test_labels, query_root = predict_hash_code(args, model, test_loader, device)
    print('Calculate MAP.....')
    MAP, R, APx = mean_average_precision(args, database_hash, test_hash, database_labels, test_labels)

    if not vis:
        return MAP
    else:
        return MAP, database_hash, database_feature, database_labels, base_root, test_hash, test_feature, test_labels, query_root


def for_retrival(args, database_hash, test_hash):
    R = args.R
    T = args.T

    database_hash[database_hash < T] = -1
    database_hash[database_hash >= T] = 1
    test_hash[test_hash < T] = -1
    test_hash[test_hash >= T] = 1
    sim = np.matmul(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)
    idx = ids[:, 0]
    ids = idx[:R]
    return ids