import argparse
from configs import parser
import torch
import h5py
import os, shutil
from termcolor import colored
from loaders.data_process import get_val_transformations, ImageNet, DataSplit
import numpy as np
from PIL import Image
from utils.evaluation_tools import for_retrival


shutil.rmtree('retrieval_results/', ignore_errors=True)
os.makedirs('retrieval_results/', exist_ok=True)


def main():
    # CUDNN
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    cls_name = DataSplit().cat

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    query_transformations = get_val_transformations(args)
    base_name = ImageNet(args, "train", query_transformations, meta=False).all_data
    query_dataset = ImageNet(args, "val", query_transformations, meta=True).all_data

    f1 = h5py.File(f"data_map/{args.base_model}_{args.hash_len}_transfer.hdf5")
    database_hash = np.array(f1["database_hash"])
    database_labels = np.array(f1["database_labels"])
    test_hash = np.array(f1["test_hash"])
    test_labels = np.array(f1["test_labels"])

    index = args.index
    query_sample = np.array([test_hash[index]])
    img_query = Image.open(query_dataset[index][0]).convert('RGB').resize([260, 260], resample=Image.BILINEAR)
    img_query.save(f"retrieval_results/query_{cls_name[int(test_labels[index])]}.png")
    ids = for_retrival(args, database_hash, query_sample)
    print("generating retrieval samples")
    for i in range(len(ids)):
        current_is = ids[i]
        img_re = Image.open(base_name[current_is][0]).convert('RGB').resize([260, 260], resample=Image.BILINEAR)
        img_re.save(f"retrieval_results/re_{i}_{cls_name[int(database_labels[current_is])]}.png")


if __name__ == '__main__':
    args = parser.parse_args()
    main()