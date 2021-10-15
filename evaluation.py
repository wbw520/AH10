import argparse
from configs import parser
import torch
from termcolor import colored
from model.main_model import BaseModel
from utils.tools import MetricLog
from loaders.data_process import get_train_transformations, get_val_transformations, ImageNet, \
    get_train_dataloader, get_val_dataloader
from utils.evaluation_tools import test_MAP
import h5py


def main():
    # CUDNN
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    query_transformations = get_val_transformations(args)
    base_dataset = ImageNet(args, "train", query_transformations, meta=False)
    query_dataset = ImageNet(args, "val", query_transformations, meta=True)

    base_dataloader = get_val_dataloader(args, base_dataset)
    query_dataloader = get_val_dataloader(args, query_dataset)

    print('Train samples %d - Val samples %d' % (len(base_dataset), len(query_dataset)))

    # Model
    print(colored('Get model', 'blue'))
    model = BaseModel(args)
    model = model.cuda()
    print(colored('trainable parameter name: ', "blue"))
    checkpoint = torch.load("saved_models/building_" + args.base_model + "_" + str(args.hash_len) +
                           f"{'meta-training' if args.meta_training else 'transfer'}" +".pth.tar", map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)

    MAP, database_hash, database_feature, database_labels, base_root, test_hash, test_feature, test_labels, query_root \
        = test_MAP(args, model, base_dataloader, query_dataloader, device, vis=True)
    print(MAP)

    f = h5py.File(
        f"data_map/{args.base_model}_{args.hash_len}_{'meta-training' if args.meta_training else 'transfer'}.hdf5",
        "w")
    d1 = f.create_dataset("database_hash", data=database_hash)
    d2 = f.create_dataset("database_labels", data=database_labels)
    d3 = f.create_dataset("test_hash", data=test_hash)
    d4 = f.create_dataset("test_labels", data=test_labels)
    d5 = f.create_dataset("database_feature", data=database_feature)
    d6 = f.create_dataset("test_feature", data=test_feature)
    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    args.R = 10
    main()