import argparse
from configs import parser
import torch
from termcolor import colored
from model.main_model import BaseModel
from utils.tools import MetricLog
from loaders.data_process import get_train_transformations, get_val_transformations, ImageNet, \
    get_train_dataloader, get_val_dataloader
from engine import train_model
from utils.evaluation_tools import test_MAP


def main():
    # CUDNN
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    base_transformations = get_train_transformations(args)
    query_transformations = get_val_transformations(args)
    base_dataset_t = ImageNet(args, "train", base_transformations, meta=args.meta_training)
    base_dataset_e = ImageNet(args, "train", query_transformations, meta=args.meta_training)
    query_dataset = ImageNet(args, "val", query_transformations, meta=True)

    base_dataloader_t = get_train_dataloader(args, base_dataset_t)
    base_dataloader_e = get_val_dataloader(args, base_dataset_e)
    query_dataloader = get_val_dataloader(args, query_dataset)
    print('Train transforms:', base_transformations)
    print('Validation transforms:', query_transformations)
    print('Train samples %d - Val samples %d' % (len(base_dataset_t), len(query_dataset)))

    # Model
    print(colored('Get model', 'blue'))
    model = BaseModel(args)
    model = model.cuda()
    if not args.meta_training:
        checkpoint = torch.load("saved_models/building_" + args.base_model + "_" + str(args.hash_len) +
                                "meta-training.pth.tar",
                                map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        print("load meta model finished")
    print(colored('trainable parameter name: ', "blue"))

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    params = [p for p in model.parameters() if p.requires_grad]
    if not args.meta_training:
        args.lr = 0.0001
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    start_epoch = 0
    best_map = 0

    # Main loop
    print(colored('Starting main loop', 'green'))
    log = MetricLog()

    for epoch in range(start_epoch, args.epochs):
        print(colored('Epoch %d/%d' % (epoch+1, args.epochs), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        if epoch == args.lr_drop:
            print("Adjusted learning rate to 1/10")
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1

        train_model(args, base_dataloader_t, model, optimizer, epoch, device)

        if epoch % 2 == 0:
            print('Evaluate via val set ...')
            map = test_MAP(args, model, base_dataloader_e, query_dataloader, device)
            log.record["map"].append(map)

            if map > best_map:
                print('New higher acc on validation set: %.4f -> %.4f' % (best_map, map))
                best_map = map
                torch.save({'model': model.state_dict()}, "saved_models/building_" + args.base_model + "_" + str(args.hash_len) +
                           f"{'meta-training' if args.meta_training else 'transfer'}" +".pth.tar")
            else:
                print('No higher acc on validation set: %.4f -> %.4f' % (best_map, map))
            log.print_metric()


if __name__ == '__main__':
    args = parser.parse_args()
    main()