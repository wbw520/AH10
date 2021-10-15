# Code for building project
# Bowen Wang
# {bowen.wang}@is.ids.osaka-u.ac.jp

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of OCT project")
parser.add_argument('--data_root', type=str, default="/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e/oct_tif")
parser.add_argument('--output_dir', type=str, default="saved_model")

# ========================= Model Configs ==========================
parser.add_argument('--base_model', default="resnet18", type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--img_size', default=260, help='path for save data')
parser.add_argument('--pre_trained', default=True, type=bool,
                    help='whether use ImageNet pre-train parameter for backbone')
parser.add_argument('--hash_len', default=32, help='length for hash code')
parser.add_argument('--bias', type=float, default=0.1, help='bias for quantization loss')

# ========================= Training Configs ==========================
parser.add_argument('--database_size', default=0.9, help='set the ratio for database and query split')
parser.add_argument('--meta_training', default=False, help='doing meta training')

# ========================= Learning Configs ==========================
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_drop', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=15, type=int)

# ========================= Machine Configs ==========================
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')

# ========================= mAP Configs ==========================
parser.add_argument('--R', type=int, default=10, help='MAP@R')
parser.add_argument('--T', type=float, default=0, help='Threshold for binary')
