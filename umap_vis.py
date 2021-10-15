import h5py
import argparse
import umap
from utils.draw_tools import draw_umap
from configs import parser
from loaders.data_process import DataSplit
import numpy as np


def main():
    f1 = h5py.File(f"data_map/{args.base_model}_{args.hash_len}_meta-training.hdf5")
    feature1 = np.array(f1["database_feature"])
    labels1 = np.array(f1["database_labels"])

    f2 = h5py.File(f"data_map/{args.base_model}_{args.hash_len}_transfer.hdf5")
    feature2 = np.array(f2["database_feature"])
    labels2 = np.array(f2["database_labels"])

    results1 = umap.UMAP(random_state=2).fit_transform(feature1)
    results2 = umap.UMAP(random_state=2).fit_transform(feature2)

    cls = DataSplit().cat
    draw_umap([results1, results2], [labels1, labels2], cls)


if __name__ == '__main__':
    args = parser.parse_args()
    main()