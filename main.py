from dataset.movielensdataset import MovieLensDataSet
from CF.UserBasedCF import UserBasedCF
from CF.ItemBasedCF import ItemBasedCF
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning For Recommender System')
    parser.add_argument('--dataset', type=str, default='movielensdataset')
    parser.add_argument('--dataset-path', type=str, default='./dataset/ml-100k/')
    parser.add_argument('--method', type=str, default='ItemBasedCF')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset = None
    if args.dataset == "movielensdataset":
        dataset = MovieLensDataSet(dataset_path)
    method = None
    if args.method == "UserBasedCF":
        method = UserBasedCF(dataset)
    elif args.method == "ItemBasedCF":
        method = ItemBasedCF(dataset) 
    elif args.method == "DQN":
        method = DQN(dataset)


    method.test_model()
