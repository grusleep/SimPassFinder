from src import *
import argparse
from pprint import pprint

dataset_path = "dataset"



def init():

    parser = argparse.ArgumentParser(description="Parser for model configuration")

    parser.add_argument('--dataset_path', type=str, required=True, help='dataset_path')
    parser.add_argument('--setting', type=str, required=True, help='graph learning setting: inductive/transductive')
    parser.add_argument('--model_path', type=str, required=False, default="./model/", help='model grand path')
    parser.add_argument('--model_name', type=str, default="model", help='model folder name')
    parser.add_argument('--agg_type', type=str, default="attn", help='aggregation type')
    parser.add_argument('--edge_thv', type=float, default=0.5, help='threshold for edge')
    parser.add_argument('--valid', type=float, default=0.2, help='split ratio of validation set')
    parser.add_argument('--test', type=float, default=0.2, help='split ratio of test set')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed for initialization')
    parser.add_argument('--relu', type=float, default=0.2, help='ReLU threshold')
    parser.add_argument('--reuse_th', type=float, default=0.5, help='threshold for reuse')
    parser.add_argument('--batch_size', type=int, default=65536, help='batch size for training')
    parser.add_argument('--embed_size', type=int, default=256, help='size of the embedding layer')
    parser.add_argument('--hidden_size', type=int, default=256, help='size of the hidden layer')
    parser.add_argument('--gnn_depth', type=int, default=2, help='depth of the GNN')
    parser.add_argument('--max_lr', type=float, default=0.001, help='maximum learning rate')
    parser.add_argument('--warmup', type=float, default=0.1, help='warmup ratio for learning rate')
    parser.add_argument('--max_epoch', type=int, default=200, help='maximum number of epochs')
    parser.add_argument('--early_stop', type=int, default=40, help='early stopping epochs')

    args = parser.parse_args()
    
    print("[*] Arguments")
    pprint(vars(args))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'[*] device: {device}')
    print("\n")
    return args, device



def init_dataset(args):
    graph = CustomDataset(args)
    graph.load_node()
    graph.load_edge()
    graph.encoding_node()
    graph.build_graph()
    graph.split()
    
    return graph
        
        
        
if __name__ == "__main__":
    args, defice = init()
    dataset = init_dataset(args)