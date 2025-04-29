from src import *
import argparse
from pprint import pprint

dataset_path = "dataset"



def init():
    print("="*10+"Initializing"+"="*10)
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
    
    print(f"[+] Arguments")
    pprint(vars(args))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'[+] device: {device}')
    print("\n")
    return args, device



def init_dataset(args, device):
    print("="*10+"Initializing Dataset"+"="*10)
    graph = CustomDataset(args, device)
    graph.load_node()
    graph.load_edge()
    graph.encoding_node()
    graph.build_graph()
    graph.split()
    print("")
    return graph


def train(args, device, dataset):
    print("="*10+"Train"+"="*10)
    print(f"[*] Getting dataset loader")
    train_loader = dataset.get_train_loader("train")
    valid_loader = dataset.get_valid_loader("valid")
    test_loader = dataset.get_test_loader("test")
    
    train_nfeat = dataset.pop_node_feature("train")
    valid_nfeat = dataset.pop_node_feature("valid")
    test_nfeat = dataset.pop_node_feature("test")
    
    print(f"[*] Initializing model")    
    model = GraphSAGE(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.max_lr))
    loss_fn = torch.nn.BCELoss()
    
    train_loss_list = []
    valid_loss_list = []
    f1_list = []
    result_list = []
    not_improved = 0
    
    print(f"[*] Training model")
    model.train()
    for epoch in range(args.max_epoch):
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = model(batch, train_nfeat)
            loss = loss_fn(pred, batch.y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                pred = model(batch, valid_nfeat)
                loss = loss_fn(pred, batch.y.float())
                valid_loss += loss.item()
        
        valid_loss /= len(valid_loader)
        valid_loss_list.append(valid_loss)
        
        f1_score = evaluate(model, test_loader, test_nfeat, device)
        f1_list.append(f1_score)
        
        print(f"Epoch {epoch+1}/{args.max_epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, F1 Score: {f1_score:.4f}")
        
        if epoch > 0 and valid_loss > min(valid_loss_list[:-1]):
            not_improved += 1
            if not_improved >= args.early_stop:
                print("Early stopping")
                break
        else:
            not_improved = 0
    
    
        
        
        
if __name__ == "__main__":
    args, device = init()
    dataset = init_dataset(args, device)