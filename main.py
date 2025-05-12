import argparse
import shutil
from pprint import pprint

from src import *

TOTAL_WIDTH = 50



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
    parser.add_argument('--model', type=str, default="mlp", help='model type')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed for initialization')
    parser.add_argument('--relu', type=float, default=0.2, help='ReLU threshold')
    parser.add_argument('--batch_size', type=int, default=65536, help='batch size for training')
    parser.add_argument('--embed_size', type=int, default=256, help='size of the embedding layer')
    parser.add_argument('--hidden_size', type=int, default=256, help='size of the hidden layer')
    parser.add_argument('--gnn_depth', type=int, default=2, help='depth of the GNN')
    parser.add_argument('--max_lr', type=float, default=0.001, help='maximum learning rate')
    parser.add_argument('--warmup', type=float, default=0.1, help='warmup ratio for learning rate')
    parser.add_argument('--max_epoch', type=int, default=1000, help='maximum number of epochs')
    parser.add_argument('--early_stop', type=int, default=50, help='early stopping epochs')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='early stopping epochs')

    args = parser.parse_args()
    logger = Logger(args)
    logger.print("Initializing".center(TOTAL_WIDTH, "="))
    
    logger.print(f"[+] Arguments")
    pprint(vars(args))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.print(f'[+] device: {device}')
    logger.print("\n")
    return args, device, logger



def init_dataset(args, device, logger):
    logger.print("Initializing Dataset".center(TOTAL_WIDTH, "="))
    graph = CustomDataset(args, device, logger)
    graph.load_node()
    graph.load_edge()
    graph.encoding_node()
    graph.build_graph()
    graph.split()
    logger.print("")
    return graph


def train(args, device, dataset, logger):
    logger.print("Training".center(TOTAL_WIDTH, "="))
    logger.print(f"[*] Initializing save path")
    path = os.path.join(args.model_path, args.setting, args.model_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    
    logger.print(f"[*] Getting dataset loader")
    train_loader = dataset.get_dataset_loader("train")
    valid_loader = dataset.get_dataset_loader("valid")
    test_loader = dataset.get_dataset_loader("test")
    
    if args.setting == "inductive":
        train_nfeat = dataset.pop_node_feature("train")
        valid_nfeat = dataset.pop_node_feature("valid")
        test_nfeat = dataset.pop_node_feature("test")
    elif args.setting == "transductive":
        train_nfeat = dataset.pop_node_feature("train")
        valid_nfeat = train_nfeat
        test_nfeat = train_nfeat
    
    logger.print(f"[*] Initializing model {args.model}")
    if args.model == "mlp":
        model = MLP(args).to(device)
    elif args.model == "graph_sage":    
        model = GraphSAGE(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.max_lr))
    loss_fn = torch.nn.BCELoss()
    early_stopper = EarlyStopping(args)
    
    logger.print(f"[*] Training model")
    train_loss_list = []
    valid_loss_list = []
    f1_list = []
    result_list = []
    
    model.train()
    for epoch in range(1, args.max_epoch+1):
        total_loss = 0.0
        for input_nodes, edge_sub, blocks in train_loader:
            optimizer.zero_grad()
            batch_inputs, batch_labels = load_subtensor(*train_nfeat, edge_sub, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            output, attn = model(edge_sub, blocks, *batch_inputs)
            probs = torch.sigmoid(output)
            batch_labels = torch.nn.functional.one_hot(batch_labels.long(), num_classes=2).float()
            loss = loss_fn(probs.float(), batch_labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        train_loss_list.append(train_loss)
        
        total_loss = 0.0
        for input_nodes, edge_sub, blocks in valid_loader:
            optimizer.zero_grad()
            batch_inputs, batch_labels = load_subtensor(*valid_nfeat, edge_sub, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            output, attn = model(edge_sub, blocks, *batch_inputs)
            probs = torch.sigmoid(output)
            batch_labels = torch.nn.functional.one_hot(batch_labels.long(), num_classes=2).float()
            loss = loss_fn(probs.float(), batch_labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()    
        valid_loss = total_loss / len(valid_loader)
        valid_loss_list.append(valid_loss)
        
        valid_result = evaluate(model, valid_loader, valid_nfeat, device)
        result_list.append(valid_result)
        f1_score = valid_result['macro avg']['f1-score']
        f1_list.append(f1_score)
        
        logger.print(f"[*] Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | f1-score: {f1_score:.4f} | Best f1-score: {max(f1_list):.4f}")
        
        path = os.path.join(args.model_path, args.setting, args.model_name, f"model_{epoch}.pth")
        save_checkpoint(path, model, optimizer, valid_result)
        if f1_score >= max(f1_list):
            path = os.path.join(args.model_path, args.setting, args.model_name, "model_best.pth")
            save_checkpoint(path, model, optimizer, valid_result)
            
        early_stopper(valid_loss)
        if early_stopper.early_stop:
            logger.print(f"[*] Early stopping at epoch {epoch}")
            break
    test_result = evaluate(model, test_loader, test_nfeat, device)
    logger.print(f"[+] Test result | accuracy: {test_result['accuracy']:.4f} | f1-score: {test_result['macro avg']['f1-score']:.4f} | precision: {test_result['macro avg']['precision']:.4f} | recall: {test_result['macro avg']['recall']:.4f}")
    logger.print(f"[+] Done training\n\n")
        
        
        
if __name__ == "__main__":
    args, device, logger = init()
    dataset = init_dataset(args, device, logger)
    train(args, device, dataset, logger)