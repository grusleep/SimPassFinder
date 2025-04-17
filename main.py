from src import *

dataset_path = "dataset"



class Config():
    def __init__(self):
        self.dataset_path = dataset_path
        self.edge_thv = 0.5
        
        
        
if __name__ == "__main__":
    config = Config()
    graph = CustomDataset(config)
    graph.load_node()
    graph.load_edge()
    graph.encoding_node()
    graph.build_graph()