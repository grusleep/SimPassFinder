# python3 python/main.py --dataset_path ./dataset --setting transductive --model_name SPF_depth_1_mean --model graph_sage --gnn_depth 1 --agg_type mean 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting transductive --model_name SPF_depth_2_mean --model graph_sage --gnn_depth 2 --agg_type mean 
# python3 python/main.py --dataset_path ./dataset --setting transductive --model_name SPF_depth_1_gcn --model graph_sage --gnn_depth 1 --agg_type gcn 
# python3 python/main.py --dataset_path ./dataset --setting transductive --model_name SPF_depth_2_gcn --model graph_sage --gnn_depth 2 --agg_type gcn 
# python3 python/main.py --dataset_path ./dataset --setting transductive --model_name SPF_depth_1 --model graph_sage --gnn_depth 1 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting transductive --model_name SPF_depth_2 --model graph_sage --gnn_depth 2 
# python3 python/main.py --dataset_path ./dataset --setting transductive --model_name SPF_depth_1_no_hidden --model graph_sage --gnn_depth 1 --agg_type no_hidden 
# python3 python/main.py --run_type test --dataset_path ./dataset --setting transductive --model_name SPF_depth_2_no_hidden --model graph_sage --gnn_depth 2 --agg_type no_hidden 
# python3 python/main.py --dataset_path ./dataset --setting transductive --model_name SPF_depth_1_no_neighbor --model graph_sage --gnn_depth 1 --agg_type no_neighbor 
# python3 python/main.py --dataset_path ./dataset --setting transductive --model_name SPF_depth_2_no_neighbor --model graph_sage --gnn_depth 2 --agg_type no_neighbor 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting transductive --model_name SPF_gcn --model gcn --gnn_depth 2
# python3 python/main.py --run_type train --dataset_path ./dataset --setting transductive --model_name SPF_gat --model gat --gnn_depth 2
python3 python/main.py --run_type train --dataset_path ./dataset --setting transductive --model_name SPF_mlp --model mlp
# python3 python/main.py --run_type train --dataset_path ./dataset --setting inductive --model_name SPF_mlp_all --model mlp --feature all 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting inductive --model_name SPF_mlp_site --model mlp --feature site 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting inductive --model_name SPF_mlp_category --model mlp --feature category 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting inductive --model_name SPF_mlp_country --model mlp --feature country 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting inductive --model_name SPF_mlp_sl --model mlp --feature sl 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting inductive --model_name SPF_mlp_ip --model mlp --feature ip 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting inductive --edge_type reuse --model_name SPF_mlp_reuse --model mlp 



# python3 python/main.py --dataset_path ./dataset --setting inductive --model_name SPF_depth_1_mean --model graph_sage --gnn_depth 1 --agg_type mean 
# python3 python/main.py --run_type test --dataset_path ./dataset --setting inductive --model_name SPF_depth_2_mean --model graph_sage --gnn_depth 2 --agg_type mean 
# python3 python/main.py --dataset_path ./dataset --setting inductive --model_name SPF_depth_1_gcn --model graph_sage --gnn_depth 1 --agg_type gcn 
# python3 python/main.py --dataset_path ./dataset --setting inductive --model_name SPF_depth_2_gcn --model graph_sage --gnn_depth 2 --agg_type gcn 
# python3 python/main.py --dataset_path ./dataset --setting inductive --model_name SPF_depth_1 --model graph_sage --gnn_depth 1 
# python3 python/main.py --run_type train --dataset_path ./dataset --setting inductive --model_name SPF_depth_2 --model graph_sage --gnn_depth 2 
# python3 python/main.py --dataset_path ./dataset --setting inductive --model_name SPF_depth_1_no_hidden --model graph_sage --gnn_depth 1 --agg_type no_hidden 
# python3 python/main.py --run_type test --dataset_path ./dataset --setting inductive --model_name SPF_depth_2_no_hidden --model graph_sage --gnn_depth 2 --agg_type no_hidden 
# python3 python/main.py --run_type test --dataset_path ./dataset --setting inductive --model_name SPF_depth_1_no_neighbor --model graph_sage --gnn_depth 1 --agg_type no_neighbor 
# python3 python/main.py --run_type test --dataset_path ./dataset --setting inductive --model_name SPF_depth_2_no_neighbor --model graph_sage --gnn_depth 2 --agg_type no_neighbor 
# python3 python/main.py --run_type test --dataset_path ./dataset --setting inductive --model_name SPF_gcn --model gcn --gnn_depth 2
# python3 python/main.py --run_type train --dataset_path ./dataset --setting inductive --model_name SPF_gat --model gat --gnn_depth 2
# python3 python/main.py --run_type test --dataset_path ./dataset --setting inductive --model_name SPF_mlp --model mlp