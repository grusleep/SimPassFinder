#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <tuple>
#include <algorithm>
#include <random>

#include <nlohmann/json.hpp>      // For JSON parsing
#include <jellyfish/jaro_winkler.hpp>

using json = nlohmann::json;



struct Args {
    std::string dataset_path;
    double edge_thv;
    std::string setting;
    double valid;
    double test;
    int gnn_depth;
    int random_seed;
    int batch_size;
    std::string edge_type;
};

class CustomDataset {
public:
    CustomDataset(const Args& args, const torch::Device& device)
        : dataset_path_(args.dataset_path),
          edge_thv_(args.edge_thv),
          splitting_type_(args.setting),
          valid_(args.valid),
          test_(args.test),
          gnn_depth_(args.gnn_depth),
          device_(device),
          random_seed_(args.random_seed),
          batch_size_(args.batch_size),
          edge_type_(args.edge_type),
          num_thv_(30),
          data_type_("old") {}


    void load_node() {
        std::cout << "[*] Loading nodes" << std::endl;
        std::string node_file;
        if (!data_type_.empty()) {
            node_file = dataset_path_ + "/graph/nodes_" + data_type_ + ".json";
        } else {
            node_file = dataset_path_ + "/graph/nodes.json";
        }
        std::ifstream f(node_file);
        json j;
        f >> j;
        nodes_ = j;
        std::cout << "[+] Done loading nodes" << std::endl;
        std::cout << "[+] Number of nodes: " << nodes_.size() << std::endl;
    }


    void set_edge(bool save = true) {
        std::cout << "[*] Start Setting edges" << std::endl;
        std::ofstream fout(dataset_path_ + "/graph/edges_cpp.txt", std::ios::app);

        for (size_t i = 0; i < nodes_list_.size(); ++i) {
            for (size_t j = i + 1; j < nodes_list_.size(); ++j) {
                std::string i_site = nodes_list_[i]["site"].get<std::string>();
                std::string j_site = nodes_list_[j]["site"].get<std::string>();
                std::ifstream f_i(dataset_path_ + "/sites/" + i_site + ".json");
                std::ifstream f_j(dataset_path_ + "/sites/" + j_site + ".json");
                json i_data, j_data;
                f_i >> i_data;
                f_j >> j_data;

                std::set<std::string> i_users;
                std::set<std::string> j_users;
                for (auto& it : i_data.items()) i_users.insert(it.key());
                for (auto& it : j_data.items()) j_users.insert(it.key());

                std::vector<std::string> intersection;
                std::set_intersection(i_users.begin(), i_users.end(),
                                      j_users.begin(), j_users.end(),
                                      std::back_inserter(intersection));
                if (intersection.size() >= num_thv_) {
                    std::vector<double> pwd_sim;
                    for (auto& user : intersection) {
                        std::string pwd1 = i_data[user].get<std::string>();
                        std::string pwd2 = j_data[user].get<std::string>();
                        double sim = jellyfish::jaro_winkler_similarity(pwd1, pwd2);
                        pwd_sim.push_back(sim);
                    }
                    double weight = std::accumulate(pwd_sim.begin(), pwd_sim.end(), 0.0) / pwd_sim.size();
                    fout << i << " " << j << " " << weight << "\n";
                }
                std::cout << "[*] Setting edges: " << i << " / " << nodes_list_.size() << " | " << i << " - " << j << std::endl;
            }
        }

        std::cout << "[+] Done setting edges" << std::endl;
        std::cout << "[+] Number of edges: " << edges_.size() << std::endl;
    }

    void set_edge_reuse(bool save = true) {
        edges_list_.clear();
        std::cout << "[*] Start Setting edges" << std::endl;

        for (size_t i = 0; i < nodes_list_.size(); ++i) {
            std::cout << "[*] Setting edges: " << i << " / " << nodes_list_.size() << std::endl;
            for (size_t j = i + 1; j < nodes_list_.size(); ++j) {
                std::string i_site = nodes_list_[i]["site"].get<std::string>();
                std::string j_site = nodes_list_[j]["site"].get<std::string>();
                std::ifstream f_i(dataset_path_ + "/sites/" + i_site + ".json");
                std::ifstream f_j(dataset_path_ + "/sites/" + j_site + ".json");
                json i_data, j_data;
                f_i >> i_data;
                f_j >> j_data;

                std::set<std::string> i_users;
                std::set<std::string> j_users;
                for (auto& it : i_data.items()) i_users.insert(it.key());
                for (auto& it : j_data.items()) j_users.insert(it.key());

                std::vector<std::string> intersection;
                std::set_intersection(i_users.begin(), i_users.end(),
                                      j_users.begin(), j_users.end(),
                                      std::back_inserter(intersection));
                if (intersection.size() < num_thv_) continue;
                bool check = false;
                for (auto& user : intersection) {
                    if (i_data[user].get<std::string>() == j_data[user].get<std::string>()) {
                        check = true; break;
                    }
                }
                if (!check) continue;
                json edge;
                edge["node_1"] = i;
                edge["node_2"] = j;
                edge["weight"] = 1;
                edges_list_.push_back(edge);
            }
        }

        std::cout << "[+] Done setting edges" << std::endl;
        std::cout << "[+] Number of edges: " << edges_list_.size() << std::endl;

        if (save) {
            std::ofstream f(dataset_path_ + "/graph/edges_reuse.json");
            json j = edges_list_;
            f << j.dump(4);
            std::cout << "[+] Saved edges to " << dataset_path_ + "/graph/edges_reuse.json" << std::endl;
        }
    }

private:
    
};
