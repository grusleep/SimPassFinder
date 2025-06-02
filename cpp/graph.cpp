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

#include <dgl/graph.h>            // Placeholder for DGL C++ API
#include <torch/torch.h>          // LibTorch for tensor operations
#include <nlohmann/json.hpp>      // For JSON parsing
#include <jellyfish/jaro_winkler.hpp> // Hypothetical C++ binding for jellyfish
#include <sklearn/model_selection.hpp> // Hypothetical C++ binding
#include <sklearn/feature_selection.hpp> // Hypothetical C++ binding

using json = nlohmann::json;
using namespace dgl;
using namespace torch;

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

    void load_meta_data() {
        std::cout << "[*] Loading meta data" << std::endl;
        std::ifstream f(dataset_path_ + "/meta_data/meta_data.json");
        json j;
        f >> j;
        meta_data_ = j;
        std::cout << "[+] Done loading meta data" << std::endl;
        std::cout << "[+] Number of sites: " << meta_data_.size() << std::endl;
    }

    void load_edge() {
        std::cout << "[*] Loading edges" << std::endl;
        std::string edge_file;
        if (edge_type_ == "reuse") {
            edge_file = dataset_path_ + "/graph/edges_reuse.json";
        } else if (edge_type_ == "sim") {
            if (!data_type_.empty()) {
                edge_file = dataset_path_ + "/graph/edges_" + data_type_ + ".json";
            } else {
                edge_file = dataset_path_ + "/graph/edges.json";
            }
        } else {
            throw std::runtime_error("Unknown edge type: " + edge_type_);
        }
        std::ifstream f(edge_file);
        json j;
        f >> j;
        edges_ = j;
        std::cout << "[+] Done loading edges" << std::endl;
        std::cout << "[+] Number of edges: " << edges_.size() << std::endl;
    }

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

    void load_country() {
        std::cout << "[*] Loading countries" << std::endl;
        std::string country_file;
        if (!data_type_.empty()) {
            country_file = dataset_path_ + "/graph/countries_" + data_type_ + ".json";
        } else {
            country_file = dataset_path_ + "/graph/countries.json";
        }
        std::ifstream f(country_file);
        json j;
        f >> j;
        countries_ = j;
        std::cout << "[+] Done loading countries" << std::endl;
        std::cout << "[+] Number of countries: " << countries_.size() << std::endl;
    }

    void load_category() {
        std::cout << "[*] Loading categories" << std::endl;
        std::string category_file;
        if (!data_type_.empty()) {
            category_file = dataset_path_ + "/graph/categories_" + data_type_ + ".json";
        } else {
            category_file = dataset_path_ + "/graph/categories.json";
        }
        std::ifstream f(category_file);
        json j;
        f >> j;
        categories_ = j;
        std::cout << "[+] Done loading categories" << std::endl;
        std::cout << "[+] Number of categories: " << categories_.size() << std::endl;
    }

    void set_node(bool save = true) {
        nodes_list_.clear();
        std::unordered_set<std::string> countries;
        std::unordered_set<std::string> categories;
        std::cout << "[*] Start Setting nodes" << std::endl;
        const std::set<std::string> required_keys = {"category", "country", "sl", "ip"};

        for (auto& [site, info] : meta_data_.items()) {
            bool ok = true;
            for (auto& key : required_keys) {
                if (!info.contains(key)) { ok = false; break; }
            }
            if (!ok) continue;
            json node;
            node["site"] = site;
            node["category"] = info["category"];
            node["country"] = info["country"];
            node["sl"] = info["sl"];
            node["ip"] = info["ip"];
            nodes_list_.push_back(node);
            countries.insert(info["country"].get<std::string>());
            categories.insert(info["category"].get<std::string>());
            std::cout << "[*] Setting nodes: " << nodes_list_.size() << " / " << meta_data_.size() << " | " << site << std::endl;
        }

        std::cout << "[+] Done setting nodes" << std::endl;
        std::cout << "[+] Number of nodes: " << nodes_list_.size() << std::endl;
        std::cout << "[+] Number of countries: " << countries.size() << std::endl;
        std::cout << "[+] Number of categories: " << categories.size() << std::endl;

        if (save) {
            std::ofstream f1(dataset_path_ + "/graph/nodes.json");
            json j1 = nodes_list_;
            f1 << j1.dump(4);
            std::cout << "[+] Saved nodes to " << dataset_path_ + "/graph/nodes.json" << std::endl;

            std::ofstream f2(dataset_path_ + "/graph/countries.json");
            json j2 = json::array();
            for (auto& c : countries) j2.push_back(c);
            f2 << j2.dump(4);
            std::cout << "[+] Saved countries to " << dataset_path_ + "/graph/countries.json" << std::endl;

            std::ofstream f3(dataset_path_ + "/graph/categories.json");
            json j3 = json::array();
            for (auto& c : categories) j3.push_back(c);
            f3 << j3.dump(4);
            std::cout << "[+] Saved categories to " << dataset_path_ + "/graph/categories.json" << std::endl;
        }
    }

    void set_edge(bool save = true) {
        std::cout << "[*] Start Setting edges" << std::endl;
        std::ofstream fout(dataset_path_ + "/graph/edges.txt", std::ios::app);

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
        // Saving of edges.json is commented out in Python version
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
    std::string dataset_path_;
    double edge_thv_;
    std::string splitting_type_;
    double valid_;
    double test_;
    int gnn_depth_;
    torch::Device device_;
    int random_seed_;
    int batch_size_;
    std::string edge_type_;
    int num_thv_;
    std::string data_type_;

    json meta_data_;
    json edges_;
    json nodes_;
    json countries_;
    json categories_;
    std::vector<json> nodes_list_;
    std::vector<json> edges_list_;
};
