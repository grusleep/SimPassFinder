#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <sstream>

struct SitePassword {
    std::string site;
    std::string password;
};

struct UserData {
    int num = 0;
    std::vector<SitePassword> data;
};

using UserMap = std::unordered_map<std::string, UserData>;

// 간단한 JSON 정리 함수
std::string trim_json(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r,");
    size_t end = s.find_last_not_of(" \t\n\r,");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

UserMap load_users(const std::string& filename) {
    std::ifstream file(filename);
    std::string line, email;
    UserData current;
    UserMap result;

    while (std::getline(file, line)) {
        if (line.find(": {") != std::string::npos) {
            auto s = line.find("\"");
            auto e = line.find("\"", s + 1);
            email = line.substr(s + 1, e - s - 1);
            current = UserData();
        } else if (line.find("\"num\"") != std::string::npos) {
            auto colon = line.find(":");
            std::string num_str = trim_json(line.substr(colon + 1));
            current.num = std::stoi(num_str);
        } else if (line.find("\"site\"") != std::string::npos) {
            auto s = line.find("\"", line.find(":")) + 1;
            auto e = line.find("\"", s);
            std::string site = line.substr(s, e - s);

            std::getline(file, line); // password 줄
            auto ps = line.find("\"", line.find(":")) + 1;
            auto pe = line.find("\"", ps);
            std::string pw = line.substr(ps, pe - ps);

            current.data.push_back({site, pw});
        } else if (line.find("]") != std::string::npos && !email.empty()) {
            result[email] = current;
            email.clear();
            current = UserData();
        }
    }

    return result;
}

void merge_user_into(UserMap& base, const std::string& email, const UserData& new_data) {
    if (base.find(email) == base.end()) {
        base[email] = new_data;
    } else {
        base[email].data.insert(base[email].data.end(), new_data.data.begin(), new_data.data.end());
    }
}

void stream_merge_users(const std::string& filename, UserMap& base) {
    std::ifstream file(filename);
    std::string line, email;
    UserData current;

    while (std::getline(file, line)) {
        if (line.find(": {") != std::string::npos) {
            auto s = line.find("\"");
            auto e = line.find("\"", s + 1);
            email = line.substr(s + 1, e - s - 1);
            current = UserData();
        } else if (line.find("\"num\"") != std::string::npos) {
            auto colon = line.find(":");
            std::string num_str = trim_json(line.substr(colon + 1));
            current.num = std::stoi(num_str);
        } else if (line.find("\"site\"") != std::string::npos) {
            auto s = line.find("\"", line.find(":")) + 1;
            auto e = line.find("\"", s);
            std::string site = line.substr(s, e - s);

            std::getline(file, line); // password 줄
            auto ps = line.find("\"", line.find(":")) + 1;
            auto pe = line.find("\"", ps);
            std::string pw = line.substr(ps, pe - ps);

            current.data.push_back({site, pw});
        } else if (line.find("]") != std::string::npos && !email.empty()) {
            merge_user_into(base, email, current);
            email.clear();
            current = UserData();
        }
    }
}

void save_users(const std::string& filename, const UserMap& users) {
    std::ofstream out(filename);
    out << "{\n";
    bool first = true;
    for (const auto& [email, user] : users) {
        if (!first) out << ",\n";
        first = false;
        out << "    \"" << email << "\": {\n";
        out << "        \"num\": " << user.num << ",\n";
        out << "        \"data\": [\n";
        for (size_t i = 0; i < user.data.size(); ++i) {
            out << "            {\n";
            out << "                \"site\": \"" << user.data[i].site << "\",\n"
                << "                \"password\": \"" << user.data[i].password << "\"\n";
            out << "            }";
            if (i + 1 < user.data.size()) out << ",";
            out << "\n";
        }
        out << "        ]\n";
        out << "    }";
    }
    out << "\n}\n";
}

int main() {
    std::string dir = "../dataset/users/";
    std::string file1 = dir + "users_1_8.json";
    std::string file2 = dir + "users_9_10_11_12.json";
    std::string output = dir + "users_all.json";

    auto data = load_users(file1);
    std::cout << "[+] Loaded " << data.size() << " users from " << file1 << "\n";

    stream_merge_users(file2, data);
    std::cout << "[+] Stream merge complete. Total users: " << data.size() << "\n";

    save_users(output, data);
    std::cout << "[+] Saved to " << output << "\n";

    return 0;
}
