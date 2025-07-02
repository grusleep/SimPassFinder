#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <vector>

struct SitePassword {
    std::string site;
    std::string password;
};

struct UserData {
    int num;
    std::vector<SitePassword> data;
};

std::unordered_map<std::string, UserData> load_users(const std::string& filename) {
    std::ifstream file(filename);
    std::unordered_map<std::string, UserData> result;

    std::string line;
    std::string current_email;
    UserData current_data;

    while (std::getline(file, line)) {
        // 이메일 키 추출
        auto email_start = line.find("\"");
        auto email_end = line.find("\"", email_start + 1);
        if (email_start != std::string::npos && email_end != std::string::npos) {
            current_email = line.substr(email_start + 1, email_end - email_start - 1);
            current_data = UserData();
            continue;
        }

        // num 필드 추출
        auto num_pos = line.find("\"num\"");
        if (num_pos != std::string::npos) {
            auto colon = line.find(":", num_pos);
            int num = std::stoi(line.substr(colon + 1));
            current_data.num = num;
            continue;
        }

        // site/password 추출
        auto site_pos = line.find("\"site\"");
        if (site_pos != std::string::npos) {
            auto site_start = line.find("\"", site_pos + 6);
            auto site_end = line.find("\"", site_start + 1);
            std::string site = line.substr(site_start + 1, site_end - site_start - 1);

            std::getline(file, line); // 다음 줄: "password"
            auto pw_pos = line.find("\"password\"");
            auto pw_start = line.find("\"", pw_pos + 9);
            auto pw_end = line.find("\"", pw_start + 1);
            std::string password = line.substr(pw_start + 1, pw_end - pw_start - 1);

            current_data.data.push_back({site, password});
            continue;
        }

        // 객체 끝나면 저장
        if (line.find("},") != std::string::npos || line.find("}") != std::string::npos) {
            if (!current_email.empty()) {
                result[current_email] = current_data;
                current_email.clear();
                current_data = UserData();
            }
        }
    }

    return result;
}

void union_users(
    const std::unordered_map<std::string, UserData>& a,
    const std::unordered_map<std::string, UserData>& b,
    std::unordered_map<std::string, UserData>& result
) {
    result = a;
    for (const auto& [email, userdata] : b) {
        if (result.find(email) == result.end()) {
            result[email] = userdata;
        } else {
            result[email].data.insert(
                result[email].data.end(),
                userdata.data.begin(),
                userdata.data.end()
            );
        }
    }
}

void save_users(const std::string& filename, const std::unordered_map<std::string, UserData>& users) {
    std::ofstream out(filename);
    out << "{\n";
    bool first = true;
    for (const auto& [email, user] : users) {
        if (!first) out << ",\n";
        first = false;
        out << "  \"" << email << "\": {\n";
        out << "    \"num\": " << user.num << ",\n";
        out << "    \"data\": [\n";
        for (size_t i = 0; i < user.data.size(); ++i) {
            out << "      { \"site\": \"" << user.data[i].site
                << "\", \"password\": \"" << user.data[i].password << "\" }";
            if (i + 1 < user.data.size()) out << ",";
            out << "\n";
        }
        out << "    ]\n  }";
    }
    out << "\n}\n";
}

int main() {
    std::string user_dir = "../dataset/users/";
    std::string file1 = user_dir + "users_10000.json";
    std::string file2 = user_dir + "users_2000.json";
    std::string output = user_dir + "users_all.json";

    auto data1 = load_users(file1);
    std::cout << "[+] Loaded " << data1.size() << " users from " << file1 << "\n";

    auto data2 = load_users(file2);
    std::cout << "[+] Loaded " << data2.size() << " users from " << file2 << "\n";

    std::unordered_map<std::string, UserData> merged;
    union_users(data1, data2, merged);
    std::cout << "[+] Union complete. Total users: " << merged.size() << "\n";

    save_users(output, merged);
    std::cout << "[+] Saved to " << output << "\n";

    return 0;
}

