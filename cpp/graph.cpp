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
    std::vector<SitePassword> data;
};

using UserMap = std::unordered_map<std::string, UserData>;

// 간단한 JSON 정리 함수
std::string trim_json(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r,");
    size_t end = s.find_last_not_of(" \t\n\r,");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// 문자열 s에서 unescaped "의 위치를 찾는 함수
size_t find_unescaped_quote(const std::string& s, size_t start = 0) {
    size_t i = start;
    while (i < s.size()) {
        if (s[i] == '\\') {
            // \가 나오면 다음 문자(escape 문자) 건너뜀
            ++i;
        } else if (s[i] == '"') {
            return i;
        }
        ++i;
    }
    return std::string::npos;
}

UserMap load_users(const std::string& filename) {
    std::ifstream file(filename);
    std::string line, email;
    UserData current;
    UserMap result;

    while (std::getline(file, line)) {
        if (line.find(": [") != std::string::npos) {
            auto s = line.find("\"");
            auto e = find_unescaped_quote(line, s + 1);
            email = line.substr(s + 1, e - s - 1);
            current = UserData();
        } else if (line.find("\"site\"") != std::string::npos) {
            auto s = line.find("\"", line.find(":")) + 1;
            auto e = find_unescaped_quote(line, s);
            std::string site = line.substr(s, e - s);

            std::getline(file, line); // password 줄
            auto ps = line.find("\"", line.find(":")) + 1;
            auto pe = find_unescaped_quote(line, ps);
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

// 사용자 하나를 저장 (append 모드)
void save_user(std::ofstream& out, const std::string& email, const UserData& user, bool& first) {
    if (!first) out << ",\n";
    first = false;
    size_t data_size = user.data.size();

    out << "    \"" << email << "\": [\n";
    for (size_t i = 0; i < data_size; ++i) {
        out << "        {\n";
        out << "            \"site\": \"" << user.data[i].site << "\",\n"
            << "            \"password\": \"" << user.data[i].password << "\"\n";
        out << "        }";
        if (i + 1 < data_size) out << ",";
        out << "\n";
    }
    out << "    ]";
}

// stream 병합 및 즉시 저장
void stream_merge_and_save(const std::string& filename, UserMap& base, const std::string& output) {
    std::ifstream file(filename);
    std::ofstream out(output);
    out << "{\n";
    std::string line, email;
    UserData current;
    bool first = true;

    while (std::getline(file, line)) {
        if (line.find(": [") != std::string::npos) {
            auto s = line.find("\"");
            auto e = find_unescaped_quote(line, s + 1);
            email = line.substr(s + 1, e - s - 1);
            current = UserData();
        } else if (line.find("\"site\"") != std::string::npos) {
            auto s = line.find("\"", line.find(":")) + 1;
            auto e = find_unescaped_quote(line, s);
            std::string site = line.substr(s, e - s);
            std::getline(file, line); // password
            auto ps = line.find("\"", line.find(":")) + 1;
            auto pe = find_unescaped_quote(line, ps);
            std::string pw = line.substr(ps, pe - ps);
            current.data.push_back({site, pw});
        } else if (line.find("]") != std::string::npos && !email.empty()) {
            merge_user_into(base, email, current);
            save_user(out, email, base[email], first);
            base.erase(email); // base에서 제거
            email.clear();
            current = UserData();
        }
    }

    // base에 있었지만 새 파일에 없었던 사용자 추가 저장
    for (const auto& [email, user] : base) {
        if (user.data.empty()) continue; // 이미 저장했으면 비워뒀다는 전제
        save_user(out, email, user, first);
        base[email].data.clear();
    }

    out << "\n}\n";
}

int main() {
    std::string dir = "../dataset/users/";
    std::string file1 = dir + "users_9_10_11_12.json";
    std::string file2 = dir + "users_1_8.json";
    std::string output = dir + "users_all.json";

    auto data = load_users(file1);
    std::cout << "[+] Loaded " << data.size() << " users from " << file1 << "\n";

    stream_merge_and_save(file2, data, output);
    std::cout << "[+] Stream merge and save to " << output << "\n";

    return 0;
}