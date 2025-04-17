import re
import jellyfish



def check_pwd_security_level(pwd: list):
    level = 0
    if len(pwd) >= 12:
        level += 1
    if re.search(r'[0-9]', pwd):
        level += 1
    if re.search(r'[a-z]', pwd):
        level += 1
    if re.search(r'[A-Z]', pwd):
        level += 1
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', pwd):
        level += 1
    return level
    


if __name__ == "__main__":
    pwd1 = "test_1234!"
    pwd2 = "test_7234!"
    
    print(f"({pwd1}, {pwd2}): {jellyfish.jaro_similarity(pwd1, pwd2)}")
    