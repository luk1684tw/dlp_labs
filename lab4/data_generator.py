import json
import string
import random

char_list = [''] + list(string.ascii_lowercase)
token_list = list()

with open('new_test.txt') as tokens:
    token_list = tokens.read().split()


json_list = list()
for token in token_list:
    swap_num = random.randint(1, 3)
    
    swap_index = random.sample(range(len(token)), swap_num)
    print(swap_index)
    new_token = list(token)
    for index in swap_index:
        swap_content = random.randint(0, 26)
        new_token[index] = char_list[swap_content]

    new_str = "".join(new_token)
    json_list.append({
        "input": [new_str],
        "target": token
    })


with open('new_test.json', 'w+') as json_file:
    json.dump(json_list, json_file)
