import json
import re
# from openpyxl import Workbook
import sys
import os
sys.path.append(".")
# wb = Workbook()
# 激活工作表
# ws = wb.active
# 写入表头
# ws.append(['file','pre', 'rec', 'f1'])
def get_pres(response, ground_truth):
    acc = len([value for value in response if value in ground_truth])
    return acc/len(response) if len(response) > 0 else 0

def get_recs(response, ground_truth):
    acc = len([value for value in response if value in ground_truth])
    return acc/len(ground_truth) if len(ground_truth) > 0 else 0
def extract_numbers(text):
    numbers = re.findall('\d+', text)
    return numbers

def extract_substrings(input_string):
    pattern = re.compile(r'\[\d+\]')
    substrings = pattern.findall(input_string)
    extracted_string = ''.join(substrings)
    return extracted_string

def clean_response(sp: str):
    response = extract_substrings(sp.lower())
    if len(response) == 0:
        response = extract_numbers(sp.lower())
        new_response = []
        for res in response:
            if int(res)>20 or int(res)<=0:
                continue
            else:
                new_response.append(int(res))
        return new_response
    else:
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response
def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

file_name = "/home/gomall/work/GIT-NQ/results/Qwen3-8B-no-thinking/webap-k-sampling-.json"
file1 = open(file_name, "r", encoding="utf-8")
file2 = open("/home/gomall/work/GIT-NQ/results/Qwen3-8B-no-thinking/webap-k-sampling-selected5.json", "w", encoding="utf-8")
pres = []
recs = []
num = 0
for line in file1:
    js = json.loads(line)
    ground_truth_label = js["ground_truth_label"]
    driect = js["direct"]
    responses = []
    select_k = {}
    max_label = 3 if "trec" in file_name or "webap" in file_name  else 1
    all_selected_passages = js["all_selected_passages"][:6]
    selected_abels = js["selected_abels"]
    select_k = {}
    nums_len = {}
    for selected_passage in all_selected_passages:
        if len(selected_passage) in nums_len:
            nums_len[len(selected_passage)] += 1
        else:
            nums_len[len(selected_passage)] = 1
        for passage in selected_passage:
            if passage in select_k:
                select_k[passage] += 1
            else:
                select_k[passage] = 1

    select_k_re = sorted(select_k.items(),key = lambda x:x[1],reverse = True)  
    len_s = sorted(nums_len.items(),key = lambda x:x[1],reverse = True) 
    select_k_re = select_k_re[:len_s[0][0]]
    passages = []
    for passage, top in select_k_re:
        passages.append(passage)
    acc = 0 
    for passage in passages:
        if selected_abels[passage] == max_label:
            acc += 1
    ground_truth = []
    for index, label in enumerate(ground_truth_label):
        if label == max_label:
            ground_truth.append(index)
    if passages == []:
        pres.append(0)
    else:
        pres.append(acc/len(passages))
    if ground_truth != []:
        recs.append(acc/len(ground_truth))
    else:
        recs.append(0)
    js["k_smaple_selected"] = passages
    file2.write(json.dumps(js)+"\n")
    num += 1
print(num)
pre = 100*sum(pres)/len(pres)
rec = 100*sum(recs)/len(recs)
f1 = 2*pre*rec/(pre+rec)
print(file_name.split("/")[-1])
print("pre: ", pre)
print("rec: ", rec)
print("macro-f1: ", 2*pre*rec/(pre+rec))
# data.append({"file_name":file_name.split("/")[-1],"pre": pre, "rec":rec, "f1": f1})

        
    
    
    
        
    
                
                
        
    
    
    