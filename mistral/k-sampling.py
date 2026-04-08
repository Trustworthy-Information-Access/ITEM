import sys
import re
import os
sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"
from tqdm import tqdm
import json
import logging
import argparse
import torch
import random
# from utils.utils import load_source
from vllm import LLM, SamplingParams
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
def load_source(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data
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

def get_args(file_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=file_type)
    parser.add_argument('--outfile', type=str, default='')
    args = parser.parse_args()
    return args

def get_prefix_direct_judge_point(query):
    return [{'role': 'user',
             'content': "You are the utility judger, an intelligent assistant that can judge whether a passage has utility in answering the question or not."},
            {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
            {'role': 'user',
             'content': f"I will provide you with a passage. \n Judge whether the passage has utility in answering the question or not: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passage.'}]
def get_post_direct_judge_point(query, passage, instruct):
    if len(passage.split(" "))> 300:
        passage = ' '.join(passage.split(" ")[:300])
    return f"Question: {query}. \n Passage: {passage} \n\n The requirements for judging whether a passage has utility in answering the question are: The passage has utility in answering the question, meaning that the passage not only be relevant to the question, but also be useful in generating a correct, reasonable and perfect answer to the question. \n"+instruct
def get_direct_judge_point(question, instruct, passage):
    messages = get_prefix_direct_judge_point(question)
    messages.append({'role': 'user', 'content': get_post_direct_judge_point(question, passage, instruct)})
    return messages


def get_prefix_direct_judge_list(query, num):
    return [{'role': 'user',
             'content': "You are the utility judger, an intelligent assistant that can select the passages that have utility in answering the question."},
            {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nSelect the passages that have utility in answering the question: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
def get_post_direct_judge_list(query, instruct):
    return f"Question: {query}.\n\n The requirements for judging whether a passage has utility in answering the question are: The passage has utility in answering the question, meaning that the passage not only be relevant to the question, but also be useful in generating a correct, reasonable and perfect answer to the question. \n"+instruct
def get_direct_judge_list(question, instruct, passages):
    messages = get_prefix_direct_judge_list(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        if len(content.split(" "))> 300:
            content = ' '.join(content.split(" ")[:300])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list(question, instruct)})
    return messages


def main(file_type, llm, tokenizer, instruct, types, sampling_params):
    args = get_args(file_type)
    # path = 'level2_update_oral_dense/'+file_type+'/'
    path = '/home/gomall/work/GIT-NQ/results/Qwen3-8B-no-thinking/'
    args.outfile = path + types +'.json'
    if not os.path.exists(path):
        os.makedirs(path)
    begin = 0
    if os.path.exists(args.outfile):
        outfile = open(args.outfile, 'r', encoding='utf-8')
        for line in outfile.readlines():
            if line != "":
                begin += 1
        outfile.close()
        outfile = open(args.outfile, 'a', encoding='utf-8')
    else:
        outfile = open(args.outfile, 'w', encoding='utf-8')
    all_data = load_source(args.source)
    num_output = 0
    try:
        for sample in tqdm(all_data[begin:len(all_data)], desc="Filename: %s" % args.outfile):
            passages = sample["passage"]           
            question = sample["question"]
            labels = sample["ground_truth_label"]
            max_label = 3 if "trec" in file_type or "webap" in file_type  else 1
            sample_passages, ress = [], []
            i_round = 0
            direct = {}
            selected_abels = {}
            for i in range(len(passages)):
                direct[passages[i]] = i
                selected_abels[passages[i]] = labels[i]
            all_selected_passages = []
            messages = get_direct_judge_list(question, instruct, passages)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False)
            outputs = llm.generate([prompt,], sampling_params)
            sample_passages.append(passages)
            res = outputs[0].outputs[0].text
            ress.append(res)
            temp_passages = passages.copy()
            response = clean_response(res)
            if isinstance(response,list):
                response = [int(x)-1 for x in response]
                response = remove_duplicate(response)
            else:
                response = [int(x)-1 for x in response.split()]
                response = remove_duplicate(response)
            temp = []
            for i in response:
                if i >= len(passages):
                    continue
                temp.append(passages[i])
            all_selected_passages.append(temp) 
            while (i_round <= 6):
                i_round += 1
                print("-----------------------------------the "+ str(i_round)+" round-------------------------------")     
                random.shuffle(temp_passages)
                sample_passages.append(temp_passages)
                messages = get_direct_judge_list(question, instruct, temp_passages)
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False)
                outputs = llm.generate([prompt,], sampling_params)
                res = outputs[0].outputs[0].text
                ress.append(res)
                response = clean_response(res)
                if isinstance(response,list):
                    response = [int(x)-1 for x in response]
                    response = remove_duplicate(response)
                else:
                    response = [int(x)-1 for x in response.split()]
                    response = remove_duplicate(response)
                temp = []
                for i in response:
                    if i >= len(temp_passages):
                        continue
                    temp.append(temp_passages[i])
                all_selected_passages.append(temp)
            outfile.write(json.dumps({
                "question": sample["question"],
                "sample_passages": sample_passages,
                "prompts_exmper": messages,
                "output_all": ress,
                "ground_truth_label": labels,
                "direct": direct,
                "all_selected_passages": all_selected_passages,
                "selected_abels": selected_abels
                
            }) + "\n")
    except Exception as e:
        logging.exception(e)
    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()
if __name__ == '__main__':
    # sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop = ["<|eot_id|>"])
    # tokenizer = transformers.AutoTokenizer.from_pretrained("/home/gomall/models/llama3_8b_instruct/")
    # llm = LLM(model="/home/gomall/models/llama3_8b_instruct/")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    tokenizer = transformers.AutoTokenizer.from_pretrained("/home/gomall/models/Qwen3-8B")
    llm = LLM(model="/home/gomall/models/Qwen3-8B")
    instruct = """
    Please first generate an answer to the question based on the passages, and then output the passages you selected that have utility in answering the question. The format of the output is: 'Answer: [...], My selection:[[i],[j],...].'. Only response the answer and the selection results, do not say any word or explain. 
    """
    # main("/home/gomall/work/data/NQrandom.json", llm, tokenizer, instruct, "llama3.1-8b-k-sampling-", sampling_params)
    # main("/home/gomall/work/data/MSMrandom.json", llm, tokenizer, instruct, "llama3.1-8b-k-sampling-", sampling_params)    
    main("/home/gomall/work/data/NQrandom.json", llm, tokenizer, instruct, "NQrandom-k-sampling-", sampling_params)
    main("/home/gomall/work/data/trec_final.json", llm, tokenizer, instruct, "trec-k-sampling-", sampling_params)
    main("/home/gomall/work/data/webap_final.json", llm, tokenizer, instruct, "webap-k-sampling-", sampling_params)