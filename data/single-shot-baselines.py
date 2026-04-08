import sys
import os
from tqdm import tqdm
import json
import logging
import argparse
import torch
import random
from vllm import LLM, SamplingParams
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
# sys.path.append(".")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" 
# os.environ["WORLD_SIZE"] = "2"

def load_source(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data
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


def main(file_type, llm, tokenizer, instruct, types, number):
    args = get_args(file_type)
    # path = 'level2_update_oral_dense/'+file_type+'/'
    path = '/GIT-NQ/results/Qwen3-8B-no-thinking/'
    args.outfile = path + types  + str(number) +'.json'
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
            rank = 0
            ress = []
            all_results = []
            model_out_label = []
            if number == 0:
                prompts_list = []
                for i in range(len(passages)):
                    pair = passages[i]
                    messages = get_direct_judge_point(question, instruct, pair)
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True)
                    prompts_list.append(prompt)
                outputs = llm.generate(prompts_list, sampling_params)
                assert len(outputs) == len(prompts_list)
                pre_model_out_label = [i for i in model_out_label]
                for output in outputs:
                    res = output.outputs[0].text
                    print(res)
                    if "</think>" in res:
                        res = res.split("</think>")[1]
                    ress.append(res)
                    if ": yes, " in res.lower():
                        model_out_label.append(1)
                        print(1)
                    elif " yes, " in res.lower():
                        model_out_label.append(1)
                        print(1)
                    elif " yes " in res.lower():
                        model_out_label.append(1)
                        print(1)
                    else:
                        print(0)
                        model_out_label.append(0)
            else:
                messages = get_direct_judge_list(question, instruct, passages)
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True)
                outputs = llm.generate([prompt,], sampling_params)
                ress = outputs[0].outputs[0].text
                all_results = []
                for result in outputs[0].outputs:
                    all_results.append(result.text)
                # print(all_results)
                # assert 1 > 2
            outfile.write(json.dumps({
                "question": question,
                "real_passage": passages,
                "prompts": prompt,
                "LLM_output_all": all_results,
                "ground_truth_label": labels,
                "model_out_label": model_out_label
            }) + "\n")
    except Exception as e:
        logging.exception(e)
    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()
if __name__ == '__main__':
    # sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop = ["<|eot_id|>"])
    # tokenizer = transformers.AutoTokenizer.from_pretrained("/home/gomall/models/llama3_8b_instruct/")
    # model = LLM(model="/home/gomall/models/llama3_8b_instruct/")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    tokenizer = transformers.AutoTokenizer.from_pretrained("/home/gomall/models/Qwen3-8B")
    model = LLM(model="/home/gomall/models/Qwen3-8B")

    instruct = """
    Please first generate an answer to the question based on the passages, and then output the passages you selected that have utility in answering the question. The format of the output is: 'Answer: [...], My selection:[[i],[j],...].'. Only response the answer and the selection results, do not say any word or explain. 
    """
    main("/data/NQrandom.json", model, tokenizer, instruct, "NQrandom-directly-list-answer-passages-", 1)
    # main("/data/trec_final.json", model, tokenizer, instruct, "trec-directly-list-answer-passages-", 1)
    # main("/data/webap_final.json", model, tokenizer, instruct, "webap-directly-list-answer-passages-", 1)
    instruct = """
    Please first generate an answer to the question based on the passage, and then output whether the passage has utility in generating the answer to the question or not. If the passage has utility in answering the question, output 'Answer: [...], My judgment: Yes, the passage has utility in answering the question.'; otherwise, output 'Answer: [...], My judgment: No, the passage has no utility in answering the question.'.
    """
    main("/data/NQrandom.json", model, tokenizer, instruct, "NQrandom-directly-point-answer-passage-", 0)
    # main("/data/trec_final.json", model, tokenizer, instruct, "trec-directly-point-answer-passage-", 0)
    # main("/data/webap_final.json", model, tokenizer, instruct, "webap-directly-point-answer-passage-", 0)
    
    instruct = """
    Please first generate what information is needed to answer the question based on the passages, and then output the passages you selected that have utility in answering the question.The format of the output is: 'Information is needed: [...], My selection:[[i],[j],...].'. Only response the answer and the selection results, do not say any word or explain. 
    """
    main("/data/NQrandom.json", model, tokenizer, instruct, "NQrandom-directly-list-answer-cot-passages3-", 1)
    # main("/data/trec_final.json", model, tokenizer, instruct, "trec-directly-list-answer-cot-passages3-", 1)
    # main("/data/webap_final.json", model, tokenizer, instruct, "webap-directly-list-answer-cot-passages3-", 1)

    instruct = """
    Please first generate what information is needed to answer the question based on the passage, and then output whether the passage has utility in generating the answer to the question or not. If the passage has utility in answering the question, output 'Information is needed: ..., My judgment: Yes, the passage has utility in answering the question.'; otherwise, output 'Information is needed: ..., My judgment: No, the passage has no utility in answering the question.'.
    """
    main("/data/NQrandom.json", model, tokenizer, instruct, "NQrandom-directly-point-answer-cot-passage3-more-", 0) 
    # main("/data/trec_final.json", model, tokenizer, instruct, "trec-directly-point-answer-cot-passage3-more-", 0) 
    # main("/data/webap_final.json", model, tokenizer, instruct, "webap-directly-point-answer-cot-passage3-more-", 0) 
 

    instruct = """
    Directly output the passages you selected that have utility in answering the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
    """
    main("/data/NQrandom.json", model, tokenizer, instruct, "NQrandom-directly-list-with-thinking", 1)
    # main("/data/trec_final.json", model, tokenizer, instruct, "trec-directly-list-", 1)
    # main("/data/webap_final.json", model, tokenizer, instruct, "webap-directly-list-", 1)

    instruct = """
    Directly output whether the passage has utility in answering the question or not. If the passage has utility in answering the question, output 'My judgment: Yes, the passage has utility in answering the question.'; otherwise, output 'My judgment: No, the passage has no utility in answering the question.'.
    """
    main("/data/NQrandom.json", model, tokenizer, instruct, "NQrandom-directly-point-", 0) 
    # main("/data/trec_final.json", model, tokenizer, instruct, "trec-directly-point-", 0) 
    # main("/data/webap_final.json", model, tokenizer, instruct, "webap-directly-point-", 0) 

    
    
    