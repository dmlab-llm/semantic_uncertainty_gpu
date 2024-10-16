from transformers import pipeline
import os
import torch
import csv
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

model_id = "meta-llama/Meta-Llama-3.1-8B"
llama_2 = 'meta-llama/Llama-2-7b-chat-hf'

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

path = "/home/hjkim/semantic_uncertainty/arxiv100.csv"
dataset = pd.read_csv(path)


context = []
result = []
for i, data in enumerate(dataset):
    if i < 1:
        text = data["text"]
        context.append(text)
        whole_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for information extraction tasks. Please answer in one sentence from the context without any modifications. Keep the answer concise with no explanations. Print out the sentences in the paragraph without modification. Let's think step-by-step.<|eot_id|><|start_header_id|>user<|end_header_id|>
                Question: From the given text, what is the most important sentence that contains the meaning of the whole text? 
                Context: {text} 
                Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
        result.append(pipeline(whole_text))
print(result)

