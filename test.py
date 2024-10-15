
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import torch
from time import time
import pandas as pd
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.set_default_tensor_type(torch.cuda.HalfTensor)

def test_model(tokenizer, pipeline, message):
    """
    Perform a query
    print the result
    Args:
        tokenizer: the tokenizer
        pipeline: the pipeline
        message: the prompt
    Returns
        None
    """    
    time_start = time()
    sequences = pipeline(
        message,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,)
    time_end = time()
    total_time = f"{round(time_end-time_start, 3)} sec."
    
    question = sequences[0]['generated_text'][:len(message)]
    answer = sequences[0]['generated_text'][len(message):]
    
    return f"Question: {question}\nAnswer: {answer}\nTotal time: {total_time}"


model_path = "/mnt/data/thuandv/LIDA/models/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

query_pipeline = pipeline("text-generation",
                          model=model,
                          tokenizer=tokenizer,
                          torch_dtype=torch.bfloat16,
                          max_length=2048,
                          temperature=0.1,
                          do_sample=True,
                          repetition_penalty=1.2,
                        #   model_kwargs={"stop": [ "\n    Observation" ]},
                          device=1)
llm = HuggingFacePipeline(pipeline=query_pipeline)

df = pd.read_csv("./data/2019.csv")

# agent = create_csv_agent(llm=llm, path='./data/2019.csv', verbose=True, allow_dangerous_code=True,
#                          agent_executor_kwargs={"handle_parsing_errors": True},
#                          early_stopping_method="force",
#                          max_iterations=5,
#                          agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
#                          )

agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True,
                                      agent_executor_kwargs={"handle_parsing_errors": True},
                                      verbose=True,
                                      return_intermediate_steps=True
                                      )

query = "Plot scatter between score and GDP"
# resp = agent(query)
# print("====")
# print(resp["output"])
response = agent(query)
print("=====")
print(response)