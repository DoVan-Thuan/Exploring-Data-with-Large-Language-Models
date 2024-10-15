from dotenv import load_dotenv
import pandas as pd
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
# print(llm.invoke("Write me a ballad about LangChain"))
df = pd.read_csv("./data/2019.csv")
agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True,
                                      agent_executor_kwargs={"handle_parsing_errors": True},
                                      verbose=True,
                                      return_intermediate_steps=True
                                      )

query = "Plot scatter plot between score and GDP"
resp = agent(query)
print("====")
print(resp["output"])
response = agent(query)
python_code = response['intermediate_steps'][-1][0].tool_input
output_code = python_code.strip("```python\n")
print(output_code)
# print(response['intermediate_steps'][-1][0].tool_input)

# fig = df.plot.scatter(x='GDP per capita', y='Score').get_figure()
# fig.savefig("scatter_plot.png")
# text = "```python df.plot.scatter(x='GDP per capita', y='Score')\n```"

# print(text.strip("```python"))