import os
#from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import CSVLoader
from langchain.chains import RetrievalQAWithSourcesChain


from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI



f = open('..\\api_key.txt')
api_key = f.read()
os.environ['OPENAI_API_KEY'] = api_key

#llm = ChatOpenAI()


# 1. Load all CSVs from a folder
def load_csv_list_from_folder(folder_path):
    csv_df_list = []
    
    for filename in os.listdir(folder_path):
        print("filename: ", filename)
        if filename.endswith(".csv"):
            print("filename: ", os.path.join(folder_path, filename))
            df = pd.read_csv(os.path.join(folder_path, filename), encoding='latin1')
            csv_df_list.append(df)
    
    return csv_df_list

# 1. Load all CSVs from a folder
def load_csvs_from_folder(folder_path):
    csv_df = pd.DataFrame(columns=['Event ID'])
    
    for filename in os.listdir(folder_path):
        print("filename: ", filename)
        if filename.endswith(".csv"):
            print("filename: ", os.path.join(folder_path, filename))
            df = pd.read_csv(os.path.join(folder_path, filename), encoding='latin1')
            csv_df = pd.concat([csv_df, df], ignore_index=True)
    
    return csv_df


# 2. Create contract retriever
def create_csv_retriever_agent(folder_path):
    csv_df = load_csvs_from_folder(folder_path)
    ep_retriever = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0), csv_df, agent_type="tool-calling", 
        allow_dangerous_code=True, verbose=True
    )
    return ep_retriever
    