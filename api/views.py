from rest_framework.decorators import api_view
from rest_framework.response import Response

from langchain.chains import MultiRetrievalQAChain
from langchain.chains.router import MultiPromptChain
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import os
from agents.PdfAgent import create_contract_retriever
from agents.CsvAgent import create_csv_retriever_agent

from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import pandas as pd



def processQuestion(question):
    csv_folder_path = "D:\\ai-ml-data\\csv-input"
       
    f = open('..\\api_key.txt')
    api_key = f.read()
    os.environ['OPENAI_API_KEY'] = api_key

    llm = OpenAI()
    contract_retriever = create_contract_retriever()
    csv_agent = create_csv_retriever_agent(csv_folder_path)

    intent_prompt = PromptTemplate.from_template(
        "Classify this question into one of the following categories: Contract, Events, Quotation, Supplier, HR.\n\nQuestion: {question}\nCategory:"
    )
    intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

    query = "How many suppliers are invited for event 6"
    query = "Who is licensor for cotract_id 3"
    query = "What is rent amount for contract_id 2"
    query = "How many suppliers are invited for event 6"
    query = "Provide name of suppliers invited for event 6"
    query = "For how many events, supplier Bombay Tools Supplying Agency has given bids"
    query = "What is rent amount for cotract_id 3"
    query = "Provide count of events along with type of events with buyer names"
    query = "Provide count of events along with type of events with buyer names with sum of baseline prices"
    query = "Provide supplier wise count of events where they have accepted the invite, sort by event count descending"
    query = "Provide top 10 suppliers who have accepted the maximum count of events, sort by event count descending"
    query = "Provide name of suppliers who have accepted the invite but have not placed any bids, sort by event count descending"
    query = "Provide unique name of suppliers who have accepted the invite but have not placed any bids"
    query = "Provide count of suppliers who have accepted the invite but have not placed any bids"

    category = intent_chain.run(query).strip()

    print("category: ", category)

    retriever = {
        "Contract": contract_retriever,
        "Quotation": csv_agent,
        "Supplier": csv_agent,
        "Events": csv_agent,
        "HR": contract_retriever
    }[category]

        
    if "Contract" in category or "HR" in category:    
        retriever = contract_retriever
        #qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        
        result = qa.invoke(query)
    else:
        result = csv_agent.invoke(query)

    print(result)
    return result;


@api_view(['GET'])
def hello_world(request):
    defQuery = "Provide count of suppliers who have accepted the invite but have not placed any bids"
    name = request.GET.get('name', 'World')
    query = request.GET.get('query', defQuery)
    ans = processQuestion(query)
    return Response({"message": f"Hello, {name}! {ans}"})



