import os
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import openai

from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
vector_store_not_created_created= False

#0 create dictionary of meta data 
def create_metadata():
    meta_dict = {
        "Aryahi medical report.pdf": {
        "name": "Aryahi",
        "type": "Blood report",
        "year": "2017",
        "contract_id": 1   
        },
        "B1-105-Agreement.pdf": {
        "name": "Pragati",
        "type": "Rent Agreement",
        "year": "2018",
        "contract_id": 2   
        },
        "B2-1403-agreement.pdf": {
        "name": "Jigar",
        "type": "Rent Agreement",
        "year": "2023",
        "contract_id": 3  
        }
    }
    return meta_dict



# 1. Load all PDFs from a folder
def load_pdfs_from_folder(folder_path):
    docs = []
    m_dict = create_metadata()
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            loaded_docs = loader.load()
            file_dict = m_dict[filename]
            print (str(file_dict))
            for doc in loaded_docs:
                doc.metadata['source'] = filename  # Add filename as metadata
                doc.metadata["name"] = file_dict['name']  # Add filename as metadata
                doc.metadata["type"] = file_dict["type"]  # Add filename as metadata
                doc.metadata["year"] = file_dict["year"]  # Add filename as metadata
                doc.metadata["contract_id"] = file_dict["contract_id"]  # Add filename as metadata
                #doc.page_content  =   doc.page_content + "\n\n" + str(file_dict)  
                #print(doc.metadata)
            docs.extend(loaded_docs)
    return docs


# 2. Split text into chunks
def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            #updated_chunk = chunk + "\n\n" + "name=" + doc.metadata["name"] + "\n" + "type=" + doc.metadata["type"] + "\n" + "year=" + doc.metadata["year"] 
            new_doc = type(doc)(page_content=chunk, metadata=doc.metadata.copy())
            d_name = doc.metadata.get("name", "Unknown")  
            d_type = doc.metadata.get("type", "Unknown") 
            contract_id = doc.metadata.get("contract_id", "Unknown") 
            xxx = " XXX"
            new_doc.page_content = f"(name:{d_name}, type:{d_type}, contract_id:{contract_id}, {xxx})\n{new_doc.page_content}"
            #print(new_doc.page_content)
            split_docs.append(new_doc)
    return split_docs
    
# 3. Create vectorstore
def create_vectorstore(documents):
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    vector_dir = "D:\\ai-ml-data\\chroma_store1"
    print("here...1 vector_dir: ", vector_dir)
    # Load existing vector store
    vectorstore = Chroma(
        persist_directory=vector_dir,
        embedding_function=embedding_model
    )
    print("here...2")
    
    collection = vectorstore._collection  # Accessing Chroma's internal collection
    count = collection.count()
    print("here...3", count)

    
    if (count == 0):
        vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=vector_dir)
    return vectorstore

# 3. Create contract retriever
def create_contract_retriever():
    
    folder_path = "D:\\ai-ml-data\\pdf-input"
    f = open('..\\api_key.txt')
    api_key = f.read()
    os.environ['OPENAI_API_KEY'] = api_key
    #openai.api_key = os.getenv('OPENAI_API_KEY')
    
    print("Loading documents...")
    docs = load_pdfs_from_folder(folder_path)
    print(f"Loaded {len(docs)} documents.")
    
    print("Splitting documents...")
    split_documents = split_docs(docs)
    
    print("Creating vectorstore...")
    vectorstore = create_vectorstore(split_documents)
    retriever = vectorstore.as_retriever()
    return retriever

    
# 4. Create Retrieval QA chain
def create_qa_chain(vectorstore):
    llm = OpenAI(temperature=0.5)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa_chain

# 5. Main function
def chat_with_pdfs(folder_path):
    f = open('..\\api_key.txt')
    api_key = f.read()
    os.environ['OPENAI_API_KEY'] = api_key
    #openai.api_key = os.getenv('OPENAI_API_KEY')
    
    print("Loading documents...")
    docs = load_pdfs_from_folder(folder_path)
    print(f"Loaded {len(docs)} documents.")
    
    print("Splitting documents...")
    split_documents = split_docs(docs)
    
    print("Creating vectorstore...")
    vectorstore = create_vectorstore(split_documents)

    
    print("Setting up QA chain...")
    qa_chain = create_qa_chain(vectorstore)
    
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = qa_chain.invoke(query)
        print("Answer:", answer)


