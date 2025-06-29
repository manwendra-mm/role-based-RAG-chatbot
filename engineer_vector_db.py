import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings

#Command line suggests to use the following import for HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate



# ---------- Configuration ----------
MARKDOWN_DIR = "./DS-RPC-01/data/engineering"     # Path to directory containing .md files
CHROMA_DB_DIR = "./engineer_chroma_db"         # Where vector DB will be persisted
EMBED_MODEL_NAME = "all-MiniLM-L6-v2" # Local embedding model (HF)
CHUNK_SIZE = 500                      # Number of characters per chunk
CHUNK_OVERLAP = 50                    # Overlap between chunks
ENGINEER_BREAK_PROGRAM = False

# ---------- Step 1: Load Markdown Files ----------
def load_markdown_files(directory):
    loader = DirectoryLoader(path=directory, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} markdown documents.")
    return documents

# ---------- Step 2: Split into Chunks ----------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# ---------- Step 3: Create Embeddings Locally ----------
def create_embeddings(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

# ---------- Step 4: Store Embeddings in Chroma DB ----------
def store_documents_in_chroma(chunks, embeddings, db_path):
    if os.path.exists(db_path):
        print("Chroma DB already exists. Loading it...")
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        print("Creating new Chroma DB and adding documents...")
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_path)
        vectordb.persist()
    return vectordb

# ---------- Step 5: Query the Vector Store ----------
def query_vector_store(vectordb, k):
    #Creating object- retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": k}
    )    
    
    
    model = OllamaLLM(model="llama3.2")

    template = """
    You are an expert in answering questions asked by engineers of a company. 
    Take context from the data provided below and answer the question asked.

    Here is the data:{data}

    Here is the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    '''
    while True:
        print("\n\n----------------------------------------")
        question = input("Ask your question (q to quit): ")
        if question.lower() == 'q':
            break
        reviews = retriever.invoke(question)
        result = chain.invoke({"data": reviews, "question": question})
        print(result)
    '''

    while True:
        print("\n\n----------------------------------------")
        question = input("Ask your question (q to quit): ")
        if question.lower() == 'q':
            ENGINEER_BREAK_PROGRAM = True
            break
        
        # Get top relevant documents
        documents = retriever.invoke(question)

        # Combine their page_content into a string, for passing to the template.
        context_text = "\n\n".join(
            f"- {doc.page_content}"
            for doc in documents
        )

        #Debug print to see the context_text
        print("Debug print: ", context_text)

        # Feed into the prompt and model chain
        result = chain.invoke({"data": context_text, "question": question}) #required part
        print(result)

    
'''
# ---------- Main Orchestration ----------
if __name__ == "__main__":
    # Load and process
    raw_docs = load_markdown_files(MARKDOWN_DIR)
    chunked_docs = split_documents(raw_docs)

    # Embed
    embeddings = create_embeddings(EMBED_MODEL_NAME)

    # Store
    vectordb = store_documents_in_chroma(chunked_docs, embeddings, CHROMA_DB_DIR)


    answer = query_vector_store(vectordb, k=3) # k tells the number of top results to return
    print("\nAnswer:", answer)
    
# Note: Here the model is only returning the top 3 results. Give the retrieved data to the LLM for further processing to get better answers.

'''