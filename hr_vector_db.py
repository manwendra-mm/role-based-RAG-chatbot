#A vector is a database that's stored locally on our device. From where our model can access it and query data.
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

import os
import pandas as pd


DATA_DIRECTORY = "./DS-RPC-01/data/hr/hr_data.csv"
CHROMA_DB_DIR = "./chroma_lanchain_db"
HR_BREAK_PROGRAM = False #Initially set to false, will be set to true when user enters 'q' to quit the program.

# This function loads data, embeds it and stores it in a vector store.
def create_vector_store(): 
    #Loading Data
    df = pd.read_csv()
    #print(df.head())  #To check df is loaded correctly

    #Choosing Embedding Model
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    db_location = CHROMA_DB_DIR
    add_documents = not os.path.exists(db_location)


    if add_documents:
        documents = []
        ids = []

        for i, row in df.iterrows():
            # Build human-readable descriptive content
            page_content = (
                f"{row['full_name']} is a {row['role']} in the {row['department']} department "
                f"based in {row['location']}. "
                f"Their attendance percentage is {row['attendance_pct']}% and their performance rating is {row['performance_rating']}. "
                f"They have taken {row['leaves_taken']} leaves and have {row['leave_balance']} leave balance."
            )

            document = Document(
                page_content=page_content,
                metadata={
                    "employee_id": row["employee_id"],
                    "full_name": row["full_name"],
                    "role": row["role"],
                    "department": row["department"],
                    "location": row["location"],
                    "attendance_pct": row["attendance_pct"],
                    "performance_rating": row["performance_rating"],
                    "leaves_taken": row["leaves_taken"],
                    "leave_balance": row["leave_balance"],
                    "email": row["email"],
                    "manager_id": row["manager_id"],
                    "date_of_birth": row["date_of_birth"],
                    "date_of_joining": row["date_of_joining"],
                    "last_review_date": row["last_review_date"],
                    "salary": row["salary"],  # Optional: may choose to keep it out
                },
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)

    global vector_store #This object will be used in the hr_generate_response function, so its global
    vector_store = Chroma(
        collection_name = "hr_data",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    # Adding data to the vector store
    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)


########################################
# Filling the prompt template with data and question, then pass it to the LLM for an answer.
def hr_generate_response():
    
    # If db directory does not exist, create the vector store
    if not os.path.exists(CHROMA_DB_DIR):
        create_vector_store()

    hr_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}  # Number of documents to retrieve
    )

    
    model = OllamaLLM(model="llama3.2")

    template = """
    You are an expert in answering questions asked by HR of a company. 
    Each employee data provided from the HR data records has a unique employee ID of the format FINEMPXXXX where XXXX is a 4-digit number.
    Take context from the below data and answer the question asked.

    Here are the HR data records:{data}

    Here is the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model


    while True:
        print("\n\n----------------------------------------")
        question = input("Ask your question (q to quit): ")
        if question.lower() == 'q':
            HR_BREAK_PROGRAM = True
            break

        # Get top relevant documents
        documents = hr_retriever.invoke(question)

        # Combine the documents into the context_text string
        # (The format of this string can be changed... EWe can experiment with it to find a good one!)
        context_text = "\n\n".join(
            f"- {doc.page_content} (EmployeeID: {doc.metadata['employee_id']},Full Name , Manager ID:{doc.metadata['manager_id']}, Salary: {doc.metadata['salary']}, DOJ: {doc.metadata['date_of_joining']})"
            for doc in documents
        )
        #Debug print
        print(context_text)  #Success 

        # Feed into the prompt and model chain
        result = chain.invoke({"data": context_text, "question": question}) #required part
        print("\nAnswer:", result) 

'''
# MAIN, only for testing purposes
if __name__ == "__main__":
    hr_generate_response()
'''



