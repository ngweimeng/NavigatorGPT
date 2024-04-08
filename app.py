import streamlit as st
import openai
import langchain
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as langpinecone
from langchain.llms import OpenAI
from pinecone import Pinecone


from dotenv import load_dotenv
load_dotenv()

# read doc
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

doc = read_doc('documents/')

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

documents=chunk_data(docs=doc)
len(documents)

## Embedding technique of OpenAI
embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
embeddings

vectors=embeddings.embed_query("How are you")

# Create a Pinecone instance
pc = Pinecone(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment="gcp-starter"
)

index_name = "langchainvector"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536,  
        metric='cosine'
    )

index = langpinecone.from_documents(doc, embeddings, index_name=index_name)

## Cosine Similiarity Retrieve Results
def retrieve_query(query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results

from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

llm=OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")


## Search answers from VectorDB
def retrieve_answers(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response


# Define the main function that streamlit will run
def main():
    # Set the title of the web app
    st.title("COLREG-GPT!")

    # Create a text input box for user queries
    text_input = st.text_input("Ask all your ROR queries...") 
    
    # Define what happens when the "Ask Query" button is clicked
    if st.button("Ask Query"):
        # Make sure the input is not empty
        if len(text_input) > 0:
            # Display the user's query
            st.info("Your Query: " + text_input)
            try:
                # Retrieve the answer using the retrieval_answer function
                answer = retrieve_answers(text_input)
                # Show the answer to the user
                st.success(answer)
            except Exception as e:
                # If there's an error, show it to the user
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
