# COLREG-GPT

COLREG-GPT is an interactive application designed to assist mariners in refreshing their knowledge of the International Regulations for Preventing Collisions at Sea (COLREGs). Powered by gpt3.5 language models and vector search, COLREG-GPT answers queries related to maritime navigation rules to aid in the study and understanding of COLREGs.

## Features

- **Query Handling**: Input your questions regarding COLREGs and receive detailed explanations.
- **Document Reader**: Analyzes maritime documents to provide accurate information.
- **Vector Search**: Employs cosine similarity for relevant document retrieval.

## Architecture

- **Streamlit**: For the web app interface, allowing users to input queries and display answers.
- **OpenAI API**: Utilizes language models for understanding and generating text responses.
- **Langchain**: A framework for chaining language tasks such as document loading, splitting, and embedding.
- **Pinecone**: A vector database used to store and retrieve information based on semantic similarity.
- **Python**: The primary programming language used for developing the application.

### How It Works

1. **Document Processing**: The `PyPDFDirectoryLoader` from `langchain_community.document_loaders` loads PDF documents which are then chunked into manageable sizes using `RecursiveCharacterTextSplitter`.
2. **Vector Embedding**: Documents are converted into vector representations using `OpenAIEmbeddings`.
3. **Vector Indexing**: The vectors are indexed using Pinecone for efficient similarity searches.
4. **Query Resolution**: User queries are converted to vectors and matched against the document index to find relevant information.
5. **Response Generation**: The language model provides a cohesive and informative response to the query.

## Usage

To start using COLREG-GPT, simply run the Streamlit application (https://colreg-gpt.streamlit.app/) and enter your queries regarding the rules of the road at sea. The application will process your question and return a helpful response.

## Installation

Ensure you have Python installed and then install the necessary packages using:

```bash
pip install -r requirements.txt
