# AI Pill Bot Using Langchain 

## Project Overview

This repository contains the code and resources for the **AI Pill Bot**. The project demonstrates how to load a PDF document, embed its content, store the embeddings in a FAISS vector store, and implement a retriever to answer user queries based on the document content. The project also integrates an interactive user interface using Gradio and performs evaluation with Langchain's tracing and evaluation tools.

## Key Features

- **PDF Document Loading**: Extract and process PDF content using Langchain.
- **Embedding**: Convert document content into vector embeddings.
- **FAISS Indexing**: Store and search over embeddings efficiently using FAISS.
- **Retriever**: Implement a retriever to fetch relevant information based on user queries.
- **Gradio UI**: Add an interactive user interface for real-time question answering.
- **Evaluation**: Evaluate the performance of the model using Langchain's evaluation tools.

## Steps Followed in the Project

1. **Library Installation**:
    - Install necessary libraries such as Langchain, FAISS, PyMuPDF, and Gradio.

2. **PDF Loader**:
    - Load the PDF file using `PyMuPDFLoader` and extract its content.

3. **Text Splitting**:
    - Split the PDF content into smaller chunks using `RecursiveCharacterTextSplitter`.

4. **Embedding**:
    - Convert the text chunks into vector embeddings using the `OpenAIEmbeddings` model.

5. **FAISS Index Creation**:
    - Store the generated embeddings in a FAISS index for fast similarity search.

6. **Retriever**:
    - Implement a retriever to query the FAISS index based on user input.

7. **Question-Answering (RAG)**:
    - Create a RAG model with a prompt template to structure queries and context for LLM responses.

8. **Gradio Interface**:
    - Add a Gradio UI to interactively ask questions and retrieve answers.

9. **Evaluation**:
    - Use Langchain's tracing and evaluation tools to assess the model's performance.

## How to Run This Project in Google Colab

To run this project in Google Colab, follow these steps:

1. Open the Colab notebook or upload the `.ipynb` file into your Colab environment.
   
2. Install the necessary libraries:
    ```bash
    !pip install langchain openai faiss-cpu PyMuPDF gradio
    ```

3. Execute each cell sequentially to follow the project steps.

## Code Example

Here is a brief code example for embedding and FAISS indexing:

- **PDF Loading**:
    ```python
    from langchain.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader('path_to_pdf.pdf')
    data = loader.load()
    ```

- **Embedding and FAISS Indexing**:
    ```python
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = FAISS.from_documents(documents=splits, embedding=embedding_model)
    ```

- **Gradio Interface**:
    ```python
    import gradio as gr
    def predict(message, history):
        return rag_chain.invoke({"input": message})
    
    gr.ChatInterface(predict).launch()
    ```

## Prerequisites

- A Google account for using Google Colab.
- Basic understanding of Python and large language models.
- Required Python packages: Langchain, FAISS, PyMuPDF, Gradio.

## File Structure

```plaintext
langchain-week1-project/
│
├── SS__Langchain_Week1_Project.ipynb  # Colab notebook with project code and explanations
└── README.md                          # Project overview and setup instructions
