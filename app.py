import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
import os
from dotenv import load_dotenv

# Set the page config
st.set_page_config(page_title="Langchain PDF Chatbot", layout="wide")

# Title for the web app
st.title("Langchain PDF Chatbot")

# Sidebar for API key input
st.sidebar.header("Configuration")

# #openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
# if not openai_api_key:
#     load_dotenv()
#     openai_api_key = ""
#     print("Environemnt loaded")
# else:
#     os.environ["OPENAI_API_KEY"] = openai_api_key
load_dotenv()

# File uploader
pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Function to read and process PDF
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + ' '  # Append space for separation
    return text

# Split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
        length_function=len,
    )
    return text_splitter.split_text(text)

# Query PDF
def query_pdf(query, docs, chain):
    return chain.run(input_documents=docs, question=query)

if pdf_file:
    with st.spinner("Processing PDF..."):
        processed_text = process_pdf(pdf_file)

        if processed_text:
            st.success("PDF processed successfully.")
            chunks = split_text_into_chunks(processed_text)
            
            # Initialize embeddings and vector store
            embeddings = OpenAIEmbeddings()
            docsearch = FAISS.from_texts(chunks, embeddings)
            
            # Load QA chain
            chain = load_qa_chain(OpenAI(), chain_type="stuff")

            # Query input
            query = st.text_input("Ask a question about the PDF content:")
            if query:
                with st.spinner("Searching for answers..."):
                    docs = docsearch.similarity_search(query)
                    answer = query_pdf(query, docs, chain)
                    st.write("Answer:", answer)
        else:
            st.error("Failed to process PDF.")
else:
    st.warning("Please upload a PDF file and enter your OpenAI API key to begin.")

#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------
    
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import logging
from time import sleep
# Assuming necessary imports for actual chat functionality, vector stores, embeddings, etc.

# Set the page config with an appropriate title and layout
st.set_page_config(page_title="Professional PDF Chat Application", layout="wide")

# Application title
st.title("Welcome to the Professional PDF Chat Application")

# Display an introductory message or instructions
st.info("Upload a PDF to query its content or start a chat for general inquiries.")

load_dotenv()

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Sidebar for PDF upload and other configurations
with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.header("Chat")
    start_chat = st.button("Start Chat Session")

# Dummy function to simulate agent response generation
def generate_response(prompt):
    # Implement your response generation logic here
    # This is a placeholder for demonstration
    return f"Here is a response to your question: {prompt}"

# Function to display chat messages
def display_chat():
    for message in st.session_state.messages:
        role = "user" if message["role"] == "user" else "bot"
        with st.container():
            st.text_area("", value=message["content"], key=message["content"], height=75, disabled=True, label=role)

# Chat functionality
if start_chat or "chat_session" in st.session_state:
    st.session_state.chat_session = True
    if "agent_loaded" not in st.session_state:
        # Placeholder for actual agent loading logic
        st.session_state.agent_loaded = True

    display_chat()

    prompt = st.text_input("Your question for the chat:", key="chat_input")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = generate_response(prompt)  # Placeholder for actual response generation
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()

# Function to process PDF and query content
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + ' '  # Append space for separation
    return text

# Handle PDF file upload and processing
if pdf_file and "chat_session" not in st.session_state:
    with st.spinner("Processing PDF..."):
        processed_text = process_pdf(pdf_file)
        if processed_text:
            st.success("PDF processed successfully.")
            # Placeholder for displaying processed text or further processing
            st.text(processed_text[:500])  # Show first 500 characters as a placeholder
        else:
            st.error("Failed to process PDF.")
elif not pdf_file and "chat_session" not in st.session_state:
    st.warning("Please upload a PDF file to begin or start a chat session.")
    