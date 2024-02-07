import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from time import sleep
import logging
import os

GREETING = "Hi, I'm the study helper from ZHAW LSFM. I can help you if you have questions regarding any kind of projects (semesterarbeit, bachelor, master etc). I can only answer questions if the information is given by ZHAW documents"

# Define the uni to study mapping here
uni_to_study = {
    "bachelor": ["Applied Digital Life Sciences", 
                 "Biomedizinische Labordiagnostik", 
                 "Biotechnologie", 
                 "Chemie", 
                 "Facility Management", 
                 "Lebensmitteltechnologie", 
                 "Umweltingenieurwesen"],
    "master": ["Circular Economy Management", 
               "Applied Computational Life Sciences", 
               "Vertiefung Chemistry for the Life Sciences", 
               "Vertiefung Food and Beverage Innovation", 
               "Vertiefung Pharmaceutical Biotechnology", 
               "Preneurship for Regenerative Food Systems", 
               "Real Estate & Facility Management", 
               "Umwelt und Natürliche Ressourcen"]
}

study_to_pdf = {
    "Applied Digital Life Sciences": "Bachelor",
    "Biomedizinische Labordiagnostik": "Bachelor",
    "Biotechnologie": "Bachelor",
    "Chemie": "Bachelor",
    "Facility Management": "Bachelor",
    "Lebensmitteltechnologie": "Bachelor",
    "Umweltingenieurwesen": "Bachelor",
    "Circular Economy Management": "Master",
    "Applied Computational Life Sciences": "Master",
    "Vertiefung Chemistry for the Life Sciences": "Master",
    "Vertiefung Food and Beverage Innovation": "Master",
    "Vertiefung Pharmaceutical Biotechnology": "Master",
    "Preneurship for Regenerative Food Systems": "Master",
    "Real Estate & Facility Management": "Master",
    "Umwelt und Natürliche Ressourcen": "Master"
}



def main():
    st.set_page_config(page_title="ZHAW student support", page_icon="🧬", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Ask anything about student project guidelines. 💬📄")
    st.info("Upload your PDF and start chatting to get insights.", icon="📃")

    load_dotenv()    
    #------------- styling
        # Custom CSS to make buttons larger and centered
    button_style = """
    <style>
        .stButton>button {
            width: 100%;
            height: 50px;
            border-radius: 25px;
            font-size: 20px;
        }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    # Uni Selection Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Bachelor"):
            st.session_state.uni_selection = "bachelor"
    with col2:
        if st.button("Master"):
            st.session_state.uni_selection = "master"
    # with col3:
    #     if st.button("hebamme"):
    #         st.session_state.uni_selection = "hebamme"


    # Display Study Options Based on Uni Selection
    if "uni_selection" in st.session_state:
        study_options = uni_to_study[st.session_state.uni_selection]
        st.write("Select your study field:")
        for option in study_options:
            if st.button(option, key=option):
                st.session_state.messages = [{"role": "assistant", "content": GREETING}]  # Reset chat history
                st.session_state.study_selection = option
                # Automatically load and process the PDF for the selected study
                pdf_path = study_to_pdf[option]  # Get the PDF path from the mapping
                # Assuming pdf_path is a local file path. For URLs, additional handling is required.

                processed_text = process_pdf(pdf_path)
                if processed_text:
                    st.success("PDF processed successfully. Ready to answer your questions.")
                    chunks = split_text_into_chunks(processed_text)
                    embeddings = OpenAIEmbeddings()
                    # Overwrite docsearch with the selected study's PDF
                    st.session_state.docsearch = FAISS.from_texts(chunks, embeddings)
                    st.session_state.agentDAO = load_qa_chain(OpenAI(), chain_type="stuff")
                else:
                    st.error("Failed to process PDF. Please try another file or select a different study.")
                    st.stop()

    # PDF Upload
    with st.sidebar:
        st.header("Upload PDF")
        pdf_file_sidebar = st.file_uploader("Choose a PDF file", type=["pdf"])
        if pdf_file_sidebar:
            pdf_file = pdf_file_sidebar
        else:
            pdf_file = None


    if pdf_file:
        processed_text = process_pdf(pdf_file)
        if processed_text:
            st.success("PDF processed successfully. Ready to answer your questions.")
            chunks = split_text_into_chunks(processed_text)
            embeddings = OpenAIEmbeddings()
            # Overwrite docsearch with the new PDF
            st.session_state.docsearch = FAISS.from_texts(chunks, embeddings)
            st.session_state.agentDAO = load_qa_chain(OpenAI(), chain_type="stuff")
        else:
            st.error("Failed to process PDF. Please try another file.")
            st.stop()

    if "messages" not in st.session_state.keys(): 
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]

    if prompt := st.chat_input("Your question about the PDF:"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    docs = st.session_state.docsearch.similarity_search(prompt)
                    response = st.session_state.agentDAO.run(input_documents=docs, question=prompt)

                    # Placeholder for streaming the response
                    response_placeholder = st.empty()

                    # Initialize an empty string to accumulate the response
                    current_text = ""

                    # Stream the response
                    for word in response.split():
                        current_text += word + ' '
                        response_placeholder.markdown(current_text)  # Using markdown for consistent styling
                        sleep(0.1)  # Adjust the sleep time to control the speed of streaming

                    # Add the full response to the message history
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)

            except Exception as e:
                logging.error(f"Error generating response: {e}")
                st.error("Failed to generate response.")    

#-----------------------------------------------------------------
#-----------------------------------------------------------------

def verify_user_session():
    # Placeholder for user verification logic
    # Implement your verification mechanism here
    return True

#-----------------------------------------------------------------
#-----------------------------------------------------------------
def process_pdf(pdf_input):
    text = ''
    try:
        if os.path.isdir(pdf_input):  # Check if pdf_input is a directory
            for filename in os.listdir(pdf_input):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(pdf_input, filename)
                    with open(file_path, 'rb') as pdf_file:
                        text += read_pdf_text(pdf_file) + ' '
        elif os.path.isfile(pdf_input) and pdf_input.endswith('.pdf'):  # Check if it's a PDF file
            with open(pdf_input, 'rb') as pdf_file:
                text += read_pdf_text(pdf_file)
        else:
            # st.error("Invalid input. Please provide a valid PDF file or a directory containing PDF files.")
            print("hi")
            return None
        return text
    except Exception as e:
        # st.error(f"Failed to process PDF due to: {e}")
        print("hi")
        return None

def read_pdf_text(pdf_file):
    """
    Extract text from a single PDF file.
    """
    pdf_text = ''
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            pdf_text += page_text + ' '
    return pdf_text

#-----------------------------------------------------------------
#-----------------------------------------------------------------

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
        length_function=len,
    )
    return text_splitter.split_text(text)


if __name__ == "__main__":
    main()

