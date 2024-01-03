import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# from typing_extensions import Concatenate

# User Interface using streamlit


with st.sidebar:
    st.title("Chat with Pdf file in your own way.")

# Uploading pdf file
load_dotenv()
pdf = st.file_uploader("Upload file", type='pdf')

if pdf is not None:
    pdfreader = PdfReader(pdf)

    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

# st.write(raw_text)

# splitting text

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,)

    texts = text_splitter.split_text(raw_text)

# st.write(len(texts))

    embeddings = OpenAIEmbeddings()

    document_search = FAISS.from_texts(texts, embeddings)

# st.write(document_search)

# Loading question & answering

    chain = load_qa_chain(OpenAI(), chain_type='stuff')

# Getting input from user

    query = st.text_input('Ask the question')
# st.write(query)

    if query:
        docs = document_search.similarity_search(query)
        output = chain.run(input_documents=docs, question=query)
        st.subheader('The output of your query is :')
        st.write(output)
