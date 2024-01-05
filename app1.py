import os
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

app = Flask(__name__)

# Loading openai api key
load_dotenv()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf = request.files['pdf']

        if pdf and pdf.filename.lower().endswith('.pdf'):
            pdfreader = PdfReader(pdf)

            raw_text = ''
            for i, page in enumerate(pdfreader.pages):
                content = page.extract_text()
                if content:
                    raw_text += content

            text_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            texts = text_splitter.split_text(raw_text)
            embeddings = OpenAIEmbeddings()
            document_search = FAISS.from_texts(texts, embeddings)
            chain = load_qa_chain(OpenAI(), chain_type='stuff')

            query = request.form['question']

            if query:
                docs = document_search.similarity_search(query)
                output = chain.run(input_documents=docs, question=query)
                return render_template('result.html', output=output)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
