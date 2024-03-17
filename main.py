import streamlit as st
import os
import numpy as np
import PyPDF2
from docx import Document
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import faiss

load_dotenv()

def main():
    st.title("Resume Best Search")
    jd=st.text_area("Job Description")
    files=st.file_uploader("Upload all your resumes", accept_multiple_files=True)
    submit=st.button("Upload")
    st.spinner("Uploading...")
    
#    print("Hello World !!")
    os.chdir("C:\Personal\Robin\GenAI\AICode\Resume")
    dir = os.getcwd()

    if submit:
        if files is not None:
            
            for file in files:
                if ".pdf" in file.name:
                    dirpath = dir + "\\" + file.name
                    text = read_pdf_content_to_text(dirpath)
                    text_splitter1 = CharacterTextSplitter(separator="\n",chunk_size=3000,chunk_overlap=100)
                    text_split1 = text_splitter1.split_text(text)
                    embedding1=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vector1=faiss.FAISS.from_texts(texts=text_split1,embedding=embedding1)
                    similarity=vector1.similarity_search_with_relevance_scores(jd)
                    result = file.name + " - " + str(round((similarity[0][1]*100),2)) + "% " + "matched"
                    st.subheader(result)

                if ".docx" in file.name:
                    dirpath = dir + "\\" + file.name
                    text = read_docx_content_to_text(dirpath)
                    text_splitter1 = CharacterTextSplitter(separator="\n",chunk_size=3000,chunk_overlap=100)
                    text_split1 = text_splitter1.split_text(text)
                    embedding1=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vector1=faiss.FAISS.from_texts(texts=text_split1,embedding=embedding1)
                    similarity=vector1.similarity_search_with_relevance_scores(jd)
                    result = file.name + " - " + str(round((similarity[0][1]*100),2)) + "% " + "matched"
                    st.subheader(result)                 
#                    st.subheader(text)

#def get_percentage_match(text):
#    model=genai.GenerativeModel("gemini-pro")
#    response=model.generate_content(text)
#    return response.text

def read_pdf_content_to_text(dirpath):
    with open(dirpath,"rb") as f:
        pdf = PyPDF2.PdfReader(f)
        text=""

        for i in range(0,len(pdf.pages)):
            selected_page = pdf.pages[i]
            text+=selected_page.extract_text()

        return text

def read_docx_content_to_text(dirpath):
    with open(dirpath,"rb") as f:
        doc = Document(f)
        text=""

        for i in doc.paragraphs:
            text+=i.text+"\n"

        return text

if __name__ == "__main__":
    main()
