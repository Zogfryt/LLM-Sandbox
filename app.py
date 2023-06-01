import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

with st.sidebar:
    st.title('My PDF chat')
    st.markdown("""
    ## About
    This app is purely experimental and if leaked shouldn't be used for any purpose.
    """)
    add_vertical_space(5)
    st.write('Made with ❤️')

def main():
    st.header("Hello")
    pdf = st.file_uploader("Upload your PDF", type=["pdf"])

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        st.write(chunks)

        #embeddings #tutaj nieco zbocze z kursu gdyż OpenAI jest drogie!!!
        embeddings = OpenAIEmbeddings()

        VectorStore = FAISS.from_texts(chunks, embeddings=embeddings)

if __name__ == '__main__':
    main()