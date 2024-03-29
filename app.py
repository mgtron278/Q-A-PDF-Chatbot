import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


with st.sidebar:
    st.title('LLM Q&A Chat Bot')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('by Surya Guttikonda')

def main():
    st.header("Interact with your PDF")
    

# ... other code ...

    llm = OpenAI(temperature=0, api_key=os.environ.get("OPENAPI_API_KEY"))
    #upload a pdf file

    pdf = st.file_uploader("Upload PDF", type = 'pdf')

    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        store_name = pdf.name[:-4]

      #  {(if os.path.exists(f"{store_name}.pkl"):
       #     with open(f"{store_name}.pkl", "rb") as f:
        #        VectorStore = pickle.load(f)
         #   st.write("Embeddings loaded from disk")
        
        #else:
         #   embeddings = OpenAIEmbeddings()
          #  VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
           # with open(f"{store_name}.pkl", "wb") as f:
            #    pickle.dump(VectorStore,f)
            #st.write("Embedding computation")
        
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        query = st.text_input("Type down your Question!")
        st.write(query)
        #st.write(chunks)
        #st.write(text)
        if query:

            docs = VectorStore.similarity_search(query=query,  k=3)
            llm = OpenAI(temperature=0 , )
            chain = load_qa_chain(llm=llm, chain_type = "stuff")
            response = chain.run(input_documents=docs, question =query)
            st.write(response)

            #st.write(docs)

if __name__ == '__main__':
    main()
