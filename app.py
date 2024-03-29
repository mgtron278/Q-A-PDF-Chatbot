import streamlit as st
import pickle  # Optional, for potential future caching
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Replace with transformers for embeddings
import transformers
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
        - [LangChain](https://python.langchain.com/) (partially)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('by Surya Guttikonda')


def main():
    st.header("Interact with your PDF")

    # Retrieve API key from Heroku environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable")

    pdf = st.file_uploader("Upload PDF", type='pdf')

    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Use transformers for embeddings (replace model name if needed)
        openai_model_name = "text-davinci-003"
        tokenizer = transformers.AutoTokenizer.from_pretrained(openai_model_name)
        model = transformers.AutoModel.from_pretrained(openai_model_name)

        def get_openai_embeddings(text):
            encoded_input = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                embeddings = model(**encoded_input).pooler_output
            return embeddings.squeeze(0)

        embeddings = [get_openai_embeddings(chunk) for chunk in chunks]
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        query = st.text_input("Type down your Question!")
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0, api_key=openai_api_key)  # Use retrieved API key
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__ == '__main__':
    main()
