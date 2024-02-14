import os
import streamlit as st
import nltk
from llama_index.llms.replicate import Replicate
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings


nltk.download('stopwords')

os.environ["REPLICATE_API_TOKEN"] = st.secrets.replicate_key

st.header("Why Can't We All Just Get Along Chatbot")

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing Henry and Fry's chapter – hang tight! This should take 1-2 minutes."):
        llm = Replicate(model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5")
        Settings.llm = llm

        Settings.embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

        documents = SimpleDirectoryReader("/Users/es26698/Documents/Book_Club/book_chapter/").load_data()

        index = VectorStoreIndex.from_documents(documents,)

        return index

index = load_data()


query_engine = index.as_query_engine()
prompt = st.text_input('Enter your question here:', 'Your question...')

if st.button('Submit Query'):
	with st.spinner("Generating response... this may take a few minutes..."):
		resp = query_engine.query(prompt)
		st.write(resp.response)

