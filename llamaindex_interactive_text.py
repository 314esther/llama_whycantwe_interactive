import os
import streamlit as st
from llama_index.llms.replicate import Replicate
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings


os.environ["REPLICATE_API_TOKEN"] = "r8_YsEQrL7lwAwNCuWmyClDa1bh06TYArK4GDCjG"
'''
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"
)
Settings.llm = llm

Settings.embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")

documents = SimpleDirectoryReader("/Users/es26698/Documents/Book_Club/book_chapter/").load_data()

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()

resp = query_engine.query("How many children do Jack and Jill have?")
'''

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

'''
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating a response..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
'''

