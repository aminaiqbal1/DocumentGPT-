from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
import streamlit as st
import os,uuid

st.set_page_config(page_title="Chat with your file",page_icon="👥")
st.header("DocumentGPT")

def main():

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf'],accept_multiple_files=True)
        process = st.button("Process")

    if process:
        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        st.write("file chunks created...")
        # create vetore stores
        vetorestore = get_vectorstore(text_chunks)
        st.write("Vectore Store Created...")
         # create conversation chain
        st.session_state.conversation = get_conversation_chain(vetorestore) 

        st.session_state.processComplete = True
    else:
        st.info("Please Upload files To Continue:")
    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)

# Function to get the input file and read the text from it.
def generate_unique_filename(directory, original_filename):
    if not os.path.exists(directory):
        os.mkdir(directory)

    base_name, file_extension = os.path.splitext(original_filename)
    unique_id = uuid.uuid4()
    unique_filename = f"{base_name}_{unique_id}{file_extension}"
    return os.path.join(directory, unique_filename)
        
def get_files_text(uploaded_files):
    Documents = []
    for uploaded_file in uploaded_files:            
        split_tup = os.path.splitext(uploaded_file.name)
        file_path = generate_unique_filename("data",uploaded_file.name)
        try:
            with open(file_path,'wb') as file:
                file.write(uploaded_file.read())
            file_extension = split_tup[1]
            if file_extension == ".pdf":
                Documents.extend(get_pdf_text(file_path))
        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            os.remove(file_path)

    return Documents

# Function to read PDF Files
def get_pdf_text(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def get_txt_text(file):
    loader = TextLoader(file.name)
    return loader.load()

def get_text_chunks(text):
    # spilit ito chuncks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len)
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    cohere_api_key = "cN7g47aoZHbuQBR9RmcfgeIbLL1tCfFilRl03sxq"
    api_key = "QIV6uxD-dQ_59DvlvJkzvzZWzdXBqIHPyDt2lop-YYc513O58x3oWQ"
    from langchain_cohere import CohereEmbeddings
    from langchain_qdrant import Qdrant

    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=cohere_api_key)
    url = "https://72bf1d04-d4fb-4db7-afee-04a29311f1ed.us-east4-0.gcp.cloud.qdrant.io:6333"
    qdrant = Qdrant.from_documents(
    text_chunks,
    embeddings,
    url=url,
    api_key=api_key,
    prefer_grpc=True,
    collection_name="my_documents",
    force_recreate=True,)
    return qdrant

def get_conversation_chain(vetorestore):
    llm = ChatGoogleGenerativeAI(convert_system_message_to_human=True,google_api_key=os.getenv("google_api_key"),model='gemini-1.5-flash',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory)
    return conversation_chain

def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

if __name__ == '__main__':
    main()