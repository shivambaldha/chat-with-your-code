# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os, time
import gc
import re
import glob
import uuid
import subprocess
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
import streamlit as st

from llama_index.core import Settings
# from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings


open_ai_api_key = os.getenv("API_KEY")

# setting up the llm
@st.cache_resource
def load_llm(model_name = "gpt-4o", key = ""):
    if key != "":
        api_key = key
    else:
        api_key = open_ai_api_key

    # Initialize OpenAI LLM with the provided API key and model name
    llm=OpenAI(model=model_name, api_key = api_key)
    return llm


# utility functions
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def clone_repo(repo_url):
    return subprocess.run(["git", "clone", repo_url], check=True, text=True, capture_output=True)


def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

def parse_docs_by_file_types(ext, language, input_dir_path):
    
    files = glob.glob(f"{input_dir_path}/**/*{ext}", recursive=True)
    
    if len(files) > 0:
        loader = SimpleDirectoryReader(
            input_dir=input_dir_path, required_exts=[ext], recursive=True
        )
        docs = loader.load_data()
        
        if ext == ".md":
            parser = MarkdownNodeParser()
        else:

            parser = CodeSplitter(
                    language=language,
                    chunk_lines=40,  # lines per chunk
                    chunk_lines_overlap=5,  # lines overlap between chunks
                    max_chars=1500,  # max chars per chunk
                )

        return parser.get_nodes_from_documents(docs)
    else:
        return []


# create an qdrant collection and return an index
def create_index(nodes, client):
    unique_collection_id = uuid.uuid4()
    collection_name = f"chat_with_docs_{unique_collection_id}"
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )
    return index

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


with st.sidebar:

    open_ai_key = st.text_input("Enter your Open AI Key", type="password")
    api_key_button = st.button("Load OPEN AI Key",icon=":material/key:", use_container_width=True)
    if api_key_button and open_ai_key == "":
        st.error("Please enter your Open AI Key")
    elif api_key_button:
        st.info("Open AI Key Loaded successfully")

    # Input for GitHub URL
    github_url = st.text_input("GitHub Repository URL")

    # Button to load and process the GitHub repository
    process_button = st.button("Load Repo",icon=":material/mood:", use_container_width=True)
    if process_button and github_url == "":
        st.error("Enter GitHub URL")

    message_container = st.empty()  # Placeholder for dynamic messages

    if process_button and github_url:
        owner, repo = parse_github_url(github_url)
        if validate_owner_repo(owner, repo):
            with st.status(f"Loading {repo} repository by {owner}..."):
                try:
                    input_dir_path = f"./{repo}"
                    
                    if not os.path.exists(input_dir_path):
                        subprocess.run(["git", "clone", github_url], check=True, text=True, capture_output=True)

                    if os.path.exists(input_dir_path):
                        file_types = {
                            ".md": "markdown",
                            ".py": "python",
                            ".ipynb": "python",
                            ".js": "javascript",
                            ".ts": "typescript"
                        }

                        nodes = []
                        for ext, language in file_types.items():
                            print(language)
                            nodes += parse_docs_by_file_types(ext, language, input_dir_path)
                    else:    
                        st.error('Error occurred while cloning the repository, carefully check the url')
                        st.stop()
                    
                    # setting up the embedding model
                    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")                 
                    try:
                        index = create_index(nodes)
                    except:
                        index = VectorStoreIndex(nodes=nodes)

                    # ====== Setup a query engine ======
                    Settings.llm = load_llm(model_name="gpt-4o", key = open_ai_key)
                    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
                    
                    # ====== Customise prompt template ======
                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    if nodes:
                        message_container.success("Data loaded successfully!!")
                    else:
                        message_container.write(
                            "No data found, check if the repository is not empty!"
                        )
                    st.session_state.query_engine = query_engine

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop()

                st.success("Ready to Chat!")
        else:
            st.error('Invalid owner or repository')
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with your code! </>")
    st.markdown("```Welcome to the chat server with your code, Here we only support .py, .ipynb, .md, .js and .ts files```")

with col2:
    st.button("Clear ↺", on_click=reset_chat)


# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # context = st.session_state.context
        query_engine = st.session_state.query_engine

        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx
        st.download_button(label="Copy", data=full_response, file_name="response.txt", mime="text/plain")


    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

