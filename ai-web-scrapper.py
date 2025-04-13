import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

st.title('AI Web Scrapper')
template = '''
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.
Context: {context}
Question: {question}
Helpful Answer:
'''

llm = OllamaLLM(model='llama3.2')
embedings = OllamaEmbeddings(model='llama3.2')
vector_store = InMemoryVectorStore(embedings)


def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    return documents

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    data = text_splitter.split_documents(docs)
    return data

def index_docs(docs):
    vector_store.add_documents(docs)

def retrieve_query(query):
    result = vector_store.similarity_search(query)
    return result

def generate(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt|llm
    response = chain.invoke({'question': question, 'context': context})
    return response

url = st.text_input('Enter url:')
docs = load_page(url)
all_splits = split_text(docs)
index_docs(all_splits)

question = st.chat_input()

if question:
    st.chat_message('User').write(question)
    response = retrieve_query(question)
    context = '\n\n'.join([doc.page_content for doc in response])
    content = generate(question, context)
    st.chat_message('AI').write(content)
