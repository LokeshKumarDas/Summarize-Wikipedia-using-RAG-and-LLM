# importing libraries
import wikipediaapi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import shutil
import os
import streamlit as st   


def get_wikipedia_summary_to_file(place_name):
    # Define a custom User-Agent
    custom_user_agent = f'SceneTravelGuideApp/1.0 ({email})'

    # Initialize Wikipedia API with User-Agent
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=custom_user_agent
    )

    # Fetch the page
    page = wiki_wiki.page(place_name)

    if not page.exists():
        return "Sorry, the requested page doesn't exist on Wikipedia."

    # Extract the complete content
    content = page.text

    return content

# Query-based summarization
def summarize_with_query(text, query):
    input_text = f"Given the Context based on the query give a discriptive answer of query'\n Query: '{query}'\n\nContext: {text}"
    summary = summarizer(input_text)
    return summary[0]['generated_text']

# Load the DistilBART model
model_name = 'sshleifer/distilbart-cnn-12-6'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create the pipeline
summarizer = pipeline(
    'text2text-generation',
    model=model,
    tokenizer=tokenizer,
    max_length=150,  
    min_length=80,
    num_beams=4,
    temperature=0.7
)

# title of our web-application
st.title('Summarize Wikipedia using RAG and LLM ')

# getting search query 
email = st.text_area("Enter the Email", height=68)
topic = st.text_area("Enter the topic", height=68)
query = st.text_area("Enter the query", height=68)

if st.button("Search"): 
  
    st.subheader('Answer:-')

    # search wiki and get the info about the topic
    text = get_wikipedia_summary_to_file(topic) 
    
    # converting text to doc
    document = Document(page_content=text)

    # splitting the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,  chunk_overlap=50)
    text_chunks = text_splitter.split_documents([document])  

    # loading embeddings
    embeddings = HuggingFaceEmbeddings()

    # if database in path already exist remove it
    if os.path.exists('db'):
        shutil.rmtree('db')

    # creating database
    db = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )

    # retrive top 5 related document
    retriever = db.as_retriever(search_kwargs = {'k':5})
    relevant_docs = retriever.get_relevant_documents(query)

    # add all the documnet
    combined_text = " "
    for i in relevant_docs:
        combined_text += i.page_content + " "

    # output
    output = summarize_with_query(combined_text, query)

    st.write(output)


