# import os
# import streamlit as st
# import pickle
# import time
# import re
# import langchain
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# import html

# from dotenv import load_dotenv
# load_dotenv()
# groq_api_key=os.getenv("GROQ_API_KEY")

# # Set page config
# st.set_page_config(layout="wide", page_title="Document QA System")

# # Custom CSS for better styling
# st.markdown("""
#     <style>
#     .stApp {
#         max-width: 100%;
#     }
#     .output-container {
#         background-color: #f0f2f6;
#         padding: 20px;
#         border-radius: 10px;
#         margin: 10px 0;
#     }
#     .source-container {
#         background-color: #e6e9ef;
#         padding: 15px;
#         border-radius: 8px;
#         margin-top: 10px;
#         font-size: 0.9em;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = None
# if 'chain' not in st.session_state:
#     st.session_state.chain = None

# # Prompt template
# qa_prompt_template = """You are an experienced news editior named joe, and you are given the following pieces of context. Give answer in a compact and precise manner .If you don't know the answer, just say "I don't know" - don't try to make up an answer.

# Context:
# {summaries}

# Question: {question}

# Please provide a detailed answer based only on the context provided above. If the context doesn't contain enough information to answer the question fully, please state that explicitly.

# Answer:"""

# QA_PROMPT = PromptTemplate(
#     template=qa_prompt_template,
#     input_variables=["summaries", "question"]
# )

# def clean_text(text):
#     """Clean and format the text output"""
#     # Remove HTML tags
#     text = html.unescape(text)
#     text = re.sub(r'<[^>]+>', '', text)
    
#     # Remove special characters but keep basic punctuation
#     text = re.sub(r'[^\w\s.,!?-]', '', text)
    
#     # Remove extra whitespace
#     text = ' '.join(text.split())
    
#     return text

# def initialize_llm():
#     """Initialize the LLM with Groq"""
#     return ChatGroq(
#         model_name="llama3-8b-8192",
#         temperature=0.3,
#         max_tokens=1000,
#     )

# def process_urls(urls):
#     """Process URLs and create vector store"""
#     try:
#         with st.spinner('Loading and processing URLs...'):
#             # Load URLs
#             loader = UnstructuredURLLoader(urls=[url.strip() for url in urls if url.strip()])
#             data = loader.load()
            
#             # Split text
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 length_function=len,
#                 separators=["\n\n", "\n", ".", "!", "?", " ", ""]
#             )
#             docs = text_splitter.split_documents(data)
            
#             # Create embeddings
#             embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",
#                 model_kwargs={'device': 'cpu'}
#             )
            
#             # Create vector store
#             vector_store = FAISS.from_documents(docs, embeddings)
            
#             return vector_store
#     except Exception as e:
#         st.error(f"Error processing URLs: {str(e)}")
#         return None

# def setup_qa_chain(vector_store):
#     """Set up the QA chain"""
#     if vector_store is None:
#         return None
        
#     llm = initialize_llm()
    
#     return RetrievalQAWithSourcesChain.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vector_store.as_retriever(
#             search_type="mmr",
#             search_kwargs={
#                 "k": 4,
#                 "fetch_k": 6,
#                 "lambda_mult": 0.7
#             }
#         ),
#         chain_type_kwargs={
#             "prompt": QA_PROMPT,
#             "document_separator": "\n\n",
#         },
#         return_source_documents=True
#     )

# def main():
#     # Create two columns
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("Document Sources")
        
#         # URL inputs
#         urls = []
#         for i in range(3):
#             url = st.text_input(f"URL {i+1}", key=f"url_{i}")
#             if url:
#                 urls.append(url)
        
#         # Process button
#         if st.button("Process URLs"):
#             if urls:
#                 st.session_state.vector_store = process_urls(urls)
#                 if st.session_state.vector_store is not None:
#                     st.session_state.chain = setup_qa_chain(st.session_state.vector_store)
#                     st.success("URLs processed successfully!")
#             else:
#                 st.warning("Please enter at least one URL")
    
#     with col2:
#         st.header("Ask Questions")
        
#         # Question input
#         question = st.text_input("Enter your question")
        
#         # Answer button
#         if st.button("Get Answer"):
#             if not st.session_state.chain:
#                 st.warning("Please process URLs first")
#                 return
                
#             if not question:
#                 st.warning("Please enter a question")
#                 return
            
#             try:
#                 # Create a placeholder for the loading animation
#                 with st.spinner("🤔 Thinking..."):
#                     # Get answer
#                     result = st.session_state.chain(
#                         {"question": question},
#                         return_only_outputs=True
#                     )
                    
#                     # Clean and display the answer
#                     answer = clean_text(result.get('answer', 'No answer found'))
#                     sources = clean_text(result.get('sources', 'No sources found'))
                    
#                     # Display results in styled containers
#                     st.markdown("<div class='output-container'>", unsafe_allow_html=True)
#                     st.markdown("### Answer")
#                     st.write(answer)
#                     st.markdown("</div>", unsafe_allow_html=True)
                    
#                     st.markdown("<div class='source-container'>", unsafe_allow_html=True)
#                     st.markdown("### Sources")
#                     st.write(sources)
#                     st.markdown("</div>", unsafe_allow_html=True)
                    
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()



# import os
# import streamlit as st
# import pickle
# import time
# import re
# import langchain
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# import html

# from dotenv import load_dotenv
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# # Set page config
# st.set_page_config(layout="wide", page_title="Document QA System")

# # Custom CSS for better styling
# st.markdown("""
#     <style>
#     .stApp {
#         max-width: 100%;
#     }
#     .output-container {
#         background-color: #f0f2f6;
#         padding: 20px;
#         border-radius: 10px;
#         margin-top: 10px;
#     }
#     .source-container {
#         background-color: #e6e9ef;
#         padding: 15px;
#         border-radius: 8px;
#         margin-top: 10px;
#         font-size: 0.9em;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = None
# if 'chain' not in st.session_state:
#     st.session_state.chain = None

# # Prompt template
# qa_prompt_template = """You are an experienced news editior named joe, and you are given the following pieces of context. Give answer in a compact and precise manner, and stick to information in the context only .If you don't know the answer, just say "I don't know" - don't try to make up an answer.

# Context:
# {summaries}

# Question: {question}

# Please provide a detailed answer based only on the context provided above. If the context doesn't contain enough information to answer the question fully, please state that explicitly.

# Answer:"""

# QA_PROMPT = PromptTemplate(
#     template=qa_prompt_template,
#     input_variables=["summaries", "question"]
# )

# def clean_text(text):
#     """Clean and format the text output"""
#     # Remove HTML tags
#     text = html.unescape(text)
#     text = re.sub(r'<[^>]+>', '', text)
    
#     # Remove special characters but keep basic punctuation
#     text = re.sub(r'[^\w\s.,!?-]', '', text)
    
#     # Remove extra whitespace
#     text = ' '.join(text.split())
    
#     return text

# def initialize_llm():
#     """Initialize the LLM with Groq"""
#     return ChatGroq(
#         model_name="llama3-8b-8192",
#         temperature=0.3,
#         max_tokens=200,
#     )

# def process_urls(urls):
#     """Process URLs and create vector store with animated steps"""
#     try:
#         # Step-by-step loading animation
#         with st.spinner("Gathering information from URLs..."):
#             time.sleep(1)
#             loader = UnstructuredURLLoader(urls=[url.strip() for url in urls if url.strip()])
#             data = loader.load()
            
#         with st.spinner("Splitting data..."):
#             time.sleep(1)
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 length_function=len,
#                 separators=["\n\n", "\n", ".", "!", "?", " ", ""]
#             )
#             docs = text_splitter.split_documents(data)
        
#         with st.spinner("Creating embeddings..."):
#             time.sleep(1)
#             embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",
#                 model_kwargs={'device': 'cpu'}
#             )
        
#         with st.spinner("Building vector database..."):
#             time.sleep(1)
#             vector_store = FAISS.from_documents(docs, embeddings)
        
#         st.success("✅ Vector database is ready!")
#         return vector_store
#     except Exception as e:
#         st.error(f"Error processing URLs: {str(e)}")
#         return None

# def setup_qa_chain(vector_store):
#     """Set up the QA chain"""
#     if vector_store is None:
#         return None
        
#     llm = initialize_llm()
    
#     return RetrievalQAWithSourcesChain.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vector_store.as_retriever(
#             search_type="mmr",
#             search_kwargs={
#                 "k": 4,
#                 "fetch_k": 6,
#                 "lambda_mult": 0.7
#             }
#         ),
#         chain_type_kwargs={
#             "prompt": QA_PROMPT,
#             "document_separator": "\n\n",
#         },
#         return_source_documents=True
#     )

# def main():
#     # Create two columns
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("Document Sources")
        
#         # URL inputs
#         urls = []
#         for i in range(3):
#             url = st.text_input(f"URL {i+1}", key=f"url_{i}")
#             if url:
#                 urls.append(url)
        
#         # Process button
#         if st.button("Process URLs"):
#             if urls:
#                 st.session_state.vector_store = process_urls(urls)
#                 if st.session_state.vector_store is not None:
#                     st.session_state.chain = setup_qa_chain(st.session_state.vector_store)
#                     st.success("URLs processed successfully!")
#             else:
#                 st.warning("Please enter at least one URL")
    
#     with col2:
#         st.header("Ask Questions")
        
#         # Question input
#         question = st.text_input("Enter your question")
        
#         # Answer button
#         if st.button("Get Answer"):
#             # Clear previous results
#             st.session_state.pop('answer', None)
#             st.session_state.pop('sources', None)
            
#             if not st.session_state.chain:
#                 st.warning("Please process URLs first")
#                 return
                
#             if not question:
#                 st.warning("Please enter a question")
#                 return
            
#             try:
#                 # Display loading spinner for answer retrieval
#                 with st.spinner("🤔 Thinking..."):
#                     result = st.session_state.chain(
#                         {"question": question},
#                         return_only_outputs=True
#                     )
                    
#                     # Clean and display the answer
#                     answer = clean_text(result.get('answer', 'No answer found'))
#                     sources = clean_text(result.get('sources', 'No sources found'))
                    
#                     # Display results in styled containers
#                     # st.markdown("<div class='output-container'>", unsafe_allow_html=True)
#                     st.markdown("### Answer")
#                     st.write(answer)
#                     st.markdown("</div>", unsafe_allow_html=True)
                    
#                     # st.markdown("<div class='source-container'>", unsafe_allow_html=True)
#                     st.markdown("### Sources")
#                     st.write(sources)
#                     st.markdown("</div>", unsafe_allow_html=True)
                    
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()


import os
import streamlit as st
import pickle
import time
import re
import langchain
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import html

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set page config
st.set_page_config(layout="wide", page_title="Document QA System")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    .output-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .source-container {
        background-color: #e6e9ef;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []

# Prompt template
qa_prompt_template = """You are an experienced news editior named joe, and you are given the following pieces of context. Give answer in a compact and precise manner, and stick to information in the context only. If you don't know the answer, just say "I don't know" - don't try to make up an answer.

Context:
{summaries}

Question: {question}

Please provide a detailed answer based only on the context provided above. If the context doesn't contain enough information to answer the question fully, please state that explicitly.

Answer:"""

QA_PROMPT = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["summaries", "question"]
)

def clean_text(text):
    """Clean and format the text output"""
    # Remove HTML tags
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def initialize_llm():
    """Initialize the LLM with Groq"""
    return ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=1000,
    )

def process_urls(urls):
    """Process URLs and create vector store"""
    try:
        # Save processed URLs in session state
        st.session_state.processed_urls = urls
        
        with st.spinner("Gathering information from URLs..."):
            loader = UnstructuredURLLoader(urls=[url.strip() for url in urls if url.strip()])
            data = loader.load()
            
            # Add source URLs to metadata
            for doc, url in zip(data, urls):
                doc.metadata['source'] = url
        
        with st.spinner("Processing text..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", " ", ""]
            )
            docs = text_splitter.split_documents(data)
            
            # Ensure source URL is preserved in split documents
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = urls[0]  # Fallback to first URL if source is lost
        
        with st.spinner("Creating embeddings..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            vector_store = FAISS.from_documents(docs, embeddings)
            
        st.success("✅ Processing complete!")
        return vector_store
    
    except Exception as e:
        st.error(f"Error processing URLs: {str(e)}")
        return None

def setup_qa_chain(vector_store):
    """Set up the QA chain"""
    if vector_store is None:
        return None
        
    llm = initialize_llm()
    
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 6,
                "lambda_mult": 0.7
            }
        ),
        chain_type_kwargs={
            "prompt": QA_PROMPT,
        },
        return_source_documents=True
    )
    
    return chain

def main():
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 2.5em;'>
            Newz Cut – The Smart Cut for Smart Readers.
        </h1>
        """, 
        unsafe_allow_html=True
    )
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Document Sources")
        
        # URL inputs
        urls = []
        for i in range(3):
            url = st.text_input(f"URL {i+1}", key=f"url_{i}")
            if url:
                urls.append(url)
        
        # Process button
        if st.button("Process URLs"):
            if urls:
                st.session_state.vector_store = process_urls(urls)
                if st.session_state.vector_store is not None:
                    st.session_state.chain = setup_qa_chain(st.session_state.vector_store)
            else:
                st.warning("Please enter at least one URL")
    
    with col2:
        st.header("Ask Questions")
        
        # Question input
        question = st.text_input("Enter your question")
        
        # Answer button
        if st.button("Get Answer"):
            if not st.session_state.chain:
                st.warning("Please process URLs first")
                return
                
            if not question:
                st.warning("Please enter a question")
                return
            
            try:
                with st.spinner("🤔 Thinking..."):
                    # Get answer and sources
                    result = st.session_state.chain({
                        "question": question
                    })
                    
                    # Display answer
                    if 'answer' in result:
                        st.markdown("### Answer")
                        st.markdown(result['answer'], 
                                  unsafe_allow_html=True)
                    
                    # Display sources
                    if 'source_documents' in result:
                        st.markdown("### Sources")
                        # st.markdown("<div class='source-container'>", unsafe_allow_html=True)
                        
                        # Create a set of unique source URLs
                        source_urls = set()
                        for doc in result['source_documents']:
                            if 'source' in doc.metadata:
                                source_urls.add(doc.metadata['source'])
                        
                        # Display each unique source URL
                        for url in source_urls:
                            st.markdown(f"- [{url}]({url})")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("No source information available.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Full error:", exc_info=True)

if __name__ == "__main__":
    main()