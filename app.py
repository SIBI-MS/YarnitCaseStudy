import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage #schemas
from langchain_community.document_loaders import WebBaseLoader  #load content from url
from langchain.text_splitter import RecursiveCharacterTextSplitter #used make chunks using the contents from url
from langchain_community.vectorstores import Chroma #to store vectors created my using the above chunks
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from sentence_transformers import SentenceTransformer


