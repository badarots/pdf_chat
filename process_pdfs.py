import os
from dotenv import load_dotenv

import pandas as pd
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


def process_pdfs(directory: str) -> pd.DataFrame:
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    dict_chunks = [{'text': chunk.page_content, 'source': chunk.metadata.source} for chunk in chunks]

    vector_store = pd.DataFrame(dict_chunks)

    embeddings = genai.get_embeddings(
        model='models/text-embeddings-004',
        content=vector_store['text'].tolist(),
        task_type="RETRIEVAL_DOCUMENT"
    )
    # RETRIEVAL_QUERY

    vector_store['embeddings'] = embeddings['embeddings']

    return vector_store
