import os
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key is None:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

genai.configure(api_key=api_key)


def process_pdfs(directory: str) -> pd.DataFrame:
    # Primeiro extraímos o texto dos PDFs
    print("Extraíndo textos dos pdf...")
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No documents found in {directory}")

    print(f"{len(documents)} páginas processadas")

    # Fragmentamos os textos
    print("Fragmentando textos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Fragmentos gerados: {len(chunks)}")

    dict_chunks = [{'text': chunk.page_content, 'source': chunk.metadata["source"]} for chunk in chunks]

    vector_store = pd.DataFrame(dict_chunks)

    # Calculamos seus embeddings
    print("Gerando embeddings...")
    embeddings = genai.embed_content(
        model='models/text-embedding-004',
        content=vector_store['text'].tolist(),
        task_type="RETRIEVAL_DOCUMENT"
    )
    # RETRIEVAL_QUERY

    vector_store['embeddings'] = embeddings['embedding']
    return vector_store


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


# Encontramos os documentos mais similares à pergunta
def get_most_similar_documents(question: str, vector_store: pd.DataFrame, top_n=5):
    query_embedding = genai.embed_content(
        model='models/text-embedding-004',
        content=[question],
        task_type="RETRIEVAL_QUERY"
    )['embedding'][0]

    similarity = vector_store['embeddings'].apply(lambda x: cosine_similarity(query_embedding, x))
    indexes = similarity.sort_values(ascending=False).head(top_n).index

    return vector_store.iloc[indexes][['text', 'source']]


# Formatamos o contexto e a lista de referências a partir dos documentos mais similares
def format_context(chunks: pd.DataFrame) -> tuple:
    context = ""
    sources = set()
    i = 1
    for _, row in chunks.iterrows():
        context += f"<source id={i}>{row.text}</source>"
        sources.add(row.source)
        i += 1

    references = ""
    for i, source in enumerate(sources, 1):
        references += f"[{i}] {source}\n"

    return context, references


def generate_context(question: str, vector_store: pd.DataFrame, top_n=5):
    similar_documents = get_most_similar_documents(question, vector_store, top_n=top_n)

    return format_context(similar_documents)


# Set up the model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# Usamos few-shot learning para ensinar o modelo a responder perguntas com base nos documentos

system_instruction = """Você é um assistente para tarefas de perguntas e respostas. \
Use os trechos de contexto dentro da tag XML <context> para responder à pergunta dentro da tag XML <question>.
Não adicione informações externas, e se não souber a resposta com as informações do contexto, \
diga apenas que não pode responder à pergunta com os documentos que tem acesso.
Responda a pergunta na mesma lingua que foi feita, na dúvia responda em português brasileiro.

Exemplos:
###
Pergunta:
<context>Alura é uma escola de tecnologia que oferece cursos online e presenciais.
Que o ferece diversos cursos de tecnologia como Python, Javascript, C++, IA e muito mais.
===
Alura e Google se uniram para te ajudar a mergulhar em IA com o Google Gemini e impulsionar o seu crescimento.

Mais de 1 milhão de pessoas já participaram de imersões da Alura. Chegou a sua vez de se desenvolver com a mão na massa!

E a melhor parte: ao final da Imersão você vai ganhar um certificado Alura + Google e pode receber premiação pelo seu projeto.</context>
<question>Quais os cursos que a Alura oferece?</question>

Resposta:
Alura oferece cursos de tecnologia como Python, Javascript, C++, IA e muito mais, e recentemente se uniu ao Google para oferecer imersões em IA.

###
Pergunta:
<context>This comprehensive review paper offers a detailed
examination of the progression of RAG paradigms, encompassing the Naive RAG, the Advanced RAG, and the Modular RAG.
It meticulously scrutinizes the tripartite foundation of RAG
frameworks, which includes the retrieval, the generation and the
augmentation techniques. The paper highlights the state-of-the-
art technologies embedded in each of these critical components,
providing a profound understanding of the advancements in RAG
systems.
====
Large pre-trained language models have been shown to store factual knowledge
in their parameters, and achieve state-of-the-art results when fine-tuned on down stream NLP tasks. However, their ability to access and precisely manipulate
knowledge is still limited, and hence on knowledge-intensive tasks, their perfor mance lags behind task-specific architectures.</context>

<question>Quais os tipos de RAGs?</question>

Resposta:
Há três tipos de RAGs:
* **Naive RAG:** A metodologia inicial que se tornou popular logo após o ChatGPT. Segue um processo tradicional de indexação, recuperação e geração.
* **Advanced RAG:**  Desenvolvido para superar as limitações do Naive RAG, focando em melhorar a eficiência e o desempenho.
* **Modular RAG:**  Aborda as deficiências do Naive RAG, dividindo o processo RAG em módulos interconectados para maior flexibilidade e adaptabilidade.
"""

if __name__ == "__main__":

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        system_instruction=system_instruction,
        safety_settings=safety_settings
    )

    question_template = "<context>{context}</context>\n\n<question>{question}</question>"

    messages = []

    vector_store = process_pdfs("docs")


    while True:
        question = input("\nVocê: ")

        context, references = generate_context(question, vector_store)

        # Adicionamos o contexto do corpo da mensagem enviada ao modelo
        messages.append({'role':'user', 'parts': [question_template.format(context=context, question=question)]})
        response = model.generate_content(messages)

        response_content = response.text
        messages.append({'role': 'model', 'parts': [response_content]})

        print("\nPdf Chat: ", f"{response_content}\n\n Referências:\n{references}")
