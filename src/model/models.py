
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.question_answering import load_qa_chain
import warnings

from vector_store import load_chunk_persist_pdf, get_existing_vectordb

# Suppress FutureWarning for deprecated parameter
warnings.filterwarnings(
    "ignore",
    # message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.",
    category=FutureWarning,
)


def get_prompt():
    # Prompt
    template = """
    You are comrade GPT, a true believer in the impending communist revolution. 
    Your task is to inspire new revolutionaries by teaching them leftist theory.
    Write in the style of Lenin.
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def create_agent_chain():
    # Local LLM
    ollama_llm = "llama3"
    prompt = get_prompt()
    llm = ChatOllama(model=ollama_llm)
    chain = prompt | load_qa_chain(llm, chain_type="stuff")
    return chain


# def get_llm_response(query):
#     vectordb = load_chunk_persist_pdf()
#     chain = create_agent_chain()
#     matching_docs = vectordb.similarity_search(query)
#     answer = chain.invoke(input_documents=matching_docs, question=query).content
#     return answer




# query = "What is the highest stage of capitalism?"
# print(get_llm_response(query))


# loader = PyPDFLoader("src\\data\\state-and-revolution.pdf")
# documents = loader.load_and_split()


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(documents)


# vectorstore = Chroma.from_documents(
#     documents=all_splits,
#     collection_name="rag-chroma",
#     persist_directory=chroma_dir,
#     embedding=get_embeddings(),
# )

# #vectorstore = Chroma(persist_directory=".\\chroma_db", embedding_function=embeddings)
# retriever = vectorstore.as_retriever()

# # Prompt
# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# # Local LLM
# ollama_llm = "llama3"
# model_local = ChatOllama(model=ollama_llm)

model_name = "llama3"
retriever = get_existing_vectordb().as_retriever()
#retriever = load_chunk_persist_pdf().as_retriever()
prompt = get_prompt()
llm = ChatOllama(model=model_name)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_llm_response(question):
    # Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("question")
    print(question)
    #answer = chain.invoke(question)
    stream = chain.stream(question)
    #print("answer")
    #print(answer)
    return stream #answer

#print(get_llm_response("What is exposure literature?"))

#print(chain.invoke("What role does theory play in the vanguard party?"))

# from langchain_core.runnables import RunnableParallel


# rag_chain_from_docs = (
#     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# rag_chain_with_source = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# ).assign(answer=rag_chain_from_docs)

# rag_chain_with_source.invoke("What is Task Decomposition")

# Wrap the input question in a dictionary
# input_data = {"context": [], "question": "How does the state affect the working class?"}
# answer = rag_chain_with_source.invoke(input_data["question"])
# #answer = rag_chain_from_docs.invoke("How does the state affect the working class?")
# print(answer)
