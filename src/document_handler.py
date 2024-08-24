from langchain_community.document_loaders import DirectoryLoader, UnstructuredExcelLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import shutil
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

def retrieve_with_llm(llm_model: Ollama, embedding_model: OllamaEmbeddings, faiss_path: str, query: str, is_retrieve_with_llm=True):
    """
    This function retrieves documents relevant to a given query using either a Retrieval-Augmented Generation (RAG) approach or a simple vector search.

    Parameters:
    llm_model (str): The name or path of the language model to be used for RAG.
    embedding_model (str): The name or path of the embedding model to be used for vector search.
    faiss_path (str): The path to the FAISS index file.
    query (str): The query to be used for retrieving documents.
    is_retrieve_with_llm (bool, optional): A flag indicating whether to use RAG (default is True).

    Returns:
    tuple: If is_retrieve_with_llm is True, returns a tuple containing the response from the RAG chain and the context.
           If is_retrieve_with_llm is False, returns a tuple containing the relevant documents and None.
    """
    llm = Ollama(model=llm_model)
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    if is_retrieve_with_llm:
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}""")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": query})
        context = " ".join([doc.page_content for doc in response["context"]])
        return response, context
    else:
        retriever = vector.as_retriever()
        relevant_docs = retriever.get_relevant_documents(query)
        return relevant_docs, None

def create_index_for_documents(faiss_path: str, embedding_model: OllamaEmbeddings, chunks: list):
    """
    This function creates a FAISS index for a given list of chunks and saves it to a specified path.
    The function first clears out the existing database at the specified path, if it exists.
    Then, it initializes an embedding model and creates a new FAISS database from the chunks.
    Finally, it saves the FAISS database to the specified path and prints a success message.

    Parameters:
    faiss_path (str): The path where the FAISS index will be saved.
    embedding_model (str): The name or path of the embedding model to be used for creating the index.
    chunks (List[Document]): A list of Document objects representing the chunks to be indexed.

    Returns:
    None
    """
    # Clear out the database first.
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)

    # Initialize embedding model
    embeddings = OllamaEmbeddings(model=embedding_model) 

    # Create a new DB from the documents.
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(faiss_path)
    print(f"Saved {len(chunks)} chunks to {faiss_path}.")


def load_documents(data_source: str):
    """
    This function loads documents from a specified data source, currently supporting PDF and Excel files.
    It uses the LangChain Community's DirectoryLoader and UnstructuredPDFLoader/UnstructuredExcelLoader to load the documents.

    Parameters:
    data_source (str): The path to the directory containing the PDF and Excel files.

    Returns:
    List[Document]: A list of Document objects representing the loaded documents.
    """
    pdf_loader = DirectoryLoader(data_source, glob="*.pdf", loader_cls=UnstructuredPDFLoader)
    pdf_docs = pdf_loader.load()
    excel_loader = DirectoryLoader(data_source, glob="*.xlsx", loader_cls=UnstructuredExcelLoader)
    excel_docs = excel_loader.load()
    documents = pdf_docs + excel_docs
    print(f"Loaded {len(documents)} documents.")
    return documents



def split_into_chunks(documents: list):
    """
    This function splits a list of documents into smaller chunks using a RecursiveCharacterTextSplitter.
    Each chunk is a Document object containing a portion of the original document's content.

    Parameters:
    documents (List[Document]): A list of Document objects to be split into chunks.

    Returns:
    List[Document]: A list of Document objects representing the chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    # if chunks:
    #     document = chunks[0]
        # print(document.page_content)
        # print(document.metadata)
    return chunks