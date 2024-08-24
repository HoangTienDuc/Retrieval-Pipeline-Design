from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader

data_source = "/develop/assignment/ollama-local-rag/docs"
pdf_loader = DirectoryLoader(data_source, glob="*.pdf", loader_cls=UnstructuredPDFLoader)
pdf_docs = pdf_loader.load()
print(pdf_docs)
