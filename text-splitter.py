from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("futureinternet-15-00179.pdf")

docs = loader.load()

def len_splitter(docs, chunk_size=200, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=''
    )
    return text_splitter.split_documents(docs)

def recursive_len_splitter(docs, chunk_size=200, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)


res = len_splitter(docs)

print(res[10].page_content)