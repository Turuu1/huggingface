from langchain.document_loaders import PyPDFLoader, TextLoader
loaders = [
    # Duplicate documents on purpose - messy data
    # PyPDFLoader("D:\Downloads\\Jour-2022.pdf"),
    # PyPDFLoader("D:\Downloads\\test.pdf")
    TextLoader("D:/annotation/journal.txt", encoding="UTF-8")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

#Load the document by calling loader.load()
# Define the Text Splitter 
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

#Create a split of the document using the text splitter
splits = text_splitter.split_documents(docs)


from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings()

persist_directory = './journalchroma'

# Create the vector store
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())