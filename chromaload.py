from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

embedding = HuggingFaceEmbeddings()

# Specify the path to the directory where the vector database was saved
persist_directory = './journalchroma/'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print(vectordb._collection.count())



# Define your question for similarity search


# Perform similarity search
# docs = vectordb.similarity_search(question, k=3)

# Print the results
# for doc in docs:
#     print(f"Document: {doc['document']}, Similarity Score: {doc['score']}")

model_name = "ai-forever/mGPT-1.3B-mongol"  # Replace with your desired model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vectordb.as_retriever()
)

question = "М.А.К. Халлидэйн Тогтолцоот үүргийн хэлзүй"
result = qa_chain({"query": question})
print(result["result"])
print(result)