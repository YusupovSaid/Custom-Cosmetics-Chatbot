# import json
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.documents import Document
# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA

# # Load your dataset
# with open("fullqa_trimmed.jsonl", "r", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]
# print("Datasets loaded")
# # Convert data into LangChain Document format
# documents = [
#     Document(page_content=entry["answer"], metadata={"question": entry["question"]})
#     for entry in data
# ]
# print("converted into langchain")
# # Initialize Ollama embeddings
# embedding_model = OllamaEmbeddings(model="llama3")
# print("ollama embeddings")
# # Create a Chroma vector store
# vectorstore = Chroma.from_documents(documents, embedding=embedding_model)
# print("chroma vector store created")
# # Initialize the LLaMA 3 model via Ollama
# llm = Ollama(model="llama3")
# print("llama3 initialized")
# # Set up the RetrievalQA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectorstore.as_retriever(),
#     return_source_documents=True
# )
# print("retrivial qa stted up")
# # Example interaction
# query = "What is your return policy?"
# result = qa_chain.invoke(query)

# print("Answer:", result["result"])





# import json
# import os
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.documents import Document

# # 1. Load the dataset
# with open("fullqa_trimmed.jsonl", "r", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]
# print(f"✅ Loaded {len(data)} entries from fullqa_trimmed.jsonl")

# # 2. Convert to LangChain Document format
# documents = [
#     Document(page_content=entry["answer"], metadata={"question": entry["question"]})
#     for entry in data
# ]
# print("✅ Converted to LangChain Document format")

# # 3. Create embeddings
# embedding_model = OllamaEmbeddings(model="llama3")
# print("✅ Initialized Ollama embeddings")

# # 4. Save Chroma vector store
# persist_dir = "./chroma_index"
# if not os.path.exists(persist_dir):
#     os.makedirs(persist_dir)

# vectorstore = Chroma.from_documents(documents, embedding=embedding_model, persist_directory=persist_dir)
# vectorstore.persist()
# print(f"✅ Chroma vector store created and saved at {persist_dir}")




import json
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Load the dataset and limit to 30 entries
with open("fullqa_trimmed.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
print(f"✅ Loaded {len(data)} entries (for testing) from fullqa_trimmed.jsonl")

# 2. Convert to LangChain Document format: Q + A
documents = [
    Document(
        page_content=f"Q: {entry['question']}\nA: {entry['answer']}",
        metadata={"question": entry["question"]}
    )
    for entry in data
]
print("✅ Converted to LangChain Document format with Q+A")

# 3. Initialize HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Initialized HuggingFace embeddings (all-MiniLM-L6-v2)")

# 4. Save Chroma vector store
persist_dir = "./chroma_index"  
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

vectorstore = Chroma.from_documents(documents, embedding=embedding_model, persist_directory=persist_dir)
vectorstore.persist()
print(f"✅ Chroma test vector store saved at: {persist_dir}")
