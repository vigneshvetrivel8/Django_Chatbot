from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from pinecone import Pinecone, ServerlessSpec

os.environ["GOOGLE_API_KEY"] = "<GOOGLE-GEMINI-API-KEY>"
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents

# Load_pdf with the directory path containing all the PDFs
extracted_data = load_pdf(os.getcwd())

print("data extracted.")

import re
# Clean the text by removing unnecessary new line characters
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)  # Replace multiple new lines with a single space
    text = re.sub(r' {2,}', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))

print("Text chunks created successfully.")

# Function to generate embeddings for text chunks
def generate_embeddings(chunks, embeddings_model):
    embeddings = []
    for chunk in chunks:
    # for chunk in enumerate(chunks:
        chunk_embedding = embeddings_model.embed_query(clean_text(chunk.page_content))
        embeddings.append(chunk_embedding)
    return np.array(embeddings)

# Generate embeddings for text chunks
embeddings = generate_embeddings(text_chunks, embeddings_model)
print("embeddings.shape:", embeddings.shape)

index_name = 'corpus'

# Initialize Pinecone client with your API key and index name
pc = Pinecone(
    api_key= "<PINECONE-API-KEY>"
)

# Now do stuff
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name='corpus',
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
index = pc.Index(index_name)

# Generate sequential numeric IDs
ids = list(range(1, len(embeddings) + 1))

embeddings_list = embeddings.tolist()

print("upserting..")

#Upsert embeddings into the index
for emb, id, chunk in zip(embeddings_list, ids, text_chunks):
  index.upsert(vectors=[{"id": str(id), "values": emb, "metadata": {"content" : clean_text(chunk.page_content)} }])
