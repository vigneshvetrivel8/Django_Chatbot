from django.shortcuts import render
import pinecone
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize Pinecone index
index = pinecone.Index(host="<PINECONE-HOST-URL>", api_key="<PINECONE-API-KEY>")

import os
os.environ["GOOGLE_API_KEY"] = "<GOOGLE-GEMINI-API-KEY>"

# Initialize generative AI and embeddings models
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize conversation memory
memory = ConversationBufferMemory()

# Initialize ConversationChain with language model and memory
conversation = ConversationChain(llm=llm, memory=memory)

def get_response(user_query):
    query_embedding = embeddings_model.embed_query(user_query)

    # Perform a similarity search in the Pinecone index
    matching_ids = index.query(vector=query_embedding, top_k=3)
    content_list = []

    for id in matching_ids['matches']:
        result = index.fetch(ids=[id['id']])
        content = result['vectors'][str(id['id'])]['metadata']['content']
        content_list.append(content)

    prompt = (
        """
        Question: {question}
        Context: {context}

        Rules:
        1) Answer the question according to the context provided.
        2) If context is related but doesn't provide clarity or enough information, ask the user to contact the business directly.
        3) If the question is too general or unrelated to the context or something like greetings, answer directly.
        """
    ).format(context=content_list, question=user_query)

    # Predict response based on prompt
    output = conversation.predict(input=prompt)
    return output

def index_view(request):
    answer = None
    if request.method == 'POST':
        question = request.POST.get('question')
        answer = get_response(question)
    return render(request, 'myapp/index.html', {'answer': answer})
