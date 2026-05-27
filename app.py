import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer


with open("documents.txt", 'r', encoding='utf-8') as f:
    documents = f.read().split("\n")

documents = [doc.strip() for doc in documents if doc.strip()]

print(documents)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

document_embeddings = embedding_model.encode(documents)

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(document_embeddings).astype("float32"))


# Retrieval Agent
def retrieve_documents(query, top_k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k=top_k)

    retrieved_docs = [documents[idx] for idx in indices[0]]

    return retrieved_docs


## Hugging face agents
#1.
summarier_agent = pipeline(
    task="summarization", 
    model = "facebook/bart-large-cnn"
)

#2. 
qa_agent = pipeline(
    task='question-answering',
    model = 'distilbert/distilbert-base-cased-distilled-squad'
)

#3.
sentiment_agent = pipeline(
    task='text-classification',
    model = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
)

generator_agent = pipeline(
    task = 'text-generation',
    model = 'gpt2'
)



def coordinator(query):
    print('User query: ',query)

    retrieved_docs = retrieve_documents(query)

    context = " ".join(retrieved_docs)

    print("\n=========================================")
    print("Retrieved context \n")
    print(context)

    summary = summarier_agent(
        context, 
        max_length=60,
        min_length=20,
        do_sample = False
    )[0]["summary_text"]

    qa_result = qa_agent(
        question = query,
        context = context
    )

    sentiment = sentiment_agent(context)[0]

    generated = generator_agent(
        f"Context: {context}, \nQuestion : {query}\n Answer : {qa_result}",
        max_new_tokens = 50,
        do_sample = True,
        temperature = 0.7
    )[0]["generated_text"]

    print("\n=======================================\n")
    print("Summary: ")
    print(summary)

    print("\n=======================================\n")
    print("QA result")
    print(qa_result["answer"])

    print("\n=======================================\n")
    print("Sentiment")
    print(sentiment)
    
    print("\n=======================================\n")
    print("Generated")
    print(generated)
    


query = input("Enter your question: ")
coordinator(query)



