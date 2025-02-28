import pymongo
import requests

client = pymongo.MongoClient("mongoDB URI")
db = client.experta
collection = db.expert_data

hf_token = "xxxxxxxxxxxxxxxxxxxxxxxx"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

import time

def generate_embedding(text: str, retries=3, delay=5) -> list[float]:
    for attempt in range(retries):
        response = requests.post(
            embedding_url,
            headers={"Authorization": f"Bearer {hf_token}"},
            json={"inputs": text}
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

    raise ValueError(f"Request failed after {retries} attempts with status code {response.status_code}: {response.text}")




def update_composite_embedding():
    """
    Concatenates the text from multiple fields into one string,
    generates a single embedding, and stores it as 'composite_embedding_hf'.
    """
    for doc in collection.find({"username": {"$exists": True}}):
        print(f"Processing document ID: {doc['_id']}")
        # Create composite text from the fields
        composite_text = (
            f"Username: {doc.get('username', '')} "
            f"Bio: {doc.get('bio', '')} "
            f"Rating: {str(doc.get('rating', ''))}"
        )
        print(f"Composite text: {composite_text}")
        
        # Generate the composite embedding
        embedding = generate_embedding(composite_text)
        print(f"Generated embedding: {embedding}")
        
        # Store the embedding in the document
        doc["composite_embedding_hf"] = embedding
        collection.replace_one({"_id": doc["_id"]}, doc)
    
    print("Composite embeddings updated.")


def search_composite(query: str, index_name: str, limit: int = 4):
    """
    Performs a vector search using the composite embedding.
    """
    # composite_embedding_hf
    print(f"Query: {query}")
    query_embedding = generate_embedding(query)
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "composite_embedding_hf",    
                "numCandidates": 100,
                "limit": limit,
                "index": index_name,
            }
        }
    ]
    results = list(collection.aggregate(pipeline))
    for doc in results:
        # Adjust the printed fields according to your document structure.
        print(f"Username: {doc.get('username')}, Bio: {doc.get('bio')}, Rating: {doc.get('rating')}")



if __name__ == "__main__":
    # update_composite_embedding()  
    print("Searching using composite embeddings.........................")
     
    query = "i want a cybersecurity expert with rating more than 3"
    print(query)
    search_composite(query, index_name="composite_embedding_hf")
    print("-----------------------------------------------------------------------------------------------")
    query1 = "i want digital marketing expert"
    search_composite(query1, index_name="composite_embedding_hf")
    print("-----------------------------------------------------------------------------------------------")
    query2 = "i want ui/ux devloper with rating more than 3"
    search_composite(query1, index_name="composite_embedding_hf")

  
    



