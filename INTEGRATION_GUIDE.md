# Synapse Cache Integration Guide

Synapse Cache isn't just an MCP server for Claude; it is a full-fledged, independent Vector Database. You can easily connect your own custom RAG applications, LangChain scripts, or standard Python apps directly to it.

This guide will show you how to use the raw Python driver (`SynapseClient`) to build a custom RAG pipeline.

## 1. Prerequisites

First, ensure the Synapse Cache Go server is running (either via Docker or natively):
```bash
# Easiest method:
docker compose up -d nearby
```

In your custom Python project, you will need two files from this repository:
1. `mcp_server/client.py` (The raw TCP Database Driver)
2. `mcp_server/protocol.py` (The RESP parsing logic used by the client)

Copy these two files into your custom project folder.

## 2. Environment Setup

Because Synapse Cache is secured with TLS and Password Authentication, your Python script needs to know the credentials. You can set these in your terminal before running your script, or using `os.environ` inside your script.

```bash
export SYNAPSE_PASSWORD="YourSuperSecurePassword123!"
export SYNAPSE_TLS="true"
export SYNAPSE_INSECURE_SKIP_VERIFY="true"
```

## 3. Basic Client Usage

Here is a simple example showing how to connect to the database and use basic Key-Value storage:

```python
from client import SynapseClient

# Initialize the client
# (It will automatically read the SYNAPSE_PASSWORD and SYNAPSE_TLS env variables)
db = SynapseClient(host="localhost", port=6380)

# Store some raw text (perfect for storing chunk data)
db.set("doc:1", "This is the content of my document.")

# Retrieve it
text = db.get("doc:1")
print(f"Retrieved: {text}")
```

## 4. Advanced Vector Integration (Custom RAG)

If you are building a custom RAG application using OpenAI embeddings, here is exactly how you store and search vectors.

### Storing Vectors (`VSET`)
```python
from client import SynapseClient
from openai import OpenAI

db = SynapseClient(host="localhost", port=6380)
openai = OpenAI(api_key="sk-your-api-key")

# 1. You have a chunk of text
chunk_id = "chunk:finance_report:1"
chunk_text = "The company generated $5M in revenue in Q3."

# 2. Get the embedding from OpenAI
response = openai.embeddings.create(
 input=chunk_text,
 model="text-embedding-3-small"
)
vector = response.data[0].embedding

# 3. Store the RAW TEXT in standard storage
db.set(chunk_id, chunk_text)

# 4. Store the VECTOR in vector storage with metadata
db.vset(
 namespace="my_rag_app", 
 key=chunk_id, 
 vector=vector, 
 metadata={"source": "finance_report.pdf", "author": "John Doe"}
)
print("Vector stored successfully!")
```

### Searching Vectors (`VSIMILARITY`)
```python
# 1. The user asks a question
user_query = "How much revenue did we make in Q3?"

# 2. Embed the user's question
response = openai.embeddings.create(
 input=user_query,
 model="text-embedding-3-small"
)
query_vector = response.data[0].embedding

# 3. Ask Synapse Cache to find the top 3 most similar vectors
results = db.vsimilarity(
 namespace="my_rag_app", 
 vector=query_vector, 
 k=3
)

# 4. Process the results
for result in results:
 doc_id = result["id"]
 similarity_score = result["score"]
 metadata = result["metadata"]
 
 # Retrieve the actual text using the ID
 raw_text = db.get(doc_id)
 
 print(f"Match Score: {similarity_score}")
 print(f"Source: {metadata.get('source')}")
 print(f"Content: {raw_text}\n")
```

## 5. Architecture Best Practices

> [!TIP]
> **Performance Note:** Synapse Cache specifically separates text storage from vector storage. You store the raw text payload via `SET` and the vector via `VSET`. 
> 
> This split-architecture ensures that similarity calculations (`VSIMILARITY`) in Go only calculate math on dense float arrays and don't have to load massive text blobs into memory during the search loop, keeping the server incredibly fast. Always fetch the raw text via `GET` only *after* retrieving the matching IDs!
