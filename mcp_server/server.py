import os
from mcp.server.fastmcp import FastMCP
from retrieve import HybridRetriever

# Boot FastMCP instance
mcp = FastMCP("SynapseRAG")

# We lazy-initialize the retriever inside the tool to ensure the server starts immediately
# even if Synapse Cache or other resources are starting up.
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        port = int(os.environ.get("NEARBY_PORT", 6380))
        directory = os.environ.get("KNOWLEDGE_BASE_DIR", "./knowledge_base")
        _retriever = HybridRetriever(directory=directory, port=port)
    return _retriever

@mcp.tool()
def search_internal_knowledge(query: str) -> str:
    """
    Search the local-first internal knowledge base.
    Uses a hybrid retrieval system combining Vector Search (Synapse Cache cosine similarity)
    and Keyword Search (BM25) combined using Reciprocal Rank Fusion (RRF).
    Returns the top 5 most relevant document chunks along with source metadata.
    """
    try:
        retriever = get_retriever()
        results = retriever.retrieve(query)
        
        if not results:
            return "No matching internal knowledge found for the query."
            
        formatted_chunks = []
        for i, res in enumerate(results, 1):
            formatted_chunks.append(
                f"[Source: {res['source']}] (Relevance Rank Score: {res['rrf_score']:.4f})\n"
                f"Content:\n{res['text'].strip()}\n"
                f"{'='*60}"
            )
            
        return "\n\n".join(formatted_chunks)
    except Exception as e:
        return f"Error executing hybrid search: {str(e)}"

if __name__ == "__main__":
    # Run via stdio transport for direct agent integration
    mcp.run()
