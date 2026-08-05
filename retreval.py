import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


load_dotenv()


persistent_directory = "db/chroma_db"


class NvidiaNemotronEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings class for NVIDIA Nemotron 3 Embed 1B.

    Uses:
    - input_type="passage" for document embeddings during ingestion
    - input_type="query" for query embeddings during retrieval
    """

    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        self.model = "nvidia/nemotron-3-embed-1b"

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
            extra_body={
                "input_type": "passage",
                "truncate": "END"
            }
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float",
            extra_body={
                "input_type": "query",
                "truncate": "END"
            }
        )

        return response.data[0].embedding


# Load embeddings and vector store
embedding_model = NvidiaNemotronEmbeddings()

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# Search for relevant documents
query = "Who founded Nvidia?"

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# Optional threshold-based retriever
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3
#     }
# )

relevant_docs = retriever.invoke(query)


print(f"User Query: {query}")
print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Combine the query and the relevant document contents
if not relevant_docs:
    combined_input = f"""
The user asked:

{query}

No relevant documents were retrieved.

Please respond with:
"I don't have enough information to answer that question based on the provided documents."
"""
else:
    context = "\n\n".join(
        [
            f"Document {i + 1}:\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ]
    )

    combined_input = f"""
Based on the following documents, answer the user's question.

User Question:
{query}

Documents:
{context}

Instructions:
- Use only the information provided in the documents.
- Do not use outside knowledge.
- If the answer is not present in the documents, say:
  "I don't have enough information to answer that question based on the provided documents."
- Give a clear and helpful answer.
"""


# Create NVIDIA OpenAI-compatible client for LLM generation
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)


completion = client.chat.completions.create(
    model="nvidia/nemotron-mini-4b-instruct",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful RAG assistant. "
                "Answer strictly using only the provided document context. "
                "If the answer is not present in the context, say you do not have enough information."
            )
        },
        {
            "role": "user",
            "content": combined_input
        }
    ],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=False
)


print("\n--- Generated Response ---")
print("Content only:")
print(completion.choices[0].message.content)