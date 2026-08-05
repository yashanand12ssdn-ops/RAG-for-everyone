import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


# Load environment variables
load_dotenv()


# -----------------------------
# NVIDIA Embedding Class
# -----------------------------
class NvidiaNemotronEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class for NVIDIA Nemotron 3 Embed 1B.

    Uses:
    - input_type="passage" for document embeddings during ingestion
    - input_type="query" for user query embeddings during retrieval
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


# -----------------------------
# Setup NVIDIA Client
# -----------------------------
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)


# -----------------------------
# Connect to Chroma Database
# -----------------------------
persistent_directory = "db/chroma_db"

embeddings = NvidiaNemotronEmbeddings()

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)


# -----------------------------
# Chat History
# -----------------------------
chat_history = []


# -----------------------------
# Helper Function for NVIDIA Chat
# -----------------------------
def call_nvidia_llm(messages, temperature=0.2, max_tokens=1024):
    """
    Calls NVIDIA's OpenAI-compatible chat completion endpoint.
    """

    completion = client.chat.completions.create(
        model="nvidia/nemotron-mini-4b-instruct",
        messages=messages,
        temperature=temperature,
        top_p=0.7,
        max_tokens=max_tokens,
        stream=False
    )

    return completion.choices[0].message.content


# -----------------------------
# Ask Question Function
# -----------------------------
def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question standalone using conversation history
    if chat_history:
        rewrite_messages = [
            {
                "role": "system",
                "content": (
                    "Given the chat history, rewrite the new question so it is standalone "
                    "and searchable. Return only the rewritten question."
                )
            }
        ]

        rewrite_messages.extend(chat_history)

        rewrite_messages.append(
            {
                "role": "user",
                "content": f"New question: {user_question}"
            }
        )

        search_question = call_nvidia_llm(
            messages=rewrite_messages,
            temperature=0.1,
            max_tokens=256
        ).strip()

        print(f"Searching for: {search_question}")

    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Create final RAG prompt
    if not docs:
        combined_input = f"""
The user asked:

{user_question}

No relevant documents were retrieved.

Please respond with:
"I don't have enough information to answer that question based on the provided documents."
"""
    else:
        context = "\n\n".join(
            [
                f"Document {i + 1}:\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
        )

        combined_input = f"""
Based on the following documents, answer the user's question.

User Question:
{user_question}

Search Question Used:
{search_question}

Documents:
{context}

Instructions:
- Use only the information provided in the documents.
- Do not use outside knowledge.
- Consider the conversation history only for understanding the user's question.
- If the answer is not present in the documents, say:
  "I don't have enough information to answer that question based on the provided documents."
- Give a clear and helpful answer.
"""

    # Step 4: Get the final answer
    final_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful RAG assistant. "
                "Answer questions using only the provided document context. "
                "If the answer is not in the documents, say you do not have enough information."
            )
        }
    ]

    final_messages.extend(chat_history)

    final_messages.append(
        {
            "role": "user",
            "content": combined_input
        }
    )

    answer = call_nvidia_llm(
        messages=final_messages,
        temperature=0.2,
        max_tokens=1024
    )

    # Step 5: Save conversation history
    chat_history.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    print(f"\nAnswer: {answer}")

    return answer


# -----------------------------
# Simple Chat Loop
# -----------------------------
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()
