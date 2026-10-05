import os
import hashlib
from pathlib import Path
from operator import itemgetter

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Employee_Handbook_Kenexai.pdf"

INDEX_NAME = "rag-demo1"
NAMESPACE = "pdf-docs"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3

# Objects are initialized lazily so importing this module from
# frontend.py does not immediately perform PDF ingestion.
_vector_store = None
_retriever = None
_llm = None
_memory = None


# ============================================================
# VALIDATION
# ============================================================
def validate_environment():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to your .env file."
        )

    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError(
            "PINECONE_API_KEY not found. Add it to your .env file."
        )

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )


# ============================================================
# PINECONE
# ============================================================
def get_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    return pc.Index(INDEX_NAME)


# ============================================================
# PDF -> CHUNKS
# ============================================================
def load_pdf_chunks():
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()

    print(f"Loaded {len(pages)} pages from {PDF_PATH.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    return chunks


def create_chunk_ids(chunks):
    return [
        hashlib.md5(
            f"{c.metadata.get('source', '')}::{c.metadata.get('page', '')}::{c.page_content}".encode(
                "utf-8"
            )
        ).hexdigest()
        for c in chunks
    ]


# ============================================================
# VECTOR STORE / INGESTION
# ============================================================
def create_vector_store(index):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE,
    )


def ingest_pdf_if_needed(vector_store, index):
    """
    Store PDF chunks only when the namespace is empty.

    This prevents the PDF from being embedded and uploaded every time
    frontend.py imports rag.py.
    """
    stats = index.describe_index_stats()
    namespace_info = stats.get("namespaces", {}).get(NAMESPACE, {})
    existing_vectors = namespace_info.get("vector_count", 0)

    if existing_vectors > 0:
        print(
            f"Pinecone already contains {existing_vectors} vectors "
            f"in namespace '{NAMESPACE}'."
        )
        return

    chunks = load_pdf_chunks()
    ids = create_chunk_ids(chunks)

    vector_store.add_documents(
        chunks,
        ids=ids,
        batch_size=100,
    )

    print(f"Stored {len(chunks)} chunks in Pinecone.")


# ============================================================
# RAG PROMPT
# ============================================================
def format_docs(docs):
    return "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )


examples = [
    {
        "input": "Who are you?",
        "output": (
            "I am your encyclopedia—I will provide you with all the "
            "information you need about the credit card domain."
        ),
    },
    {
        "input": "I'm from a Human wanted to explore about cards payments?",
        "output": (
            "Absolutely. Many new learners come from a non-domain background. "
            "We start from fundamentals and take you to job-ready, step by step. "
            "No prior domain knowledge needed to begin."
        ),
    },
    {
        "input": "How you will help me?",
        "output": (
            "Whatever topic you post, I will try to give you optimized information. "
            "Information is useful to know the domain better."
        ),
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful, friendly AI assistant for ABC Domain Encyclopedia.

Use the retrieved Employee Handbook context to answer questions about Kenexai.
Answer handbook questions using ONLY the supplied context.
If the answer is not present in the context, say that you don't know based on the Employee Handbook.
Do not invent or assume information.
Mention the page number(s) used when answering from the handbook.
Answer clearly and concisely.

Retrieved Employee Handbook context:
{context}""",
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


# ============================================================
# INITIALIZATION
# ============================================================
def initialize_rag():
    """Initialize Pinecone, ingest PDF if required, and create RAG objects."""
    global _vector_store, _retriever, _llm, _memory

    if _vector_store is not None:
        return

    validate_environment()

    index = get_pinecone_index()
    _vector_store = create_vector_store(index)

    ingest_pdf_if_needed(_vector_store, index)

    _retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    _llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1024,
    )

    _memory = InMemoryChatMessageHistory()


# ============================================================
# MEMORY
# ============================================================
def demo_memory():
    """Return a fresh chat history for a frontend session."""
    return InMemoryChatMessageHistory()


# ============================================================
# CONVERSATION
# ============================================================
def demo_conversation(input_text, memory=None):
    """
    Send a user question to the RAG chatbot.

    Parameters:
        input_text: User's question.
        memory: Optional InMemoryChatMessageHistory. If omitted, a module-level
                memory is used.

    Returns:
        (assistant_reply, memory)
    """
    global _memory

    initialize_rag()

    if memory is None:
        memory = _memory

    # Retrieve PDF content relevant to the current question.
    retrieved_docs = _retriever.invoke(input_text)
    context = format_docs(retrieved_docs)

    trimmer = trim_messages(
        max_tokens=1000,
        strategy="last",
        token_counter=_llm,
        include_system=False,
        start_on="human",
    )

    base_chain = (
        RunnablePassthrough.assign(
            history=itemgetter("history") | trimmer,
            context=lambda _: context,
        )
        | RAG_PROMPT
        | _llm
    )

    chat_with_history = RunnableWithMessageHistory(
        base_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="history",
    )

    try:
        result = chat_with_history.invoke(
            {"input": input_text},
            config={
                "configurable": {
                    "session_id": "bepec-session"
                }
            },
        )

        return result.content, memory

    except Exception as e:
        return f"Sorry, something went wrong: {str(e)}", memory


# ============================================================
# OPTIONAL: CLEAR MEMORY
# ============================================================
def clear_memory(memory=None):
    """Clear conversation history."""
    global _memory

    target = memory if memory is not None else _memory

    if target is not None:
        target.clear()


# ============================================================
# TEST WHEN RUN DIRECTLY
# ============================================================
if __name__ == "__main__":
    answer, memory = demo_conversation(
        "What is the sick leave policy in Kenexai?"
    )
    print("\nAnswer:\n")
    print(answer)
