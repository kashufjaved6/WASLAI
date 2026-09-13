
import os
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="WASLAI",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------------------------
# Read OpenAI API key safely
# --------------------------------------------------

def get_openai_api_key():
    """
    Read the OpenAI API key from Streamlit Secrets
    or from an environment variable.
    """

    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY")


# --------------------------------------------------
# Read documents from the data folder
# --------------------------------------------------

def read_documents(data_folder):
    documents = []

    for file_path in sorted(data_folder.iterdir()):

        # Read PDF files
        if file_path.suffix.lower() == ".pdf":

            try:
                reader = PdfReader(str(file_path))

                document_text = (
                    f"\nSOURCE: {file_path.name}\n"
                )

                for page in reader.pages:
                    page_text = page.extract_text() or ""

                    if page_text.strip():
                        document_text += page_text + "\n"

                if document_text.strip():
                    documents.append(document_text)

            except Exception as error:
                st.warning(
                    f"Could not read {file_path.name}: {error}"
                )

        # Read TXT files
        elif file_path.suffix.lower() == ".txt":

            try:
                text = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                document_text = (
                    f"\nSOURCE: {file_path.name}\n{text}"
                )

                if document_text.strip():
                    documents.append(document_text)

            except Exception as error:
                st.warning(
                    f"Could not read {file_path.name}: {error}"
                )

    return documents


# --------------------------------------------------
# Load documents and create the vector database
# --------------------------------------------------

@st.cache_resource
def load_vectorstore():

    data_folder = Path(__file__).parent / "data"

    if not data_folder.exists():
        return None

    documents = read_documents(data_folder)

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    text_chunks = text_splitter.split_text(
        "\n".join(documents)
    )

    if not text_chunks:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    return vectorstore


# --------------------------------------------------
# Create the language model
# --------------------------------------------------

@st.cache_resource
def load_llm():

    api_key = get_openai_api_key()

    if not api_key:
        return None

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )


# --------------------------------------------------
# Generate an answer using retrieved documents
# --------------------------------------------------

def generate_answer(question, vectorstore, llm):

    retrieved_documents = vectorstore.similarity_search(
        question,
        k=3
    )

    if not retrieved_documents:
        return (
            "I could not find relevant information in the "
            "uploaded policy documents."
        ), []

    context_parts = []

    for document in retrieved_documents:
        context_parts.append(document.page_content)

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are WASLAI, an accessibility assistant for students
with disabilities in Pakistan.

Answer the user's question only from the provided context.
Do not invent policies, facilities, deadlines, or eligibility
criteria.

If the context does not contain enough information, clearly say:
"I could not find this information in the uploaded documents."

Give a clear, simple, and helpful answer.
Mention that users should verify important policy information
with the relevant official institution.

Context:
{context}
"""
            ),
            (
                "human",
                "{question}"
            )
        ]
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    return response.content, retrieved_documents


# --------------------------------------------------
# WASLAI user interface
# --------------------------------------------------

st.title("🎓 WASLAI")
st.subheader("24/7 AI Voice Mentor for PWD Students")

st.write(
    """
WASLAI helps students find information from uploaded
HEC, PEC, NUST, and disability-policy documents.
"""
)

st.info(
    "Ask a question about accessibility, scholarships, "
    "or disability-related educational policies."
)

vectorstore = load_vectorstore()
llm = load_llm()

if vectorstore is None:

    st.error(
        "No readable PDF or TXT documents were found in the "
        "'data' folder. Please upload your documents."
    )

elif llm is None:

    st.error(
        "OPENAI_API_KEY is missing. Please add it in "
        "Streamlit Secrets."
    )

else:

    question = st.text_input(
        "Enter your question:",
        placeholder=(
            "What facilities are available for PWD students?"
        )
    )

    if st.button("🔍 Get Answer"):

        if not question.strip():

            st.warning("Please enter a question first.")

        else:

            with st.spinner(
                "Searching policy documents and generating an answer..."
            ):

                try:
                    answer, source_documents = generate_answer(
                        question,
                        vectorstore,
                        llm
                    )

                    st.subheader("Answer")
                    st.success(answer)

                    if source_documents:

                        with st.expander(
                            "📚 View document sources"
                        ):

                            for index, document in enumerate(
                                source_documents,
                                start=1
                            ):

                                st.markdown(
                                    f"**Source {index}**"
                                )

                                st.write(
                                    document.page_content[:700]
                                )

                except Exception as error:

                    st.error(
                        "An error occurred while generating "
                        f"the answer: {error}"
                    )
