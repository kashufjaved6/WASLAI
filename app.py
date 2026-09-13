import os
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="WASLAI",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------------------------
# Read API key safely
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
# Load documents and create RAG system
# --------------------------------------------------

@st.cache_resource
def load_rag():

    data_folder = Path("data")

    if not data_folder.exists():
        st.error("The 'data' folder was not found.")
        return None

    documents = []

    for file_path in data_folder.iterdir():

        # Read PDF files
        if file_path.suffix.lower() == ".pdf":

            try:
                reader = PdfReader(str(file_path))

                document_text = f"\nSOURCE: {file_path.name}\n"

                for page in reader.pages:
                    page_text = page.extract_text()

                    if page_text:
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
                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as text_file:

                    document_text = (
                        f"\nSOURCE: {file_path.name}\n"
                        + text_file.read()
                    )

                if document_text.strip():
                    documents.append(document_text)

            except Exception as error:
                st.warning(
                    f"Could not read {file_path.name}: {error}"
                )

    if not documents:
        st.error(
            "No PDF or TXT documents were found in the 'data' folder."
        )
        return None

    # Split documents into smaller sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    text_chunks = text_splitter.split_text(
        "\n".join(documents)
    )

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    vectorstore = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    # Get OpenAI API key
    api_key = get_openai_api_key()

    if not api_key:
        st.error(
            "OPENAI_API_KEY is missing. "
            "Please add it in Streamlit Secrets."
        )
        return None

    # Create language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

    # Create Retrieval-Augmented Generation chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        return_source_documents=True
    )

    return qa_chain


# --------------------------------------------------
# WASLAI user interface
# --------------------------------------------------

st.title("🎓 WASLAI")
st.subheader("24/7 AI Voice Mentor for PWD Students")

st.write(
    """
    WASLAI helps students find information from uploaded
    HEC, PEC, and disability-policy documents.
    """
)

st.info(
    "Ask a question about HEC or PEC disability policies."
)

qa_chain = load_rag()


if qa_chain is not None:

    question = st.text_input(
        "Enter your question:",
        placeholder="What facilities are available for PWD students?"
    )

    if st.button("Get Answer"):

        if not question.strip():

            st.warning("Please enter a question first.")

        else:

            with st.spinner("Searching the policy documents..."):

                try:
                    result = qa_chain.invoke(
                        {"query": question}
                    )

                    st.subheader("Answer")
                    st.success(result["result"])

                    source_documents = result.get(
                        "source_documents",
                        []
                    )

                    if source_documents:

                        with st.expander(
                            "View document sources"
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
                        f"An error occurred while generating the answer: {error}"
                    )
