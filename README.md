# WASLAI: 24/7 AI Voice Mentor for PWD Students

## Project Overview

**WASLAI** is an AI-powered mentor designed to help students with disabilities (PWD students) access important educational and disability-related information.

The system uses **Retrieval-Augmented Generation (RAG)** to search uploaded policy documents and provide relevant answers to users' questions.

WASLAI aims to make information about HEC, PEC, and disability-support policies easier to access through an interactive AI-based application.

## Problem Statement

Students with disabilities may face difficulties finding clear and reliable information about:

* Educational facilities
* Disability-related policies
* Accessibility support
* Higher Education Commission (HEC) policies
* Pakistan Engineering Council (PEC) policies
* Available academic accommodations

WASLAI addresses this problem by providing a centralized AI mentor that searches relevant documents and presents useful answers.

## Main Features

* AI-powered question-answering system
* Retrieval-Augmented Generation (RAG)
* PDF and TXT document processing
* Semantic document search
* FAISS vector database
* Hugging Face sentence-transformer embeddings
* OpenAI language model integration
* Display of document sources
* Simple and user-friendly Streamlit interface

## Technology Stack

* **Python**
* **Streamlit**
* **LangChain**
* **OpenAI API**
* **FAISS**
* **Hugging Face Sentence Transformers**
* **PyPDF**

## Project Structure

```text
WASLAI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── HEC_Policies.pdf
    ├── PEC_Policies.pdf
    └── other_policy_documents.pdf
```

## How It Works

1. Policy documents are placed inside the `data` folder.
2. The application extracts text from PDF and TXT files.
3. The extracted text is divided into smaller sections.
4. Sentence-transformer embeddings are generated.
5. The sections are stored in a FAISS vector database.
6. When a user asks a question, relevant document sections are retrieved.
7. The OpenAI model generates an answer using the retrieved information.
8. The application displays the answer and related document sources.

## Installation

Clone the repository:

```bash
git clone https://github.com/kashufjaved6/WASLAI.git
```

Move into the project directory:

```bash
cd WASLAI
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## API Key Configuration

The OpenAI API key must be stored securely as an environment variable or through Streamlit Secrets.

Do not upload API keys, `.env` files, or `secrets.toml` files to GitHub.

For local execution, you may set the environment variable as follows:

```bash
OPENAI_API_KEY="your_api_key_here"
```

## Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

## Intended Users

WASLAI is intended to support:

* Students with disabilities
* University students
* Educational institutions
* Disability-support offices
* Academic mentors
* Researchers working on accessible education

## Future Improvements

* Voice input and voice output
* Multilingual support, including Urdu
* More disability-policy documents
* Accessibility-focused interface improvements
* User authentication
* Mobile application support
* Improved citation and document-reference features
* Integration with additional educational resources

## Disclaimer

WASLAI provides information based on the documents supplied to the application. Users should verify important policy, legal, academic, or institutional information with the relevant official authorities.

## Project Name

**WASLAI — 24/7 AI Voice Mentor for PWD Students**
