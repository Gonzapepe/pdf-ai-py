import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from template import css, user_template, bot_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(
    #     repo_id="google/flan-t5-xxl",
    #     model_kwargs={"temperature": 0.7, "max_length": 1024},
    # )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    st.write(response)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat con múltiples PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat con múltiples PDFs :books:")

    user_question = st.text_input("Hacer una pregunta sobre tus documentos: ")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}", "Hola Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hola humano"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Tus documentos")
        pdf_docs = st.file_uploader(
            "Subir tus Archivos PDF y clickea en 'Procesar'", accept_multiple_files=True
        )
        if st.button("Procesar"):
            with st.spinner("Procesando"):
                # Recibir el texto de los pdfs
                raw_text = get_pdf_text(pdf_docs)

                # Recibir chunks de texto
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # Crear guardado de vectores
                vectorstore = get_vectorstore(text_chunks)

                # Crear cadena de conversación
                st.session_state.conversation = get_conversation_chain(vectorstore)
    st.session_state.conversation


if __name__ == "__main__":
    main()
