from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace,HuggingFaceEndpoint
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


HUGGINGFACE_API_TOKEN=os.getenv('HUGGINGFACE_API_TOKEN')
# for tracking we can use langchain api key(langsmith)

# exatracting the video if from the url

def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")


# exatracting the languages for which transcriptions are present
def get_available_languages(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return [(t.language_code, t.language) for t in transcript_list]
    except Exception as e:
        return []

# getting the transcriptions, using the language_code selected by the user
def get_transcription(video_id, lang_code):
    """
    Safely fetches transcript text (manual or auto-generated) for a given video ID and language code.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        selected_transcript = None

        for t in transcript_list:
            if t.language_code == lang_code:
                selected_transcript = t
                break

        if not selected_transcript:
            return None

        
        transcript_data = selected_transcript.fetch()
        return " ".join(chunk.text for chunk in transcript_data)

    except Exception as e:
        print("Transcript fetch error:", e)
        return None




# Dividing the data into chuns
def create_chunks(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([transcript])
    # chunks
    return chunks


# embedding model which is used to convert text into vectors
def embeder():
    embedding_model=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# vector_store = FAISS.from_documents(chunks, embedding_model)

# we can save locally using
# vector_store.save_local('vectostore/db_faiss')

# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# response=retriever.invoke("What is Self Attention")
# print(response)


# designing the prompt template
prompt_temp = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context deeply.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)


# making the llm model using the ChatHuggingFace mistral model
def load_llm():
    llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text generation",
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN )

    llm=ChatHuggingFace(llm=llm)

    return llm

# we can also us this function for chain creation
def create_chain(retriever,format_docs):
    parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

    llm=load_llm()
    final_chain=parallel_chain|prompt|llm|parser
    return final_chain




# === Streamlit UI ===
st.set_page_config(page_title="Ask about the video", layout="wide")
st.title("🎥 Ask about the video")

# Step 1: YouTube URL input
url = st.text_input("Paste a YouTube video URL:")
if url:
    video_id = extract_video_id(url)
    if not video_id:
        st.warning("Invalid YouTube URL.")
        st.stop()

    langs = get_available_languages(video_id)
    if not langs:
        st.error("No transcript available for this video.")
        st.stop()

    lang_map = {f"{name} ({code})": code for code, name in langs}
    selected_lang_display = st.selectbox("Choose transcript language:", list(lang_map.keys()))
    selected_lang_code = lang_map[selected_lang_display]

    if st.button("➡️ Load Transcript"):
        with st.spinner("Loading transcript..."):
            transcript = get_transcription(video_id, selected_lang_code)
            if not transcript:
                st.error("Could not load transcript.")
                st.stop()

            # Store in session_state
            st.session_state.transcript = transcript
            st.session_state.chunks = create_chunks(transcript)
            embedding_model = embeder()
            vector_store = FAISS.from_documents(st.session_state.chunks, embedding_model)
            st.session_state.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
            st.success("✅ Transcript loaded and vector store ready!")

# === Q&A Section ===
if "retriever" in st.session_state:
    query = st.text_input("💬 Ask a question:")
    if st.button("Get Answer") and query:
        with st.spinner("Generating response..."):
            retriever = st.session_state.retriever
            context_docs = retriever.invoke(query)
            context_text = "\n\n".join(doc.page_content for doc in context_docs)

            def format_docs(x):
                return context_text

            llm = load_llm()
            parser = StrOutputParser()

            chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }) | prompt_temp | llm | parser

            result = chain.invoke(query)

            st.markdown("### ✅ Answer:")
            st.success(result)
