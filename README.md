# 🎥 YouTube RAG Chatbot

Ask intelligent questions about any YouTube video using AI!  
This project extracts video transcripts, finds the most relevant parts using semantic search (FAISS), and generates answers using a Large Language Model (LLM).
---
## 🧠 Why This Project?

YouTube videos are packed with knowledge — but finding answers from long videos is tedious.  
This project turns any YouTube video into an intelligent Q&A chatbot using Retrieval-Augmented Generation (RAG).  
Instead of watching a 20-minute video, just **ask a question** and get a smart, context-aware answer.
---
## 🚀 How It Works

1. 🎯 **Input a YouTube URL**
2. 📝 **Extract the transcript** (typed or auto-generated)
3. 📚 **Chunk the transcript** for semantic search
4. 🧠 **Embed using SentenceTransformer**
5. 🔎 **Store & search with FAISS**
6. 💬 **Use an LLM (Mistral-7B)** to answer based on retrieved context
---
## 🛠️ Tech Stack / Tools Used

| Category | Tool |
|---------|------|
| 💬 LLM | [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) via Hugging Face Inference API |
| 🔍 Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| 📄 Transcript | `youtube-transcript-api` |
| 🧠 Vector DB | `FAISS` (in-memory) |
| 🔗 Framework | `LangChain` |
| 🌐 UI | `Streamlit` |
| 🐍 Language | Python 3.10+ |

---

## 🖥️ Demo (Optional)

> Paste a YouTube URL and ask:  
> *“What is the main idea of this video?”*  
> *“What tools were discussed?”*

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/youtube-rag-chatbot.git
cd youtube-rag-chatbot

pip install -r requirements.txt

# Run the app
streamlit run app.py

🔮 Possible Enhancements
 🧠 Add support for multi-turn conversation

 🌍 Enable multilingual transcript-based Q&A

 💾 Save and load FAISS index to/from disk

 📥 Let users upload audio/video files directly

 🧠 Add OpenAI model support (e.g. gpt-3.5, gpt-4) for more accurate, faster results

 ☁️ Deploy on Streamlit Cloud or Hugging Face Spaces

⚡ Performance Tips
-If the app feels slow or times out during generation, try the following:
-⚙️ Use a smaller or faster model such as:
-google/flan-t5-base
-tiiuae/falcon-7b-instruct
-HuggingFaceH4/zephyr-7b-alpha

🧹 Reduce the number of context chunks:

Limit FAISS to top 1–2 matches

Truncate each chunk to 200–300 words

📶 Ensure stable internet — latency depends on API speed

🔁 Use st.session_state to avoid recomputing vector stores or embeddings

🚀 Use OpenAI API key (e.g. gpt-3.5-turbo) for highly efficient, low-latency answers:
