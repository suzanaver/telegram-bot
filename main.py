# -*- coding: utf-8 -*-

import json
import os
import telebot
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from pypdf import PdfReader
import chromadb
from gtts import gTTS

# =========================
# CONFIG
# =========================

TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

# =========================
# STORAGE PATH
# =========================

BASE_PATH = "/app/data"
os.makedirs(BASE_PATH, exist_ok=True)

# =========================
# EMBEDDINGS MODEL
# =========================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# USER PATHS
# =========================

def get_user_paths(user_id):
    user_path = f"{BASE_PATH}/user_{user_id}"
    docs_path = f"{user_path}/docs"
    db_path = f"{user_path}/vectordb"
    memory_path = f"{user_path}/memory.json"
    os.makedirs(docs_path, exist_ok=True)
    os.makedirs(db_path, exist_ok=True)
    return docs_path, db_path, memory_path

# =========================
# MEMORY SYSTEM
# =========================

def load_memory(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(path, memory):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

# =========================
# BUILD RAG (PDF → DB)
# =========================

def build_rag(pdf_path, db_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    if not text.strip():
        print("EMPTY PDF TEXT")
        return

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("docs")

    for i, chunk in enumerate(chunks):
        emb = embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[f"{i}_{len(chunk)}"],
            embeddings=[emb],
            documents=[chunk]
        )
    print("INDEX COMPLETE")

# =========================
# RAG ASK
# =========================

def ask_rag(question, db_path):
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("docs")

        q_emb = embedding_model.encode(question).tolist()
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=3
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            return "לא מצאתי מידע במסמכים 📄"

        context = "\n".join(docs)
        prompt = f"""You are an AI assistant.
Answer ONLY using the context below.
If no relevant info exists, say you don't know.

Context:
{context}

Question:
{question}"""

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Error in RAG: {str(e)}"

# =========================
# TELEGRAM HANDLERS
# =========================

@bot.message_handler(func=lambda message: True)
def handle(message):
    user_id = message.from_user.id
    text = message.text
    docs_path, db_path, memory_path = get_user_paths(user_id)
    memory = load_memory(memory_path)

    if text.lower().startswith("/pdf"):
        bot.reply_to(message, "📄 שלח עכשיו PDF")
        return

    answer = ask_rag(text, db_path)

    memory.append({"user": text, "bot": answer})
    save_memory(memory_path, memory)

    try:
        tts = gTTS(text=answer[:400], lang="en")
        audio = "/tmp/answer.mp3"
        tts.save(audio)
        bot.reply_to(message, answer)
        with open(audio, "rb") as f:
            bot.send_voice(message.chat.id, f)
    except:
        bot.reply_to(message, answer)


@bot.message_handler(content_types=['document'])
def handle_pdf(message):
    user_id = message.from_user.id
    docs_path, db_path, memory_path = get_user_paths(user_id)

    file_info = bot.get_file(message.document.file_id)
    downloaded = bot.download_file(file_info.file_path)

    pdf_path = f"{docs_path}/file.pdf"
    with open(pdf_path, "wb") as f:
        f.write(downloaded)

    build_rag(pdf_path, db_path)
    bot.reply_to(message, "✅ PDF נשמר ואונדקס בהצלחה!")

# =========================
# START BOT
# =========================

print("🚀 RAG BOT RUNNING")
bot.infinity_polling()
