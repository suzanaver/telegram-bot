# -*- coding: utf-8 -*-

import json
import os
import telebot
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from pypdf import PdfReader
import chromadb
from gtts import gTTS
from groq import Groq
from datetime import datetime

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

groq_client = Groq(api_key=GROQ_API_KEY)

# =========================
# STORAGE PATH
# =========================

BASE_PATH = "/content/drive/MyDrive/AI_BOT"
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
    profile_path = f"{user_path}/profile.json"
    reminders_path = f"{user_path}/reminders.json"
    os.makedirs(docs_path, exist_ok=True)
    os.makedirs(db_path, exist_ok=True)
    return docs_path, db_path, memory_path, profile_path, reminders_path

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

def format_memory_for_prompt(memory, last_n=6):
    if not memory:
        return ""
    recent = memory[-last_n:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['bot']}")
    return "\n".join(lines)

# =========================
# PROFILE SYSTEM
# =========================

def load_profile(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_profile(path, profile):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def format_profile_for_prompt(profile):
    if not profile:
        return ""
    lines = ["מידע אישי על המשתמש:"]
    fields = {
        "name": "שם", "job": "עבודה", "hobbies": "תחביבים",
        "family": "משפחה", "goals": "מטרות", "notes": "הערות נוספות"
    }
    for key, label in fields.items():
        if profile.get(key):
            lines.append(f"{label}: {profile[key]}")
    return "\n".join(lines)

# =========================
# REMINDERS SYSTEM
# =========================

def load_reminders(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_reminders(path, reminders):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reminders, f, ensure_ascii=False, indent=2)

def add_reminder(path, text):
    reminders = load_reminders(path)
    reminder = {
        "id": len(reminders) + 1,
        "text": text,
        "created": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "done": False
    }
    reminders.append(reminder)
    save_reminders(path, reminders)
    return reminder

def format_reminders(reminders):
    active = [r for r in reminders if not r.get("done")]
    if not active:
        return "אין תזכורות פעילות ✅"
    lines = ["📋 *התזכורות שלך:*\n"]
    for r in active:
        lines.append(f"*{r['id']}.* {r['text']}\n   📅 נוצר: {r['created']}")
    return "\n".join(lines)

# =========================
# זיהוי שפה אוטומטי
# =========================

def detect_language(text):
    hebrew = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
    russian = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    if hebrew > 2:
        return "he"
    elif russian > 2:
        return "ru"
    else:
        return "en"

def lang_to_gtts(lang_code):
    return {"he": "he", "ru": "ru", "en": "en"}.get(lang_code, "en")

# =========================
# VOICE → TEXT (Whisper)
# =========================

def transcribe_voice(file_path):
    with open(file_path, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            file=(file_path, f.read()),
            model="whisper-large-v3",
        )
    return transcription.text

# =========================
# TEXT → VOICE
# =========================

def text_to_voice(text, lang):
    tts = gTTS(text=text[:400], lang=lang_to_gtts(lang))
    audio_path = "/tmp/answer.mp3"
    tts.save(audio_path)
    return audio_path

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
        return False
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("docs")
    for i, chunk in enumerate(chunks):
        emb = embedding_model.encode(chunk).tolist()
        collection.add(ids=[f"{i}_{len(chunk)}"], embeddings=[emb], documents=[chunk])
    return True

def has_pdf(db_path):
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("docs")
        return collection.count() > 0
    except:
        return False

# =========================
# ASK LLM
# =========================

def ask_llm(question, memory, profile, reminders=None):
    history = format_memory_for_prompt(memory)
    profile_text = format_profile_for_prompt(profile)

    reminders_text = ""
    if reminders:
        active = [r for r in reminders if not r.get("done")]
        if active:
            reminders_text = "התזכורות הפעילות של המשתמש:\n"
            for r in active:
                reminders_text += f"- {r['text']} (נוצר: {r['created']})\n"

    prompt = f"""You are a personal AI assistant. You know the user personally and care about them.
Answer in the same language the user writes in.
Be warm, helpful and personal — use their name if you know it.

{profile_text}

{reminders_text}

Previous conversation:
{history}

Question: {question}"""

    response = llm.invoke(prompt)
    return response.content

def ask_rag(question, db_path, memory, profile, reminders=None):
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("docs")
        q_emb = embedding_model.encode(question).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=3)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ask_llm(question, memory, profile, reminders)
        context = "\n".join(docs)
        history = format_memory_for_prompt(memory)
        profile_text = format_profile_for_prompt(profile)

        reminders_text = ""
        if reminders:
            active = [r for r in reminders if not r.get("done")]
            if active:
                reminders_text = "התזכורות הפעילות של המשתמש:\n"
                for r in active:
                    reminders_text += f"- {r['text']} (נוצר: {r['created']})\n"

        prompt = f"""You are a personal AI assistant. You know the user personally and care about them.
Answer in the same language the user writes in.
Be warm, helpful and personal — use their name if you know it.

{profile_text}

{reminders_text}

Previous conversation:
{history}

Document context:
{context}

Question: {question}"""

        response = llm.invoke(prompt)
        return response.content
    except:
        return ask_llm(question, memory, profile, reminders)

# =========================
# SEND ANSWER
# =========================

def send_answer(message, answer, lang):
    bot.reply_to(message, answer)
    try:
        audio_path = text_to_voice(answer, lang)
        with open(audio_path, "rb") as f:
            bot.send_voice(message.chat.id, f)
    except:
        pass

# =========================
# PROFILE SETUP FLOW
# =========================

user_states = {}

PROFILE_QUESTIONS = [
    ("name",    "👤 מה השם שלך?"),
    ("job",     "💼 מה העבודה / תחום שלך?"),
    ("hobbies", "🎯 מה התחביבים שלך?"),
    ("family",  "👨‍👩‍👧 ספר/י על המשפחה שלך (כתוב 'דלג' לדילוג)"),
    ("goals",   "🎯 מה המטרות שלך כרגע?"),
    ("notes",   "📝 עוד משהו שחשוב שאדע? (כתוב 'דלג' לדילוג)"),
]

# =========================
# TELEGRAM HANDLERS
# =========================

@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.reply_to(message,
        "👋 שלום! אני הסוכן האישי שלך.\n\n"
        "📋 /setprofile — הגדר פרופיל אישי\n"
        "👤 /profile — הצג פרופיל\n"
        "🔔 /remind [טקסט] — הוסף תזכורת\n"
        "📋 /reminders — הצג תזכורות\n"
        "✅ /done [מספר] — סמן תזכורת כבוצעה\n"
        "🎤 שלח הודעה קולית — אשמע ואענה!\n"
        "📄 שלח PDF — אענה על שאלות ממנו\n"
        "🧹 /clear — מחק זיכרון שיחה\n"
    )

@bot.message_handler(commands=['remind'])
def handle_remind(message):
    user_id = message.from_user.id
    docs_path, db_path, memory_path, profile_path, reminders_path = get_user_paths(user_id)

    text = message.text.replace("/remind", "").strip()
    if not text:
        bot.reply_to(message, "✏️ כתוב מה לזכור:\n/remind פגישה עם דוד ביום שלישי ב-10:00")
        return

    reminder = add_reminder(reminders_path, text)
    bot.reply_to(message, f"✅ תזכורת נשמרה!\n\n*{reminder['id']}.* {text}", parse_mode="Markdown")

@bot.message_handler(commands=['reminders'])
def handle_reminders(message):
    user_id = message.from_user.id
    docs_path, db_path, memory_path, profile_path, reminders_path = get_user_paths(user_id)
    reminders = load_reminders(reminders_path)
    bot.reply_to(message, format_reminders(reminders), parse_mode="Markdown")

@bot.message_handler(commands=['done'])
def handle_done(message):
    user_id = message.from_user.id
    docs_path, db_path, memory_path, profile_path, reminders_path = get_user_paths(user_id)

    try:
        num = int(message.text.replace("/done", "").strip())
        reminders = load_reminders(reminders_path)
        found = False
        for r in reminders:
            if r["id"] == num:
                r["done"] = True
                found = True
                break
        if found:
            save_reminders(reminders_path, reminders)
            bot.reply_to(message, f"✅ תזכורת {num} סומנה כבוצעה!")
        else:
            bot.reply_to(message, f"❌ לא מצאתי תזכורת מספר {num}")
    except:
        bot.reply_to(message, "כתוב: /done 1")

@bot.message_handler(commands=['setprofile'])
def handle_setprofile(message):
    user_id = message.from_user.id
    user_states[user_id] = {"step": 0, "data": {}}
    bot.reply_to(message, f"בוא נכיר! 😊\n\n{PROFILE_QUESTIONS[0][1]}")

@bot.message_handler(commands=['profile'])
def handle_profile(message):
    user_id = message.from_user.id
    docs_path, db_path, memory_path, profile_path, reminders_path = get_user_paths(user_id)
    profile = load_profile(profile_path)
    if not profile:
        bot.reply_to(message, "אין לך פרופיל עדיין. כתוב /setprofile 😊")
        return
    lines = ["👤 *הפרופיל שלך:*\n"]
    fields = {"name": "שם", "job": "עבודה", "hobbies": "תחביבים",
              "family": "משפחה", "goals": "מטרות", "notes": "הערות"}
    for key, label in fields.items():
        if profile.get(key):
            lines.append(f"*{label}:* {profile[key]}")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown")

@bot.message_handler(commands=['clear'])
def handle_clear(message):
    user_id = message.from_user.id
    docs_path, db_path, memory_path, profile_path, reminders_path = get_user_paths(user_id)
    if os.path.exists(memory_path):
        os.remove(memory_path)
    bot.reply_to(message, "🧹 זיכרון השיחה נמחק!")

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    user_id = message.from_user.id
    docs_path, db_path, memory_path, profile_path, reminders_path = get_user_paths(user_id)
    memory = load_memory(memory_path)
    profile = load_profile(profile_path)
    reminders = load_reminders(reminders_path)

    bot.reply_to(message, "🎤 שומע אותך...")
    try:
        file_info = bot.get_file(message.voice.file_id)
        downloaded = bot.download_file(file_info.file_path)
        voice_path = "/tmp/voice.ogg"
        with open(voice_path, "wb") as f:
            f.write(downloaded)

        text = transcribe_voice(voice_path)
        bot.reply_to(message, f"🗣 שמעתי: _{text}_", parse_mode="Markdown")
        lang = detect_language(text)

        if has_pdf(db_path):
            answer = ask_rag(text, db_path, memory, profile, reminders)
        else:
            answer = ask_llm(text, memory, profile, reminders)

        memory.append({"user": text, "bot": answer})
        save_memory(memory_path, memory)
        send_answer(message, answer, lang)

    except Exception as e:
        bot.reply_to(message, f"❌ שגיאה: {str(e)}")

@bot.message_handler(func=lambda message: True)
def handle(message):
    user_id = message.from_user.id
    text = message.text
    docs_path, db_path, memory_path, profile_path, reminders_path = get_user_paths(user_id)

    # Profile setup flow
    if user_id in user_states:
        state = user_states[user_id]
        step = state["step"]
        key = PROFILE_QUESTIONS[step][0]
        if text.lower() != "דלג":
            state["data"][key] = text
        step += 1
        state["step"] = step
        if step < len(PROFILE_QUESTIONS):
            bot.reply_to(message, PROFILE_QUESTIONS[step][1])
        else:
            save_profile(profile_path, state["data"])
            del user_states[user_id]
            name = state["data"].get("name", "")
            bot.reply_to(message, f"✅ תודה {name}! הפרופיל שלך נשמר 😊")
        return

    memory = load_memory(memory_path)
    profile = load_profile(profile_path)
    reminders = load_reminders(reminders_path)
    lang = detect_language(text)

    if has_pdf(db_path):
        answer = ask_rag(text, db_path, memory, profile, reminders)
    else:
        answer = ask_llm(text, memory, profile, reminders)

    memory.append({"user": text, "bot": answer})
    save_memory(memory_path, memory)
    send_answer(message, answer, lang)

@bot.message_handler(content_types=['document'])
def handle_pdf(message):
    user_id = message.from_user.id
    docs_path, db_path, memory_path, profile_path, reminders_path = get_user_paths(user_id)
    file_info = bot.get_file(message.document.file_id)
    downloaded = bot.download_file(file_info.file_path)
    pdf_path = f"{docs_path}/file.pdf"
    with open(pdf_path, "wb") as f:
        f.write(downloaded)
    success = build_rag(pdf_path, db_path)
    if success:
        bot.reply_to(message, "✅ PDF נשמר ואונדקס בהצלחה!")
    else:
        bot.reply_to(message, "❌ לא הצלחתי לקרוא את ה-PDF")

# =========================
# START BOT
# =========================

print("🚀 PERSONAL AGENT RUNNING")
bot.infinity_polling()
