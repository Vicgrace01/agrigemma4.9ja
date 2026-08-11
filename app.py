import gradio as gr
import csv
import re
from collections import defaultdict
from llama_cpp import Llama

print("Loading Llama 3B Engine...")
llm = Llama(
    model_path="model/llama-3.2-3b.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4
)

# --- SCALABLE INVERTED INDEX RAG ---
class FastLocalRAG:
    def __init__(self, csv_path):
        self.documents = []
        self.index = defaultdict(set)
        self.load_data(csv_path)

    def load_data(self, csv_path):
        try:
            with open(csv_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for doc_id, row in enumerate(reader):
                    self.documents.append(row['content'])
                    words = set(re.findall(r'\w+', row['keywords'].lower()))
                    for word in words:
                        self.index[word].add(doc_id)
            print(f"RAG Loaded: Indexed {len(self.documents)} NAERLS records.")
        except FileNotFoundError:
            print(f"Warning: {csv_path} not found. RAG is running empty.")

    def search(self, query):
        if not query:
            return ""
        query_words = set(re.findall(r'\w+', query.lower()))
        scores = defaultdict(int)

        for word in query_words:
            if word in self.index:
                for doc_id in self.index[word]:
                    scores[doc_id] += 1

        if not scores:
            return ""

        best_doc_id = max(scores, key=scores.get)
        if scores[best_doc_id] >= 2:
            return self.documents[best_doc_id]
        return ""

rag_engine = FastLocalRAG("naerls_database.csv")

# --- BULLETPROOF TEXT EXTRACTOR ---
def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        res = []
        for item in content:
            if isinstance(item, str):
                res.append(item)
            elif isinstance(item, dict):
                res.append(item.get("text", str(item)))
            else:
                res.append(str(item))
        return " ".join(res)
    return str(content)


# --- CORE GENERATION ENGINE ---
def generate_response(message, history):
    message = extract_text(message)
    local_fact = rag_engine.search(message)

    system_instruction = (
        "You are AgriGemma, an expert agricultural AI built specifically for Nigerian farmers. "
        "Always match the user's language naturally. If they speak Nigerian Pidgin (e.g., 'i hear say na crop...'), reply fluently and warmly in natural Nigerian Pidgin. "
        "If they speak English, reply in clear, professional English.\n"
    )

    if local_fact:
        system_instruction += (
            f"VERIFIED LOCAL FACT: {local_fact}\n"
            "CRITICAL CROP CHECK: Ensure this fact strictly matches the specific crop the user is asking about. "
            "Do not apply cassava diseases to maize, yam diseases to tomatoes, or vice versa. "
            "Use this verified record to ground your advice, but speak naturally as an expert farmer."
        )
    else:
        system_instruction += (
            "You do not have a local RAG record for this specific query, so use your own expert training knowledge freely. "
            "CRITICAL CLIMATE RULE: You must critically evaluate Nigeria's regional climate compatibility before providing crop advice. "
            "If a crop cannot realistically grow in that specific state's climate (e.g., standard apples requiring winter chill hours in hot southern states like Enugu, or cool-weather crops in dry northern heat), explicitly and clearly warn the farmer why it will fail, and suggest profitable local alternatives."
        )

    prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_instruction}<|eot_id|>"

    history = history or []
    recent_history = history[-4:] if len(history) > 4 else history

    for msg in recent_history:
        role = msg.get("role", "user")
        content = extract_text(msg.get("content", ""))
        if content:
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.45,
        top_p=0.9,
        repeat_penalty=1.18,
        stop=["<|eot_id|>"],
        echo=False
    )
    return output['choices'][0]['text']


# --- STATE & CHAT SESSION LOGIC ---
def user_submit(user_message, history):
    user_message = extract_text(user_message)
    history = history or []
    if not user_message.strip():
        return "", history

    history.append({"role": "user", "content": user_message})
    return "", history

def bot_reply(history):
    history = history or []
    if not history or history[-1].get("role") != "user":
        return history

    user_message = extract_text(history[-1].get("content", ""))

    bot_message = generate_response(user_message, history[:-1])
    history.append({"role": "assistant", "content": bot_message})
    return history

def update_session_state(history, session_dict, current_session):
    history = history or []
    session_dict = session_dict or {}
    current_session = current_session or "Chat 1"
    session_dict[current_session] = history
    return session_dict

def switch_session(selected_session, session_dict):
    session_dict = session_dict or {}
    selected_session = selected_session or "Chat 1"
    return session_dict.get(selected_session, [])

def create_new_session(session_dict):
    session_dict = session_dict or {}
    new_session_name = f"Chat {len(session_dict) + 1}"
    session_dict[new_session_name] = []
    return gr.update(choices=list(session_dict.keys()), value=new_session_name), [], session_dict, new_session_name

def load_ex1(history):
    history = history or []
    history.append({"role": "user", "content": "I am seeing caterpillars with sawdust-like droppings in my maize in Kaduna. What should I spray?"})
    return "", history

def load_ex2(history):
    history = history or []
    history.append({"role": "user", "content": "My yam leaves are turning yellow with green veins in Benue state. What is wrong?"})
    return "", history

def toggle_suggestions(history):
    return gr.update(visible=not bool(history))


# --- SLEEK MODERN CHATBOT STYLING ---
custom_css = """
footer { visibility: hidden; }
#component-0 { max-width: 1000px; margin: auto; padding-top: 10px; }
.message.user {
    background: #1e293b !important;
    color: #f8fafc !important;
    border-radius: 16px 16px 4px 16px !important;
    border: 1px solid #334155 !important;
}
.message.bot {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border-radius: 16px 16px 16px 4px !important;
    border: 1px solid #1e293b !important;
}
.sidebar {
    background: #090d16;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #1e293b;
}
"""

enterprise_theme = gr.themes.Base(primary_hue="emerald", neutral_hue="slate").set(
    body_background_fill="#030712",
    background_fill_primary="#0f172a",
    block_background_fill="#0b0f19",
    block_label_text_color="#10b981"
)

dashboard_header = """
<div style="text-align: left; padding: 10px 5px; margin-bottom: 5px;">
    <h2 style="color: #f8fafc; font-size: 1.5rem; font-weight: 700; margin: 0; display: flex; align-items: center; gap: 8px;">
        🌿 AgriGemma
    </h2>
    <p style="color: #64748b; font-size: 0.95rem; margin-top: 4px; margin-bottom: 0;">What are we cultivating today?</p>
</div>
"""

with gr.Blocks() as demo:
    sessions_db = gr.State({"Chat 1": []})
    active_session = gr.State("Chat 1")

    with gr.Sidebar(open=True):
        gr.Markdown("#### 💬 Chats")
        session_selector = gr.Dropdown(choices=["Chat 1"], value="Chat 1", show_label=False, interactive=True)
        new_chat_btn = gr.Button("➕ New Chat", variant="secondary", size="sm")

    with gr.Column():
        gr.HTML(dashboard_header)

        chatbot = gr.Chatbot(
            height=460,
            avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/6202/6202850.png"),
        )

        with gr.Column(visible=True) as suggestions_pane:
            gr.Markdown("<span style='color: #64748b; font-size: 0.9rem;'>💡 Suggested Queries:</span>")
            with gr.Row():
                ex1_btn = gr.Button("🐛 Caterpillars in maize in Kaduna...", variant="secondary", size="sm")
                ex2_btn = gr.Button("🍂 Yam leaves turning yellow in Benue...", variant="secondary", size="sm")

        with gr.Row():
            msg = gr.Textbox(placeholder="Ask AgriGemma anything about crops, livestock, or soil...", show_label=False, scale=5)
            submit_btn = gr.Button("Send", variant="primary", scale=1)

    chatbot.change(
        toggle_suggestions,
        inputs=[chatbot],
        outputs=[suggestions_pane]
    )

    submit_event = msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_reply, chatbot, chatbot
    ).then(
        update_session_state, [chatbot, sessions_db, active_session], sessions_db
    )

    submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_reply, chatbot, chatbot
    ).then(
        update_session_state, [chatbot, sessions_db, active_session], sessions_db
    )

    session_selector.change(
        switch_session, [session_selector, sessions_db], chatbot
    ).then(
        lambda x: x, session_selector, active_session
    )

    new_chat_btn.click(
        create_new_session, sessions_db, [session_selector, chatbot, sessions_db, active_session]
    )

    ex1_btn.click(load_ex1, chatbot, [msg, chatbot], queue=False).then(
        bot_reply, chatbot, chatbot
    ).then(
        update_session_state, [chatbot, sessions_db, active_session], sessions_db
    )

    ex2_btn.click(load_ex2, chatbot, [msg, chatbot], queue=False).then(
        bot_reply, chatbot, chatbot
    ).then(
        update_session_state, [chatbot, sessions_db, active_session], sessions_db
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=enterprise_theme,
        css=custom_css
    )
