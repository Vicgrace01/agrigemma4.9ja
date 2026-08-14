import csv
import json
import re
from collections import Counter, defaultdict

import gradio as gr
from llama_cpp import Llama

MODEL_PATH = "model/gemma-4-E2B-it-Q3_K_M.gguf"
DATABASE_PATH = "naerls_verified.csv"
JSON_KB_PATH = "master_agro_kb.json"
TOP_K = 3
MAX_HISTORY_TURNS = 2

print("Loading AgriGemma4.9ja...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4,
    n_batch=512,
    verbose=False,
)

PIDGIN_TERMS = {
    "abeg": "please",
    "dey": "is are",
    "wetin": "what",
    "wahala": "problem issue",
    "yellow": "yellowing chlorosis",
    "rot": "rotting disease decay fungal",
    "worm": "caterpillar pest armyworm larvae insect",
    "sawdust": "frass armyworm maize damage",
    "spray": "control treatment application pesticide",
    "farm": "crop field land agriculture",
    "nko": "what next advice",
    "e": "it",
    "oga": "farmer sir",
}

STOPWORDS = {
    "a", "an", "and", "are", "be", "do", "for", "how", "i", "in", "is",
    "it", "my", "next", "of", "on", "or", "please", "step", "the", "to",
    "what", "wetin", "with", "you", "your", "will", "can", "should",
    "that", "this", "they", "them", "then", "after", "about", "quite"
}

# Nigerian greetings and smalltalk across major languages
NIGERIAN_GREETINGS = {
    # English
    "hello": "Hello! I am AgriGemma4.9ja, your agricultural extension officer. Ask me about crops, livestock, soil, pests, or weather for your farm.",
    "hi": "Hello! I am AgriGemma4.9ja, your agricultural extension officer. Ask me about crops, livestock, soil, pests, or weather for your farm.",
    "good morning": "Good morning! I am AgriGemma4.9ja, your agricultural extension officer. Ask me about crops, livestock, soil, pests, or weather for your farm.",
    "good afternoon": "Good afternoon! I am AgriGemma4.9ja, your agricultural extension officer. Ask me about crops, livestock, soil, pests, or weather for your farm.",
    "good evening": "Good evening! I am AgriGemma4.9ja, your agricultural extension officer. Ask me about crops, livestock, soil, pests, or weather for your farm.",
    "thanks": "You're welcome! I'm here to help with your farming questions. What else can I assist with?",
    "thank you": "You're welcome! I'm here to help with your farming questions. What else can I assist with?",
    "nice": "I'm glad you found that helpful! What else would you like to know about farming?",
    "hmmm nice": "Glad you liked it! What else would you like to know about farming?",
    "great": "Awesome! Let me know if you have any other farming questions.",
    "ok": "Alright! What farming topic can I help you with?",
    "good": "Great! How can I assist with your farm today?",
    "fair": "I see. What specific farming advice do you need today?",
    "hmm": "I understand. What agricultural question do you have?",
    # Nigerian Pidgin
    "how far": "How far! Na AgriGemma4.9ja. You fit ask me about crop, livestock, soil, pest, or weather for your farm.",
    "how you dey": "I dey fine, thank you! Na AgriGemma4.9ja. You fit ask me about crop, livestock, soil, pest, or weather for your farm.",
    "thank you o": "You welcome o! Any other farming wahala I fit help you with?",
    "well done": "Thank you! How can I help you with your farming today?",
    "well done chief": "Thank you, Chief! What farming advice can I give you today?",
    "well done oga": "Thank you, Oga! How can I assist with your farm?",
    "na wa o": "Na wa o! But seriously, how can I help with your farming?",
    "e be like": "I see. What agricultural question do you have?",
    "you dey": "I dey! How can I help you with your farm today?",
    "oga": "Yes, Oga! What farming advice do you need?",
    "chief": "Yes, Chief! How can I assist with your farm?",
    "how body": "My body dey fine! How you dey? Wetin I fit help you with for your farm?",
    "how e dey go": "E dey go well! How can I help you with your farming today?",
    "e dey go": "E dey go well! What farming advice do you need?",
    "no wahala": "No wahala! What farming question you get for me?",
    "wetin dey": "How far! Wetin I fit help you with for your farm?",
    "wetin dey sup": "How far! Wetin I fit help you with for your farm?",
    "sup": "How far! Wetin I fit help you with for your farm?",
    # Yoruba
    "bawo": "E kaaro! Mo wa lati ran e lowo pelu ogbin, eran, ile, kokoro, tabi oju-ọjọ fun oko rẹ.",
    "e kaaro": "E kaaro! Mo wa lati ran e lowo pelu ogbin, eran, ile, kokoro, tabi oju-ọjọ fun oko rẹ.",
    "e kaasan": "E kaasan! Mo wa lati ran e lowo pelu ogbin, eran, ile, kokoro, tabi oju-ọjọ fun oko rẹ.",
    "e kurole": "E kurole! Mo wa lati ran e lowo pelu ogbin, eran, ile, kokoro, tabi oju-ọjọ fun oko rẹ.",
    # Hausa
    "sannu": "Sannu! Ina nan don taimaka muku da noma, dabbobi, ƙasa, kwari, ko yanayi na gonarku.",
    "ina kwana": "Ina kwana! Ina nan don taimaka muku da noma, dabbobi, ƙasa, kwari, ko yanayi na gonarku.",
    "barka da rana": "Barka da rana! Ina nan don taimaka muku da noma, dabbobi, ƙasa, kwari, ko yanayi na gonarku.",
    # Igbo
    "kekwanu": "Kedu! A bịara m inyere gị aka na ọrụ ugbo, anụmanụ, ala, ahụhụ, ma ọ bụ ihu igwe maka ugbo gị.",
    "nnoo": "Nnọọ! A bịara m inyere gị aka na ọrụ ugbo, anụmanụ, ala, ahụhụ, ma ọ bụ ihu igwe maka ugbo gị.",
    "daalu": "Daalu! A bịara m inyere gị aka na ọrụ ugbo, anụmanụ, ala, ahụhụ, ma ọ bụ ihu igwe maka ugbo gị.",
}

# Non-agricultural topics to gracefully decline
NON_AGRO_TOPICS = {
    "music", "football", "soccer", "politics", "president", "election",
    "movie", "film", "celebrity", "gossip", "fashion", "crypto",
    "bitcoin", "shopping", "phone", "laptop", "school",
    "exam", "homework", "history", "geography", "math", "science",
    "joke", "poem", "story", "riddle", "game", "sport", "music"
}

# Broad agricultural terms covering crops, livestock, poultry, pests, soil, weather
AGRI_TERMS = {
    # Crops
    "maize", "corn", "yam", "cassava", "tomato", "rice", "pepper",
    "cowpea", "beans", "okra", "plantain", "banana", "cocoa", "potato",
    "sorghum", "millet", "groundnut", "soybean", "onion", "garlic",
    "ginger", "turmeric", "cabbage", "carrot", "spinach",
    # Livestock
    "swine", "pig", "poultry", "chicken", "cattle", "fish",
    "goat", "sheep", "rabbit", "snail", "grasscutter", "cane rat",
    "turkey", "duck", "quail", "guinea fowl",
    # Livestock Products
    "meat", "egg", "milk", "leather", "wool",
    # Poultry
    "breed", "broiler", "layer", "cockerel", "pullet", "feed",
    "fowl", "egg production", "hatchery", "incubation",
    # General
    "farming", "agriculture", "extension", "advisory", "farm",
    "livestock", "animal", "herd", "flock", "pen", "cage",
    # Pests & Diseases
    "pest", "disease", "infestation", "parasite", "worm",
    "armyworm", "fall armyworm", "cassava mosaic", "mosaic",
    # Weather & Soil
    "rainfall", "drought", "flood", "soil", "fertilizer",
    "manure", "compost", "irrigation", "drainage"
}

# Common-sense terms for filtering non-agricultural questions
COMMON_SENSE_TERMS = {
    "fly", "flying", "swim", "swimming", "walk", "walking",
    "human", "person", "people", "animal", "water", "sky",
    "color", "colour", "size", "height", "weight", "shape",
    "exist", "real", "true", "false", "possible", "impossible",
    "bird", "insect", "reptile", "mammal", "planet", "earth",
    "moon", "sun", "star", "space"
}

COMMON_SENSE_PATTERNS = [
    r"does .+ (fly|swim|walk|eat|sleep|live)",
    r"can .+ (fly|swim|walk|eat|sleep|live)",
    r"what is (a|an) .+",
    r"is .+ (real|true|false|possible)",
    r"do .+ (fly|swim|walk|eat|sleep|live)",
    r"what .+ (mean|means)",
]

# Greeting patterns for Gemma fallback
GREETING_PATTERNS = [
    r"well\s+done",
    r"how\s+far",
    r"how\s+you\s+dey",
    r"na\s+wa\s+o",
    r"you\s+dey",
    r"e\s+be\s+like",
    r"oga",
    r"chief",
    r"hello",
    r"hi",
    r"hey",
    r"sannu",
    r"bawo",
    r"kedu",
    r"nnoo",
    r"thank\s+you",
    r"thanks",
    r"good\s+morning",
    r"good\s+afternoon",
    r"good\s+evening",
    r"wetin\s+dey",
    r"wetin\s+dey\s+sup",
    r"sup",
    r"what's\s+up",
    r"how\s+body",
    r"how\s+e\s+dey",
    r"e\s+dey\s+go",
    r"no\s+wahala",
    r"you\s+welcom",
    r"how\s+you",
]

def is_greeting(message):
    """Check if message is a simple greeting in any supported language."""
    lower_message = message.lower().strip().rstrip("!.?")
    for greeting, response in NIGERIAN_GREETINGS.items():
        if lower_message == greeting or lower_message.startswith(greeting + " "):
            return response
    return None

def is_non_agro(message):
    """Detect clearly non-agricultural queries."""
    tokens = set(tokenize(message))
    return bool(tokens & NON_AGRO_TOPICS)

def detect_non_agro_topic(message):
    """Detect what non-agro topic the user is asking about."""
    lower_message = message.lower()

    topic_map = {
        "politics": ["election", "president", "governor", "senator", "politician", "pdp", "apc"],
        "football": ["football", "soccer", "ball", "goal", "player", "team"],
        "music": ["music", "singer", "song", "album", "musician", "concert"],
        "entertainment": ["movie", "film", "actor", "actress", "cinema", "nollywood"],
        "fashion": ["fashion", "clothes", "dress", "style", "shoe"],
        "technology": ["phone", "computer", "laptop", "software", "app", "website"],
        "finance": ["bitcoin", "crypto", "stock", "invest", "money", "bank"],
        "education": ["school", "exam", "homework", "study", "university"],
        "health": ["hospital", "doctor", "medicine", "fever", "headache"],
        "religion": ["church", "mosque", "bible", "quran", "prayer"],
        "relationships": ["love", "marriage", "girlfriend", "boyfriend", "dating"],
        "food": ["recipe", "cook", "restaurant", "food"],
        "travel": ["travel", "flight", "hotel", "visa", "passport", "airport"],
        "comics": ["spider-man", "spiderman", "superman", "batman", "avengers", "marvel"],
        "gaming": ["playstation", "xbox", "fortnite", "fifa", "nintendo", "video game"],
        "coding": ["python", "javascript", "coding", "programming", "debug", "html"],
        "general_knowledge": ["capital of", "history", "geography", "science", "math", "space"],
        "weather_forecast": ["weather forecast", "rain tomorrow", "temperature today", "will it rain"],
    }

    for topic, keywords in topic_map.items():
        for kw in keywords:
            if kw in lower_message:
                return topic

    return None

# ✅ FIXED: Use Gemma to understand user intent
def gemma_understand_intent(message):
    """Use Gemma to analyze what the user is saying."""
    
    prompt = f"""You are AgriGemma4.9ja's intent classifier. Analyze what the user is saying.

User message: "{message}"

Classify into ONE of these categories:
- greeting: User is greeting you (hello, hi, how far, well done, sannu, etc.)
- agriculture: User is asking about crops, livestock, soil, pests, weather, or farming
- follow_up: User is asking a follow-up question about previous conversation
- common_sense: User is asking a general knowledge question
- non_agro: User is asking about something NOT related to agriculture (cooking, politics, sports, etc.)
- vague: User said something very short or unclear

Reply with ONLY the category name (one word)."""
    
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10,
            stop=["\n"],
        )
        
        # ✅ FIX: Handle different response formats
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'delta' in choice and 'content' in choice['delta']:
                intent = choice['delta']['content'].strip().lower()
            elif 'message' in choice and 'content' in choice['message']:
                intent = choice['message']['content'].strip().lower()
            elif 'text' in choice:
                intent = choice['text'].strip().lower()
            else:
                print(f"Unknown response format: {choice}")
                return "vague"
        else:
            print(f"Unexpected response structure: {response}")
            return "vague"
        
        return intent
    except Exception as e:
        print(f"Intent detection failed: {e}")
        return "vague"

# ✅ FIXED: Generate greeting with Gemma
def generate_greeting_with_gemma(message):
    """Use Gemma to generate a warm greeting response."""
    
    prompt = f"""You are AgriGemma4.9ja, a friendly agricultural advisor.

User just said: "{message}"

This is a greeting. Respond warmly and naturally, then gently redirect to farming topics.
Keep it short and friendly (under 30 words)."""
    
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=50,
        )
        
        # ✅ FIX: Handle different response formats
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'delta' in choice and 'content' in choice['delta']:
                return choice['delta']['content'].strip()
            elif 'message' in choice and 'content' in choice['message']:
                return choice['message']['content'].strip()
            elif 'text' in choice:
                return choice['text'].strip()
        
        return "Hello! How can I help with your farming today?"
    except Exception as e:
        print(f"Greeting generation failed: {e}")
        return "Hello! How can I help with your farming today?"

def is_potential_greeting(message):
    """Check if message looks like a greeting (for fallback)"""
    lower = message.lower().strip()
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False

# ✅ UPDATED: Guard with Gemma collaboration
def guard_response(message, history=None):
    """Return a graceful response if the message is a greeting or off-topic, else None."""
    
    # 1. Check greetings in dictionary (fast path)
    greeting_response = is_greeting(message)
    if greeting_response:
        return greeting_response

    # 2. Check non-agro in dictionary (fast path)
    non_agro_topic = detect_non_agro_topic(message)
    if non_agro_topic:
        return (
            f"You are asking about {non_agro_topic}. That is not my area. "
            "I am AgriGemma4.9ja, built specifically to help Nigerian farmers. "
            "Ask me about crops, livestock, soil, pests, or weather for your farm."
        )
    
    # 3. Check if very short/no context
    tokens = set(tokenize(message))
    if len(tokens) <= 2 and not (tokens & AGRI_TERMS):
        if not history:
            return (
                "Could you tell me more about what you want to know? "
                "I am here to help with crops, livestock, soil, pests, or weather for your farm."
            )
        # Let Gemma handle with context
    
    # 4. Check if follow-up question
    if history and is_follow_up(message, history):
        return None
    
    # 5. Use Gemma to understand intent for unknown cases
    if is_potential_greeting(message) or len(tokens) <= 4:
        try:
            intent = gemma_understand_intent(message)
            
            if intent == "greeting":
                return generate_greeting_with_gemma(message)
            
            elif intent == "non_agro":
                return (
                    "That doesn't seem to be about farming. "
                    "I'm here to help with crops, livestock, soil, pests, or weather. "
                    "What agricultural question do you have?"
                )
            
            elif intent == "vague":
                return (
                    "I'm not quite sure what you're asking. "
                    "Could you tell me more about your farming question?"
                )
            
            # If agriculture, follow_up, or common_sense, let it through
            return None
            
        except Exception as e:
            print(f"Intent detection failed: {e}")
            return None  # Fall through to normal flow
    
    return None

def is_common_sense_question(message):
    """Check if the question is a general knowledge question, not agricultural"""
    lower = message.lower()
    
    for pattern in COMMON_SENSE_PATTERNS:
        if re.search(pattern, lower):
            return True
    
    tokens = set(tokenize(lower))
    if tokens & COMMON_SENSE_TERMS:
        if not (tokens & AGRI_TERMS):
            return True
    
    return False

NEW_TOPIC_MARKERS = {
    "new question", "different question", "another question",
    "change topic", "switch topic", "i now want to ask",
    "let me ask about", "i want to ask about"
}

FOLLOW_UP_MARKERS = {
    "next", "what next", "after that", "then", "what should i do",
    "what do i do", "how do i stop it", "will it work", "will this work",
    "is it serious", "serious", "spreading", "everywhere", "worse",
    "getting worse", "rain", "neem", "spray", "treatment", "control",
    "how much", "when should", "can i", "should i", "e dey spread",
    "e serious", "wetin next", "wetin i go do", "nko"
}

SYSTEM_PROMPT = """You are AgriGemma4.9ja, an authoritative, professional Nigerian agricultural extension officer advising smallholder farmers.

NIGERIAN AGRO-ECOLOGICAL KNOWLEDGE BASE:
- Sahel/Sudan Savanna (e.g., Sokoto, Borno, Kano, Jigawa): Short rainfall (May-Sept), long dry season, high heat. Sandy/loam soils. Best crops: Millet, sorghum, cowpea, onions, groundnut.
- Guinea Savanna / Middle Belt (e.g., Kaduna, Benue, Kogi, Niger): Moderate rainfall (Apr-Oct). Loamy soils. Best crops: Yam, maize, soybean, cassava, sorghum.
  *High Altitude Exception: Plateau State (Jos) and Taraba (Mambilla) have cold temperatures/chill hours. Only these areas naturally support temperate crops like Irish potato, apples, strawberries, and tea.
- Rainforest Zone (e.g., Oyo, Enugu, Imo, Edo, Ogun): Heavy rainfall (Mar-Nov). Clay/loam soils. Best crops: Cassava, oil palm, cocoa, plantain, yam.
- Swamp/Mangrove Zone (e.g., Rivers, Bayelsa, Delta, Akwa Ibom): Very heavy rainfall, high humidity, waterlogged/acidic soils. Best crops: Aquaculture (fish farming), rice, rubber, plantain.

RULES:
1. Speak with absolute regional authority based on the Agro-Ecological Knowledge Base above. Never use vague disclaimers like "assess your farm's microclimate" or "get a soil test". State climatic mismatches directly.
2. If a crop is biologically unsuited for a state, explicitly state that and point out the correct high-altitude zones.
3. Match the user's language naturally. If they ask in Nigerian Pidgin, reply in direct, clean Pidgin. If in English, reply in concise, practical English.
4. Eliminate conversational filler and generic headers. Go straight to actionable advice.
5. Use retrieved NAERLS and master knowledge base evidence as your primary source.
6. Never invent precise pesticide doses, official citations, real-time weather forecasts, or market prices.
7. Continue the active farm case when the message refers to it. Switch only when a new crop or topic is introduced.
8. If the retrieved evidence directly answers the farmer's question, give that answer immediately.
9. Always complete your thoughts and sentences fully within your token limit.
10. For common-sense questions (e.g., "does fish fly"), answer directly without overcomplicating.

HANDLING GREETINGS AND SMALL TALK:
- If a user greets you in English, Pidgin, Yoruba, Hausa, or Igbo, respond warmly and naturally.
- For Pidgin greetings like "well done chief", "how far", "na wa o", respond in friendly Pidgin.
- After greeting, gently redirect to agricultural topics.
- Example: "well done chief" → "Thank you, Chief! How can I help with your farming today?"
- Example: "how far" → "How far! Wetin I fit help you with for your farm?"
"""

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("value") or "")
    if isinstance(content, list):
        parts = [extract_text(item) for item in content]
        return " ".join(part for part in parts if part)
    return str(content or "")

def tokenize(text):
    return [
        word for word in re.findall(r"[a-z0-9']+", text.lower())
        if word not in STOPWORDS and len(word) > 1
    ]

def expanded_terms(text):
    terms = tokenize(text)
    expanded = list(terms)
    for term in terms:
        expanded.extend(tokenize(PIDGIN_TERMS.get(term, "")))
    return expanded

class FastLocalRAG:
    """Offline inverted-index RAG engine using multi-word matching."""

    def __init__(self, csv_path, json_path=None):
        self.documents = []
        self.keyword_index = defaultdict(set)
        self.load_data(csv_path)
        if json_path:
            self.load_json(json_path)

    def load_data(self, csv_path):
        with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames or not {"keywords", "content"}.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path} must contain usable 'keywords' and 'content' columns.")

            for row in reader:
                keywords = (row.get("keywords") or "").strip()
                content = (row.get("content") or "").strip()

                if not content:
                    continue

                actual_id = len(self.documents)
                self.documents.append({
                    "keywords": Counter(tokenize(keywords)),
                    "content": content,
                    "source": "csv",
                })

                for word in set(tokenize(keywords)):
                    self.keyword_index[word].add(actual_id)

        if not self.documents:
            raise ValueError(f"{csv_path} has no usable records.")

        print(f"RAG loaded: indexed {len(self.documents)} master records.")
    
    def load_json(self, json_path):
        """Load concise advisory records from master_agro_kb.json."""
        with open(json_path, mode="r", encoding="utf-8") as file:
            entries = json.load(file)

        if not isinstance(entries, list):
            raise ValueError(f"{json_path} must contain a list of records.")

        added = 0
        for entry in entries:
            keywords = (entry.get("keywords") or "").strip()
            content = (entry.get("content") or "").strip()

            if not content:
                continue

            actual_id = len(self.documents)
            self.documents.append({
                "keywords": Counter(tokenize(keywords)),
                "content": content,
                "source": "json",
            })

            for word in set(tokenize(keywords)):
                self.keyword_index[word].add(actual_id)

            added += 1

        if added > 0:
            print(f"RAG loaded: indexed {added} concise KB records from {json_path}.")

    def search(self, query, top_k=TOP_K):
        query_counts = Counter(expanded_terms(query))
        if not query_counts:
            return []

        candidate_ids = set()
        for word in query_counts:
            candidate_ids.update(self.keyword_index.get(word, set()))

        scored = []
        for doc_id in candidate_ids:
            document = self.documents[doc_id]

            keyword_score = sum(
                query_counts[word] * count
                for word, count in document["keywords"].items()
            )

            matched_unique_words = sum(1 for word in query_counts if word in document["keywords"])

            if keyword_score >= 2 and matched_unique_words >= 2:
                match_ratio = matched_unique_words / max(1, len(query_counts))
                if document.get("source") == "json":
                    score = 1000 + (match_ratio * 50) + keyword_score
                else:
                    score = (match_ratio * 50) + keyword_score
                scored.append((score, document["content"]))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [content for score, content in scored[:top_k]]

rag_engine = FastLocalRAG(DATABASE_PATH, JSON_KB_PATH)

def history_to_messages(history):
    messages = []
    for item in (history or [])[-MAX_HISTORY_TURNS * 2:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = extract_text(item.get("content"))
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages

def recent_user_context(history):
    messages = []
    for item in reversed(history or []):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = extract_text(item.get("content"))
        if content:
            messages.append(content)
        if len(messages) == MAX_HISTORY_TURNS:
            break
    return "\n".join(reversed(messages))

def agri_mentions(text):
    return set(tokenize(text)) & AGRI_TERMS

def is_follow_up(message, history):
    message = extract_text(message)
    lower_message = message.lower().strip()
    context = recent_user_context(history)

    if not context:
        return False

    current_agri = agri_mentions(message)
    old_agri = agri_mentions(context)

    if current_agri and old_agri and not current_agri.issubset(old_agri):
        return False

    if any(marker in lower_message for marker in NEW_TOPIC_MARKERS):
        return False

    if any(marker in lower_message for marker in FOLLOW_UP_MARKERS):
        return True

    message_terms = set(expanded_terms(message))
    context_terms = set(expanded_terms(context))

    if message_terms & context_terms:
        return True

    if not current_agri and len(message_terms) <= 8:
        return True

    return False

def retrieval_query(message, history):
    if is_follow_up(message, history):
        return f"{recent_user_context(history)}\nCurrent follow-up: {message}"
    return message

# ✅ UPDATED: Generate response with cooking check
def generate_response(message, history):
    message = extract_text(message).strip()
    
    # ✅ NEW: Check for cooking/food prep questions (non-agro)
    cooking_keywords = ["fry", "cook", "recipe", "boil", "grill", "roast", "bake", "season", "spice", "stew", "soup"]
    if any(kw in message.lower() for kw in cooking_keywords):
        if "fish" in message.lower() or "meat" in message.lower() or "chicken" in message.lower():
            yield "That sounds like a cooking question! I'm an agricultural advisor, not a chef. I can help with fish farming, harvesting, processing, and storage, but for cooking recipes, you might want to ask a cookbook instead. What farming or agricultural question do you have?"
            return
    
    # Guard first (static responses for known greetings + Gemma for unknown)
    guard_result = guard_response(message, history)
    if guard_result:
        yield guard_result
        return
    
    # Common-sense questions (skip RAG)
    if is_common_sense_question(message):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history_to_messages(history),
            {"role": "user", "content": message},
        ]
        
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=0.20,
            top_p=0.9,
            repeat_penalty=1.12,
            max_tokens=150,
            stream=True,
        )
        
        partial_message = ""
        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                partial_message += delta['content']
                yield partial_message.strip()
        return
    
    # Normal agricultural query with RAG
    search_query = retrieval_query(message, history)
    evidence = rag_engine.search(search_query)

    if evidence:
        evidence_block = "\n\n".join(
            f"[Master evidence {number}]\n{content[:500]}..." if len(content) > 500 else f"[Master evidence {number}]\n{content}"
            for number, content in enumerate(evidence, start=1)
        )
    else:
        evidence_block = (
            "[No directly matching record retrieved. Use "
            "general agricultural reasoning carefully.]"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"Retrieved evidence for this farmer case:\n{evidence_block}",
        },
        *history_to_messages(history),
        {"role": "user", "content": message},
    ]

    stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.20,
        top_p=0.9,
        repeat_penalty=1.12,
        max_tokens=220,
        stream=True,
    )

    partial_message = ""
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            partial_message += delta['content']
            yield partial_message.strip()

def user_submit(user_message, history):
    user_message = extract_text(user_message).strip()
    history = list(history or [])
    if not user_message:
        return "", history
    history.append({"role": "user", "content": user_message})
    return "", history

def bot_reply(history):
    history = list(history or [])
    if not history or history[-1].get("role") != "user":
        return history

    user_message = extract_text(history[-1].get("content"))
    history.append({"role": "assistant", "content": ""})

    for partial_response in generate_response(user_message, history[:-2]):
        history[-1]["content"] = partial_response
        yield history

def save_session(history, sessions, active_session):
    sessions = dict(sessions or {})
    active_session = active_session or "Chat 1"
    sessions[active_session] = list(history or [])
    return sessions

def switch_session(selected_session, sessions):
    sessions = sessions or {}
    selected_session = selected_session or "Chat 1"
    return sessions.get(selected_session, []), selected_session

def create_new_session(sessions):
    sessions = dict(sessions or {})
    chat_number = 1
    while f"Chat {chat_number}" in sessions:
        chat_number += 1
    new_name = f"Chat {chat_number}"
    sessions[new_name] = []
    return (
        gr.update(choices=list(sessions.keys()), value=new_name),
        [],
        sessions,
        new_name,
        gr.update(visible=True),
    )

def add_example(text, history):
    return user_submit(text, history)

def suggestions_visible(history):
    return gr.update(visible=not bool(history))

custom_css = """
footer { visibility: hidden; }
.sidebar {
    background: #090d16;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #1e293b;
}
.message.user {
    background: #1e293b !important;
    color: #f8fafc !important;
}
.message.bot {
    background: #0f172a !important;
    color: #e2e8f0 !important;
}
"""

app_theme = gr.themes.Base(primary_hue="emerald", neutral_hue="slate").set(
    body_background_fill="#030712",
    background_fill_primary="#0f172a",
    block_background_fill="#0b0f19",
    block_label_text_color="#10b981",
)

header_html = """
<div style="display: flex; align-items: center; gap: 14px; padding: 12px 6px; margin-bottom: 6px;">
  <div style="background: #065f46; padding: 8px; border-radius: 12px; display: flex; align-items: center; justify-content: center; border: 1px solid #10b981; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 2L3 7v10l9 5 9-5V7l-9-5z" stroke="#10b981" fill="#065f46" fill-opacity="0.4"></path>
      <path d="M12 17v-7M12 10C10.5 8.5 9 7.5 8 8c-1 .5-1 2 0 3.5l4 3.5 4-3.5c1-1.5 1-3 0-3.5-1-.5-2.5.5-4 2z" fill="#34d399" stroke="#34d399"></path>
    </svg>
  </div>
  <div>
    <h2 style="color: #f8fafc; font-size: 1.6rem; margin: 0; font-weight: 700; letter-spacing: -0.025em;">
      AgriGemma4.9ja
    </h2>
    <p style="color: #94a3b8; font-size: 0.92rem; margin: 2px 0 0;">
      Offline Nigerian agricultural extension advisory powered by Gemma &amp; RAG
    </p>
  </div>
</div>
"""

with gr.Blocks(title="AgriGemma4.9ja") as demo:
    sessions_db = gr.State({"Chat 1": []})
    active_session = gr.State("Chat 1")

    with gr.Sidebar(open=True):
        gr.Markdown("#### Chats")
        session_selector = gr.Dropdown(
            choices=["Chat 1"],
            value="Chat 1",
            show_label=False,
            interactive=True,
        )
        new_chat_btn = gr.Button("New farm conversation", variant="secondary")

    with gr.Column():
        gr.HTML(header_html)
        chatbot = gr.Chatbot(height=470)

        with gr.Column(visible=True) as suggestions_pane:
            gr.Markdown("**Suggested farm questions**")
            with gr.Row():
                maize_example = gr.Button("Caterpillars and frass in maize", variant="secondary")
                yam_example = gr.Button("Yellow yam leaves with green veins", variant="secondary")

        with gr.Row():
            message_box = gr.Textbox(
                placeholder="Ask about crops, pests, or livestock... (Max 250 characters)",
                show_label=False,
                scale=5,
                max_length=250,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)

    def submit_chain():
        return (user_submit, [message_box, chatbot], [message_box, chatbot])

    message_box.submit(*submit_chain(), queue=False).then(
        bot_reply, chatbot, chatbot
    ).then(
        save_session, [chatbot, sessions_db, active_session], sessions_db
    ).then(
        suggestions_visible, chatbot, suggestions_pane
    )

    send_button.click(*submit_chain(), queue=False).then(
        bot_reply, chatbot, chatbot
    ).then(
        save_session, [chatbot, sessions_db, active_session], sessions_db
    ).then(
        suggestions_visible, chatbot, suggestions_pane
    )

    session_selector.change(
        switch_session,
        [session_selector, sessions_db],
        [chatbot, active_session],
    ).then(
        suggestions_visible, chatbot, suggestions_pane
    )

    new_chat_btn.click(
        create_new_session,
        sessions_db,
        [session_selector, chatbot, sessions_db, active_session, suggestions_pane],
    )

    maize_example.click(
        add_example,
        [gr.State("I am seeing caterpillars with sawdust-like droppings in my maize in Kaduna. What should I do?"), chatbot],
        [message_box, chatbot],
        queue=False,
    ).then(
        bot_reply, chatbot, chatbot
    ).then(
        save_session, [chatbot, sessions_db, active_session], sessions_db
    ).then(
        suggestions_visible, chatbot, suggestions_pane
    )

    yam_example.click(
        add_example,
        [gr.State("My yam leaves are turning yellow with green veins in Benue state. What is wrong?"), chatbot],
        [message_box, chatbot],
        queue=False,
    ).then(
        bot_reply, chatbot, chatbot
    ).then(
        save_session, [chatbot, sessions_db, active_session], sessions_db
    ).then(
        suggestions_visible, chatbot, suggestions_pane
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=app_theme,
        css=custom_css,
    )
