import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import time
import json
from app import rag_engine, llm, SYSTEM_PROMPT, guard_response

# ---------------------------------------------------------
# COMPREHENSIVE TEST DATASET (50 cases)
# ---------------------------------------------------------

TEST_CASES = [
    # === CATEGORY 1: GREETINGS & LANGUAGE GUARD (5) ===
    {"category": "Greeting_Guard", "name": "Igbo Greeting", "input": "Kekwanu",
     "expected_keywords": ["kedu", "ugbo", "aka"]},
    {"category": "Greeting_Guard", "name": "Yoruba Greeting", "input": "Bawo",
     "expected_keywords": ["ogbin", "ile", "oko"]},
    {"category": "Greeting_Guard", "name": "Hausa Greeting", "input": "Sannu",
     "expected_keywords": ["noma", "gonarku", "kwari"]},
    {"category": "Greeting_Guard", "name": "Pidgin Greeting", "input": "How far",
     "expected_keywords": ["crop", "livestock", "farm"]},
    {"category": "Greeting_Guard", "name": "English Greeting", "input": "Hello",
     "expected_keywords": ["agricultural", "farm", "crops"]},

    # === CATEGORY 2: NON-AGRO GUARD (3) ===
    {"category": "Non_Agro_Guard", "name": "Politics Redirect", "input": "Who do you think will win the election?",
     "expected_keywords": ["agricultural", "crops", "livestock"]},
    {"category": "Non_Agro_Guard", "name": "Football Redirect", "input": "Which football team is the best in Nigeria?",
     "expected_keywords": ["agricultural", "crops", "livestock"]},
    {"category": "Non_Agro_Guard", "name": "Music Redirect", "input": "Who is the best musician in Africa?",
     "expected_keywords": ["agricultural", "crops", "livestock"]},

    # === CATEGORY 3: CROP-SPECIFIC KNOWLEDGE (8) ===
    {"category": "Crop_Knowledge", "name": "Maize Armyworm Kaduna", "input": "I am seeing caterpillars with sawdust-like droppings in my maize in Kaduna. What should I do?",
     "expected_keywords": ["armyworm", "neem", "spray", "early"]},
    {"category": "Crop_Knowledge", "name": "Pidgin Cassava Mosaic", "input": "Abeg, my cassava leaf dey yellow and curl. Wetin fit cause am, and wetin I fit do first make e no spread?",
     "expected_keywords": ["mosaic", "tme", "remove", "burn"]},
    {"category": "Crop_Knowledge", "name": "Pidgin Yam Planting Benue", "input": "Na which month be di best time to plant yam for Benue State, and how I go take prepare my farmland before planting?",
     "expected_keywords": ["ridge", "wood ash", "mancozeb", "rain"]},
    {"category": "Crop_Knowledge", "name": "Cowpea Storage Weevil", "input": "How do I store my cowpea beans so weevils don't destroy them?",
     "expected_keywords": ["weevil", "dry", "container", "neem"]},
    {"category": "Crop_Knowledge", "name": "Groundnut Rosette", "input": "My groundnut leaves are curling and the plants are small. What is wrong?",
     "expected_keywords": ["rosette", "aphid", "early", "samnut"]},
    {"category": "Crop_Knowledge", "name": "Onion Thrips Control", "input": "How do I control thrips in my onion farm?",
     "expected_keywords": ["neem", "insecticide", "cure", "storage"]},
    {"category": "Crop_Knowledge", "name": "Sorghum Striga Weed", "input": "I have purple-flowered weeds destroying my sorghum. What are they and how do I manage them?",
     "expected_keywords": ["striga", "rotate", "cowpea", "resistant"]},
    {"category": "Crop_Knowledge", "name": "Pidgin Plantain Sigatoka", "input": "My plantain leaves get black spots and the bunch dey small. Wetin I fit do?",
     "expected_keywords": ["sigatoka", "fungicide", "remove", "leaf"]},

    # === CATEGORY 4: LIVESTOCK & POULTRY (6) ===
    {"category": "Livestock", "name": "Goat PPR Disease", "input": "My goats are having fever and mouth sores. What disease is this and how do I prevent it?",
     "expected_keywords": ["ppr", "vaccinate", "vet"]},
    {"category": "Livestock", "name": "Newcastle Disease Poultry", "input": "My chickens are dying suddenly with twisted necks. What is happening?",
     "expected_keywords": ["newcastle", "vaccinate", "lasota"]},
    {"category": "Livestock", "name": "Fish Pond Stocking", "input": "How many catfish fingerlings should I stock per square meter in my pond?",
     "expected_keywords": ["10", "15", "fingerling"]},
    {"category": "Livestock", "name": "Cattle Tick Control", "input": "Ticks are all over my cattle. How often should I spray and with what?",
     "expected_keywords": ["acaricide", "2", "weeks"]},
    {"category": "Livestock", "name": "Pidgin Chicken Brooding", "input": "Abeg, how I go take raise my small chicks make dem no die?",
     "expected_keywords": ["warm", "vaccinate", "feed"]},
    {"category": "Livestock", "name": "Rabbit Feeding", "input": "What should I feed my rabbits for fast growth?",
     "expected_keywords": ["forage", "concentrate", "protein"]},

    # === CATEGORY 5: PEST & DISEASE ID (5) ===
    {"category": "Pest_Disease", "name": "Fall Armyworm ID", "input": "What pest leaves sawdust-like frass in maize whorls?",
     "expected_keywords": ["armyworm", "neem", "early"]},
    {"category": "Pest_Disease", "name": "Cassava Mosaic ID", "input": "What does cassava mosaic disease look like?",
     "expected_keywords": ["yellow", "curl", "whitefly"]},
    {"category": "Pest_Disease", "name": "Rice Yellow Mottle", "input": "My rice leaves are yellow with mottled patterns. What virus is this?",
     "expected_keywords": ["rymv", "resistant", "beetle"]},
    {"category": "Pest_Disease", "name": "ASF Pig Disease", "input": "My pigs have high fever and bloody diarrhea. What could this be?",
     "expected_keywords": ["asf", "african swine", "biosecurity"]},
    {"category": "Pest_Disease", "name": "Fruit Fly Mango", "input": "Maggots are inside my mango fruits. What pest is this?",
     "expected_keywords": ["fruit fly", "trap", "bagging"]},

    # === CATEGORY 6: SOIL & FERTILIZER (4) ===
    {"category": "Soil_Fertilizer", "name": "Acidic Soil Lime", "input": "My soil is very acidic. What should I apply?",
     "expected_keywords": ["lime", "ph", "compost"]},
    {"category": "Soil_Fertilizer", "name": "NPK Split Application", "input": "When should I apply NPK to my maize?",
     "expected_keywords": ["3", "6", "weeks"]},
    {"category": "Soil_Fertilizer", "name": "Compost Preparation", "input": "How do I make compost for my farm?",
     "expected_keywords": ["manure", "layer", "turn"]},
    {"category": "Soil_Fertilizer", "name": "Erosion Control Slope", "input": "How do I prevent erosion on my sloped farm?",
     "expected_keywords": ["contour", "mulch", "cover"]},

    # === CATEGORY 7: WEATHER & PLANTING CALENDAR (4) ===
    {"category": "Weather_Calendar", "name": "Planting Time Rainforest", "input": "When should I start planting in the rainforest zone?",
     "expected_keywords": ["march", "rain"]},
    {"category": "Weather_Calendar", "name": "Pidgin Planting Time", "input": "When be di best time to plant for northern Nigeria?",
     "expected_keywords": ["june", "rain", "sahel"]},
    {"category": "Weather_Calendar", "name": "Drought Coping", "input": "What can I do during a dry spell to save my crops?",
     "expected_keywords": ["mulch", "drought", "tolerant"]},
    {"category": "Weather_Calendar", "name": "Rain Onset Signs", "input": "How do I know the rains have started properly for planting?",
     "expected_keywords": ["2-3", "days", "20mm"]},

    # === CATEGORY 8: POST-HARVEST & STORAGE (4) ===
    {"category": "Post_Harvest", "name": "Grain Drying Moisture", "input": "What moisture level should I dry my maize to before storage?",
     "expected_keywords": ["12", "14", "dry"]},
    {"category": "Post_Harvest", "name": "Aflatoxin Prevention", "input": "How do I prevent mold and aflatoxin in my stored grains?",
     "expected_keywords": ["mold", "damp", "dry"]},
    {"category": "Post_Harvest", "name": "Hermetic Storage Bags", "input": "What are the best bags for storing cowpea without weevils?",
     "expected_keywords": ["hermetic", "pics", "airtight"]},
    {"category": "Post_Harvest", "name": "Onion Curing", "input": "How long should I cure my onions after harvest?",
     "expected_keywords": ["10", "14", "shade"]},

    # === CATEGORY 9: MARKET & ECONOMICS (2) ===
    {"category": "Market", "name": "Cooperative Selling", "input": "How can I get better prices for my produce?",
     "expected_keywords": ["cooperative", "bulk", "off-season"]},
    {"category": "Market", "name": "Price Timing", "input": "When is the best time to sell my maize for highest profit?",
     "expected_keywords": ["off-season", "store", "price"]},

    # === CATEGORY 10: REGIONAL MISMATCHES (4) ===
    {"category": "Regional_Mismatch", "name": "Pidgin Apples Kaduna", "input": "i wan plant apple for kaduna i hear sy e dey bring better money",
     "expected_keywords": ["chill", "temperature", "mambilla"]},
    {"category": "Regional_Mismatch", "name": "Potatoes in Rivers", "input": "I want to start commercial Irish potato farming in the swampy parts of Rivers State. Will it work?",
     "expected_keywords": ["rot", "drainage", "jos"]},
    {"category": "Regional_Mismatch", "name": "Cocoa in Sokoto", "input": "Can I grow cocoa in Sokoto State?",
     "expected_keywords": ["rainforest", "south", "cocoa"]},
    {"category": "Regional_Mismatch", "name": "Fish in Jos Plateau", "input": "Is the Jos Plateau good for catfish farming?",
     "expected_keywords": ["water", "pond", "temperature"]},

    # === CATEGORY 11: MULTI-TURN CONVERSATION (3) ===
    {"category": "Multi_Turn", "name": "Follow-up Maize", "input": "You mentioned neem extract for armyworm. How do I prepare it?",
     "expected_keywords": ["neem", "leaf", "spray"]},
    {"category": "Multi_Turn", "name": "Follow-up Cassava", "input": "Where can I get TME 419 cassava stems?",
     "expected_keywords": ["iita", "certified", "stem"]},
    {"category": "Multi_Turn", "name": "Follow-up Goat", "input": "What dewormer should I use for my goats?",
     "expected_keywords": ["deworm", "vet", "every"]},

    # === CATEGORY 12: EDGE CASES (2) ===
    {"category": "Edge_Case", "name": "Very Short Query", "input": "maize",
     "expected_keywords": ["maize", "plant", "spacing"]},
    {"category": "Edge_Case", "name": "Vague Query", "input": "help me with my farm",
     "expected_keywords": ["crop", "ask", "farm"]},
]

# ---------------------------------------------------------
# EVALUATION ENGINE
# ---------------------------------------------------------

def run_evaluation():
    print("\n" + "=" * 70)
    print("COMPREHENSIVE AGRIGEMMA4.9JA EVALUATION SUITE")
    print(f"Total test cases: {len(TEST_CASES)}")
    print("=" * 70 + "\n")

    results = []
    total_time = 0
    total_tokens = 0
    category_scores = {}

    for i, test in enumerate(TEST_CASES, start=1):
        category = test['category']
        print(f"[{i}/{len(TEST_CASES)}] {category}: {test['name']}")

        start_time = time.time()

        # Check guard first
        guard_result = guard_response(test['input'])
        if guard_result:
            response_text = guard_result
            elapsed = time.time() - start_time
            tokens_generated = len(response_text.split())
            tps = tokens_generated / elapsed if elapsed > 0 else 0
        else:
            # Search RAG
            evidence = rag_engine.search(test['input'])
            if evidence:
                evidence_block = "\n\n".join(
                    f"[Master evidence {idx}]\n{content}"
                    for idx, content in enumerate(evidence, start=1)
                )
            else:
                evidence_block = "[No directly matching record was retrieved.]"

            # Construct prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"Retrieved evidence for this farmer case:\n{evidence_block}"},
                {"role": "user", "content": test['input']}
            ]

            # Model inference
            output = llm.create_chat_completion(
                messages=messages,
                temperature=0.20,
                top_p=0.9,
                repeat_penalty=1.12,
                max_tokens=300
            )
            elapsed = time.time() - start_time
            response_text = output["choices"][0]["message"]["content"].strip()
            tokens_generated = output["usage"]["completion_tokens"]
            tps = tokens_generated / elapsed if elapsed > 0 else 0

        total_time += elapsed
        total_tokens += tokens_generated

        # Keyword verification
        lower_resp = response_text.lower()
        matched = [kw for kw in test['expected_keywords'] if kw in lower_resp]
        score = len(matched) / len(test['expected_keywords']) if test['expected_keywords'] else 0

        # Cooldown between cases to prevent thermal throttling
        if i < len(TEST_CASES):
            time.sleep(5)  # 5-second cooldown
            print('    [Cooldown 5s...]')

        # Track category scores
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)

        print(f"    Time: {elapsed:.2f}s | Tokens: {tokens_generated} | TPS: {tps:.2f}")
        print(f"    Score: {score * 100:.0f}% | Matched: {matched}")
        print(f"    Preview: {response_text[:100]}...\n")

        results.append({
            "name": test['name'],
            "category": category,
            "input": test['input'],
            "output": response_text,
            "elapsed_sec": round(elapsed, 2),
            "tokens": tokens_generated,
            "tps": round(tps, 2),
            "keyword_match_pct": round(score * 100, 1),
            "matched_keywords": matched
        })

    # Category summaries
    print("=" * 70)
    print("CATEGORY BREAKDOWN")
    print("=" * 70)
    for cat, scores in category_scores.items():
        avg = sum(scores) / len(scores)
        print(f"  {cat:30s}: {avg * 100:5.1f}%  ({len(scores)} cases)")

    # Overall summary
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    overall_score = sum(r['keyword_match_pct'] for r in results) / len(results) if results else 0

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Total cases:        {len(TEST_CASES)}")
    print(f"  Overall score:      {overall_score:.1f}%")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Average TPS:        {avg_tps:.2f}")
    print("=" * 70 + "\n")

    # Save results
    output = {
        "overall_score_pct": round(overall_score, 1),
        "average_tps": round(avg_tps, 2),
        "total_tokens": total_tokens,
        "total_time_sec": round(total_time, 2),
        "category_breakdown": {cat: round(sum(s) / len(s) * 100, 1) for cat, s in category_scores.items()},
        "results": results
    }

    with open('eval_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()
