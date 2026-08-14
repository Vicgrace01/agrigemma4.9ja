from pathlib import Path
from llama_cpp import Llama

llm = Llama(
    model_path="model/gemma-4-E2B-it-Q3_K_M.gguf",
    n_ctx=4096,
    n_threads=4,
    verbose=False,
)

tests = {
    "English": "My maize has sawdust-like frass in the whorl. Explain the likely problem in simple English.",
    "Nigerian Pidgin": "Abeg, my maize get sawdust-like thing for inside whorl. Wetin be the problem and wetin I fit do?",
    "Igbo": "Biko, akpụkpa m na-acha odo odo ma akwụkwọ ya na-akụkọta. Kedu ihe nwere ike ịbụ nsogbu ahụ?",
    "Hausa": "Ganyen rogo nawa suna rawaya kuma suna naɗewa. Mene ne matsalar?",
    "Yoruba": "Ewe gbaguda mi n di ofeefee, o si n yi. Kini isoro le je?",
}

output = Path("language_test_results.md")

with output.open("w", encoding="utf-8") as report:
    report.write("# AgriGemma4.9ja language test\n\n")

    for language, prompt in tests.items():
        print(f"\n--- {language} ---")
        result = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an agricultural assistant. Reply only in the "
                        "same language as the user's message. Keep the answer short."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=180,
        )

        answer = result["choices"][0]["message"]["content"].strip()
        print(answer)
        report.write(f"## {language}\n\n**Prompt:** {prompt}\n\n**Response:** {answer}\n\n")

print(f"\nSaved results to {output}")
