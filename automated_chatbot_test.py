import os
import time
import json
import google.generativeai as genai
from chatbot import ProcessChatbot

# Predefined Test Scenarios
TEST_PROMPTS = [
    {
        "category": "General Assessment",
        "question": "Are my processes organized?",
        "expects": "bottleneck analysis, conformance score, process health assessment"
    },
    {
        "category": "Bottleneck Detection",
        "question": "Where is the biggest bottleneck?",
        "expects": "Identify 'SRM: Transfer Failed' or similar delay, mention wait times or frequency"
    },
    {
        "category": "Actionable Suggestions",
        "question": "What should I do to improve throughput?",
        "expects": "Suggest hiring, reassigning, automation, or specific process changes to reduce wait time"
    }
]

class TestEvaluator:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # Use a separate model instance for grading
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def evaluate(self, question, answer, expected_concepts):
        prompt = f"""
        Act as an expert exam grader.
        
        QUESTION: "{question}"
        EXPECTED CONCEPTS: {expected_concepts}
        
        STUDENT ANSWER: "{answer}"
        
        TASK:
        1. Does the Student Answer cover the Expected Concepts? (Synonyms and semantic matches are ACCEPTED).
        2. Is the answer grounded in data (numbers, specific names)?
        
        Output JSON only:
        {{
            "score": <0-100 integer>,
            "reason": "<brief explanation>"
        }}
        """
        try:
            res = self.model.generate_content(prompt)
            text = res.text.strip()
            # Clean JSON markdown
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"): text = text[4:]
            return json.loads(text)
        except Exception as e:
            return {"score": 0, "reason": f"Grading Error: {e}"}

def run_tests():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY missing.")
        return

    print("============================================================")
    print("  Phase 5: Semantic AI Test Suite (LLM-as-a-Judge)")
    print("============================================================")

    # Initialize System Under Test
    try:
        bot = ProcessChatbot(api_key=api_key)
    except Exception as e:
        print(f"[FAIL] Chatbot init failed: {e}")
        return

    # Initialize Grader
    grader = TestEvaluator(api_key)
    
    results = []
    total_score = 0
    
    for i, test in enumerate(TEST_PROMPTS):
        print(f"\n[Test {i+1}/{len(TEST_PROMPTS)}] {test['category']}")
        print(f"Q: {test['question']}")
        
        # 1. Get Chatbot Response
        start = time.time()
        try:
            response = bot.ask(test['question'])
            elapsed = time.time() - start
            print(f"A: [Generated in {elapsed:.1f}s]")
            
            # 2. Grade Response
            print("   -> Grading...")
            grade = grader.evaluate(test['question'], response, test['expects'])
            
            score = grade.get("score", 0)
            reason = grade.get("reason", "No reason provided")
            
            print(f"   -> Score: {score}/100")
            print(f"   -> Reason: {reason}")
            
            results.append({
                "category": test["category"],
                "score": score,
                "reason": reason,
                "response_snippet": response[:100] + "..."
            })
            total_score += score
            
        except Exception as e:
            print(f"[ERROR] Test execution failed: {e}")
            
    avg_score = total_score / len(TEST_PROMPTS)
    print("\n============================================================")
    print(f"  FINAL SEMANTIC SCORE: {avg_score:.1f}/100")
    print("============================================================")
    
    with open("chatbot_test_results.json", "w") as f:
        json.dump({"tests": results, "average_score": avg_score}, f, indent=2)

if __name__ == "__main__":
    run_tests()
