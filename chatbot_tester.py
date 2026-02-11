"""
Phase 4 Checklist — Chatbot Tester
Simple script to interact with the ProcessChatbot and verify its responses.
"""

import sys
import time
from chatbot import ProcessChatbot

def main():
    print("="*60)
    print("  AI Process Analyst — Testing Interface")
    print("="*60)
    
    # Check API Key
    api_key = None
    # Try loading from env or args
    import os
    if "GEMINI_API_KEY" in os.environ:
        api_key = os.environ["GEMINI_API_KEY"]
    else:
        print("\n[!] GEMINI_API_KEY not found in environment variables.")
        api_key = input("    Please enter your Google Gemini API Key: ").strip()
        if not api_key:
            print("    [ERROR] API Key required.")
            return

    try:
        print("\n[INFO] Initializing Chatbot (loading graph context)...")
        bot = ProcessChatbot(api_key=api_key)
        print("[SUCCESS] Chatbot ready!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize chatbot: {e}")
        return

    print("\n------------------------------------------------------------")
    print("Type 'exit' to quit. Type 'score' to rate last response.")
    print("------------------------------------------------------------")

    last_response = ""
    
    while True:
        try:
            query = input("\n[USER]: ").strip()
            if not query:
                continue
                
            if query.lower() in ('exit', 'quit'):
                break
                
            if query.lower() == 'score':
                if not last_response:
                    print("    No response to score yet.")
                    continue
                try:
                    score = float(input("    Rate response (0-100): "))
                    print(f"    [Recorded] Score: {score}/100")
                    # In a real app, we'd log this for RLHF or improvement
                except ValueError:
                    print("    Invalid score.")
                continue

            print("\n[AI]: Thinking...", end="\r")
            start = time.time()
            response = bot.ask(query)
            elapsed = time.time() - start
            
            print(f"[AI] ({elapsed:.1f}s):\n")
            print(response)
            last_response = response
            print("-" * 60)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")

    print("\nGoodbye!")

if __name__ == "__main__":
    main()
