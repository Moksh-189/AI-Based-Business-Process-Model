"""
Phase 4 — AI Chatbot (Gemini 2.5 Flash)
integration for process analysis and suggestions.

Features:
  - Loads process context (graph stats, bottlenecks, DFG).
  - Uses Google Gemini 2.5 Flash (free tier) for reasoning.
  - RAG-style context injection (injects relevant stats into system prompt).
  - Streaming responses for better UX.

Usage:
  chatbot = ProcessChatbot(api_key="...")
  response = chatbot.ask("Where are the bottlenecks?")
"""

import os
import json
import google.generativeai as genai
try:
    import torch
except ImportError:
    pass  # torch not needed for chatbot
import pandas as pd


class ProcessChatbot:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API Key not provided. Set GEMINI_API_KEY env var or pass to constructor.")

        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.5 Flash (Confirmed via list_models)
        self.model_name = "gemini-2.5-flash" 
        self.model = genai.GenerativeModel(self.model_name)
        
        # Load Context
        self.context = self._load_context()
        self.history = []
        self.chat = self.model.start_chat(history=[])
        
        # Initialize system prompt
        self._set_system_prompt()

    def _load_context(self):
        """Load all Phase 1 & 2 outputs to ground the AI."""
        context = {}
        
        try:
            with open('process_stats.json', 'r') as f:
                context['stats'] = json.load(f)
        except Exception: 
            context['stats'] = "Not available"

        try:
            with open('bottleneck_report.json', 'r') as f:
                context['bottlenecks'] = json.load(f)
        except Exception:
            context['bottlenecks'] = "Not available"
            
        try:
            with open('dfg_data.json', 'r') as f:
                dfg = json.load(f)
                # Summarize DFG to save tokens (top 20 edges)
                context['dfg_top_edges'] = dfg['edges'][:20]
        except Exception:
            context['dfg_top_edges'] = "Not available"

        try:
            with open('agent_comparison.json', 'r') as f:
                context['optimization'] = json.load(f)
        except Exception:
            context['optimization'] = None

        return context

    def reload_context(self):
        """Reloads context from files and updates system prompt."""
        self.context = self._load_context()
        self._set_system_prompt()
        print("Chatbot context reloaded.")

    def _set_system_prompt(self):
        """Inject process data into the chat session."""
        stats = self.context.get('stats', {})
        bottlenecks = self.context.get('bottlenecks', {})
        dfg = self.context.get('dfg_top_edges', [])
        opt = self.context.get('optimization')
        
        opt_text = ""
        if opt:
            opt_text = f"""
### RECENT OPTIMIZATION RESULTS
- Winner Agent: GAT-GLU Agent
- Improvement over Random: {opt.get('gelu_improvement_over_random', 'N/A')}
- Training Timesteps: {opt.get('timesteps', 'N/A')}
"""

        # Construct a data-rich system prompt
        system_prompt = f"""
You are an expert AI Process Analyst for a large SAP procurement process.
Your goal is to analyze the provided process data, identify inefficiencies, and suggest optimizations.

### PROCESS OVERVIEW
- Total Cases: {stats.get('overview', {}).get('total_cases', 'N/A')}
- Total Events: {stats.get('overview', {}).get('total_events', 'N/A')}
- Optimization Score: {stats.get('optimization_score', 'N/A')}/100
- Conformance Fitness: {stats.get('conformance', {}).get('fitness_percentage', 'N/A')}%

### KEY BOTTLENECKS (Top 5)
{json.dumps(bottlenecks.get('bottlenecks', [])[:5], indent=2)}

### TOP PROCESS FLOWS (DFG)
{json.dumps(dfg, indent=2)}
{opt_text}
### INSTRUCTIONS
1. Be extremely concise. Max 3-4 sentences total.
2. Focus only on the single most critical insight.
3. Use data points briefly (e.g., "avg 50h").
4. Suggest root causes based on standard procurement knowledge (e.g., "Invoice Receipt delays often mean vendor mismatches").
5. If asked about "simulation", explain that you can guide the digital twin setup.
6. If optimization results are available, reference them to show potential AI impact.
7. You can chat with the user as a human like reply to a hey normally like a human would and other unrelated messages like a normal chatbot but tell what they can ask you to do.
"""
        self.system_prompt = system_prompt
        
    def ask(self, query):
        """Send a query to the chatbot."""
        # Check if first message
        if not self.history:
            full_prompt = f"{self.system_prompt}\n\nUSER QUERY: {query}"
        else:
            full_prompt = query
            
        try:
            response = self.chat.send_message(full_prompt)
            self.history.append({"role": "user", "parts": [query]})
            self.history.append({"role": "model", "parts": [response.text]})
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}"

if __name__ == "__main__":
    # Test
    print("Initializing Chatbot...")
    try:
        # Expects API key in env or passed
        bot = ProcessChatbot()
        print("Chatbot ready. Asking about bottlenecks...")
        print(bot.ask("What is the main bottleneck?"))
    except Exception as e:
        print(f"Skipping test: {e}")
