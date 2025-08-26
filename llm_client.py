import os
from typing import List, Dict

# Minimal wrapper. If no OPENAI_API_KEY, return a canned response.
# You can drop in your preferred LLM client later.

async def get_empathetic_reply(emotion_label: str, topk: List[Dict], recent_transcript: str, model="gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # Fallback text so the UI still works
        templates = {
            "happy": "I’m glad you’re feeling good! Want to share what’s going well?",
            "sad": "That sounds really tough. I’m here to listen—what’s weighing on you most right now?",
            "angry": "I can hear the frustration. What do you wish could change about this situation?",
            "fear": "Feeling worried is understandable. What’s the biggest fear on your mind?",
            "disgust": "It’s okay to feel put off. What part felt most wrong to you?",
            "surprise": "That’s unexpected! How are you feeling about it now?",
            "neutral": "I’m here—tell me what’s on your mind.",
        }
        return templates.get(emotion_label, templates["neutral"])

    # Example: you can integrate official OpenAI SDK here if desired.
    # For now, just return the same fallback to keep the sample self-contained.
    return f"(LLM placeholder) I’m picking up {emotion_label}. Tell me more—what would help right now?"
