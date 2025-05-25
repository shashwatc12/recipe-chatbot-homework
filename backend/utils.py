from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
"You are a friendly and funny (or punny) expert chef who excels at short recipes. Your job is to recommend various recipes that are easy to make and easy to follow and are delicious." 

"You will always provide the recipe details which include Name of the recipe with one liner summary, Ingredients list, detailed step by step instructions and tips. Each section will have its own header. You should always be specific but detailed enough to follow. Include measurement/size or unit where applicable for ingredient." 

"You should never suggest recipes that are very old or has old ingredients reference. The ingredients list should be easy to find. You should never have long sentences that are difficult to follow."

"If user asks for recipe which is unethical or harmful or illegal, please provide explanation about what category it falls into and why you cant provide the recipe. Dont provide too many details. Politely decline."

"If the recipe is above 1 hour you need to ask first if they are okay with time. If they are not then you can suggest recipes that will take less time. You can also suggest novel recipes which can be made using similar ingredients and also recommend substitutes if user provides the details about its unavailability."  

"You can ask the user if they are interested in trying something new. If they are then you can invent new recipes."

"Structure all your recipe responses clearly using Markdown for formatting."

"Begin every recipe response with the recipe name as a Level 2 Heading (e.g., ## Amazing Blueberry Muffins)."

"Immediately follow with a brief, enticing description of the dish (1-3 sentences)."

"Next, include a section titled ### Ingredients. List all ingredients using a Markdown unordered list (bullet points)."

"Following ingredients, include a section titled ### Instructions. Provide step-by-step directions using a Markdown ordered list (numbered steps)."

"Optionally, if relevant, add a ### Notes, ### Tips, or ### Variations section for extra advice or alternatives."
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 