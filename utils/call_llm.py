import json
import backoff
import re
from .prompts import *
from .api_keys import *
from .constants import *
from .logging import log_to_file, reset_file
import requests
from typing import Dict, Any, Optional
from .secrets import BAILAB_HTTP

class OllamaError(Exception):
    """Custom exception for Ollama API errors"""
    pass

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, OllamaError),
    max_time=999, max_tries=9999
)
def ollama_completion_with_backoff(**kwargs) -> Dict[str, Any]:
    """Make a request to Ollama API with exponential backoff"""
    try:
        response = requests.post(BAILAB_HTTP, json=kwargs)
        response.raise_for_status()
        return json.loads(response.text)
    except requests.exceptions.HTTPError as e:
        raise OllamaError(f"HTTP error: {e}")
    except json.JSONDecodeError:
        raise OllamaError("Invalid JSON response")

def remove_reasoning(response_content: str) -> str:
    """Remove reasoning part if present"""
    match = re.search(r"</think>\s*(.*)", response_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response_content.strip()

def ask(
    user_prompt: str,
    sys_prompt: str = "",
    model_name: str = "deepseek-v2:16b",
    max_tokens: int = 64000,
    temperature: float = 0.3,
    reasoning_level: Optional[str] = None,
) -> str:
    """Generate text using Ollama API with similar interface to OpenAI"""
    # Combine system and user prompts if system prompt is provided
    prompt = f"{sys_prompt}\n\n{user_prompt}" if sys_prompt else user_prompt
    
    # Prepare request parameters
    params = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": max_tokens
        }
    }
    
    # Add reasoning parameters if specified
    if reasoning_level:
        params["options"]["reasoning_effort"] = reasoning_level
    
    # Make the API call
    response = ollama_completion_with_backoff(**params)
    
    # Extract and clean response
    result = response.get("response", "")
    response = remove_reasoning(result)
    return response

def extract_note(notes: str, llm = "qwen2.5:7b-instruct", ) -> str:
    def refine_note(extracted: str) -> str:
        response = ask(user_prompt=refine_prompt_tmpl.format(text=notes, entities=extracted), model_name=llm)
        return response
    
    def merge_note(notaA: str, noteB: str) -> str:
        response = ask(user_prompt=note_merge_prompt_tmpl.format(noteA=notaA, noteB=noteB), model_name=llm)
        refined = refine_note(response)
        return refined

    res1 = refine_note(ask(user_prompt=ner_prompt_tmpl.format(text=notes), model_name=llm))
    res2 = refine_note(ask(user_prompt=ner_prompt_tmpl.format(text=notes), model_name=llm))
    answer = merge_note(res1, res2)
    # room to grow
    return answer

def create_summary(ehr, notes, nodes, edges, llm = "deepseek-v2:16b") -> str:
    prompt = summary_prompt_tmpl.format(ehr=ehr, notes=notes, nodes=nodes, edges=edges)
    response = ask(user_prompt=prompt, model_name=llm, reasoning_level="low")
    # log_to_file(LLM_LOG, f"=== Prompt ===\nModel: {llm}\n{prompt}\n=== Response ===\n{response}\n", show_time=True)
    return response