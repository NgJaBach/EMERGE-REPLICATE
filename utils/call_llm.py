import json
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import openai
import backoff
import re
from .prompts import *
from .api_keys import *
from .constants import *
from .logging import log_to_file, reset_file

# client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
# client = OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.BadRequestError, openai.InternalServerError),
    # max_time=999, max_tries=9999
    max_time=0, max_tries=0
)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def remove_reasoning(response_content: str) -> str:
    match = re.search(r"</think>\s*(.*)", response_content, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        return final_answer
    else:
        return response_content.strip()

def ask(user_prompt: str, 
        sys_prompt: str = "",
        llm_name="gpt-oss:20b", 
        max_token=20000,
        temperature=0.3,
        reasoning_level=None, # ["low", "medium", "high"]
        ) -> str:
    if reasoning_level == None:
        response = completions_with_backoff(
            model=llm_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_token,
            n=1,
            stop=None,
            temperature=temperature,
        ).choices[0].message.content
    else:
        response = completions_with_backoff(
            model=llm_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_token,
            n=1,
            stop=None,
            temperature=temperature,
            reasoning={"effort": reasoning_level},
            extra_body={"reasoning_effort": reasoning_level}
        ).choices[0].message.content
    response = remove_reasoning(response)
    log_to_file(LLM_LOG, f"=== Prompt ===\nModel: {llm_name}\n{user_prompt}\n=== Response ===\n{response}\n", show_time=True)
    return response

def extract_note(notes: str, llm = "qwen/qwen3-4b:free") -> str:
    def refine_note(extracted: str) -> str:
        response = ask(user_prompt=refine_prompt_tmpl.format(text=notes, entities=extracted), llm_name=llm)
        return response
    
    def merge_note(notaA: str, noteB: str) -> str:
        response = ask(user_prompt=note_merge_prompt_tmpl.format(noteA=notaA, noteB=noteB), llm_name=llm)
        refined = refine_note(response)
        return refined
    
    res1 = ask(user_prompt=ner_prompt_tmpl.format(input=notes), llm_name=llm)
    res2 = ask(user_prompt=ner_prompt_tmpl.format(input=notes), llm_name=llm)
    answer = refine_note(merge_note(res1, res2))
    # room to grow
    return answer

def create_summary(ehr, notes, nodes, edges, llm = "deepseek/deepseek-chat-v3-0324:free") -> str:
    response = ask(user_prompt=summary_prompt_tmpl.format(ehr=ehr, notes=notes, nodes=nodes, edges=edges), llm_name=llm)
    return response