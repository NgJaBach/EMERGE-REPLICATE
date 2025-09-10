import json
from tqdm import tqdm
import random
random.seed(42)
import pandas as pd
from openai import OpenAI
import openai
import backoff
import re

client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.BadRequestError, openai.InternalServerError),
    max_time=999, max_tries=9999
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

def ask(user: str, 
        llm_name="gpt-oss:20b", 
        max_token=128000,
        temperature=0.3,
        reasoning_effort="low"
        ) -> str:
    response = completions_with_backoff(
        model=llm_name,
        messages=[
            {"role": "user", "content": user},
        ],
        max_tokens=max_token,
        n=1,
        stop=None,
        temperature=temperature,
        extra_body={"reasoning_effort": reasoning_effort}
    ).choices[0].message.content
    return remove_reasoning(response)

note_merge_prompt_tmpl = """
You are given 2 lists of diseases extracted from two different clinical notes of the same patient. Your task is to merge these two lists into a single list, ensuring that there are no duplicate disease names.

[List A]
{noteA}

[List B]
{noteB}

[Answer]
"""

def merge_note(notaA: str, noteB: str) -> str:
    response = ask(
        note_merge_prompt_tmpl.format(noteA=notaA, noteB=noteB), 
        llm_name="gpt-oss:20b", temperature=0.3, reasoning_effort="low"
    )
    return response

ner_prompt_tmpl = """
[Instruction]
You are tasked with performing Named Entity Recognition (NER) specifically for diseases in a given medical case description to help with healthcare tasks (eg. readmission, motality, length of stay, drug prediction).  Follow the instructions below:
1. Input: You will receive a medical case description in the [Input].
2. NER Task: Focus on extracting the names of diseases as the target entity.
3. Output: Provide the extracted disease names.

Ensure that the output only includes the names of diseases mentioned in the provided [Input], excluding any additional content. The goal is to perform NER exclusively on disease names within the given text.

Example:
[Input]
 1:19 pm abdomen ( supine and erect ) clip  reason : sbo medical condition : 63 year old woman with reason for this examination : sbo final report indication : 63-year-old woman with small bowel obstruction . findings : supine and upright abdominal radiographs are markedly limited due to the patient 's body habitus . on the supine radiograph , there is a large distended loop of bowel in the right mid-abdomen , which extends towards the left . this measures 8 cm . it is difficult to accurately ascertain whether this is small or large bowel . if this is large bowel , a volvulus should be considered . if this is small bowel , it is markedly dilated . other prominent loops of small bowel are seen . additionally on the supine radiograph centered lower , there is a small amount of gas centered over the left femoral head . this could represent an incarcerated left inguinal hernia . impression : limited radiographs due to patient 's body habitus . this is consistent with an obstruction which may or may not involve a large bowel volvulus . these findings were telephoned immediately to the emergency room physician , . , caring for the patient .\",\"12:56 am chest ( portable ap ) clip reason : placement of cvl- r/o ptx , check position admitting diagnosis : small bowel obstruction medical condition : 63 year old woman with reason placement of cvl- r/o ptx , check position admitting diagnosis : small bowel obstruction medical condition : 63 year old woman with reason for this examination : placement of cvl- r/o ptx , check position final report indication : central venous line placement . views : single supine ap view , no prior studies . findings : the endotracheal tube is in satisfactory position approximately 4 cm from the carina . the right internal jugular central venous line is in satisfactory position with tip at the proximal superior vena cava . the study is limited by a lordotic position . low lung volumes are present bilaterally . the heart size appears enlarged . the pulmonary vascularity is difficult to assess . no pneumothorax is identified . no definite pulmonary infiltrates are present . the right costophrenic angle is sharp . the left costophrenic angle is excluded from the study . a nasogastric tube is seen which is looped within the fundus of the stomach with the tip pointing caudad within the distal stomach.
 
[Answer]
"small bowel obstruction", "large bowel volvulus", "incarcerated left inguinal hernia", "lordotic position", "low lung volumes", "enlarged heart", "pneumothorax"]

[Input]
{input}

[Answer]
"""

def extract_note(notes: list[str], llm_name="gpt-oss:20b") -> str:
    all_notes = "\n".join(notes)
    return ask(ner_prompt_tmpl.format(input=all_notes), llm_name=llm_name, temperature=0.3, reasoning_effort="low")

    answer = ""
    for note in tqdm(notes):
        run1 = ask(ner_prompt_tmpl.format(input=note), llm_name=llm_name, temperature=0.3, reasoning_effort="low")
        run2 = ask(ner_prompt_tmpl.format(input=note), llm_name=llm_name, temperature=0.3, reasoning_effort="low")
        merged = merge_note(run1, run2)
        answer = merge_note(answer, merged) if answer else merged
    # room to grow
    return answer


# if __name__ == "__main__":
    # extract_dataset('./mimic4_all/ts_note_all.pkl', './mimic4_all/output6000.json', start_idx=6000+2895, end_idx=9000)