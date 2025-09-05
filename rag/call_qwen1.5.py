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

def ask(user: str, llm_name="gpt-oss:20b", max_token=128000) -> str:
    response = completions_with_backoff(
        model=llm_name,
        messages=[
            {"role": "user", "content": user},
        ],
        max_tokens=max_token,
        n=1,
        stop=None,
        temperature=0.3,
        extra_body={"reasoning_effort": "low"}
    ).choices[0].message.content
    return remove_reasoning(response)

ner_prompt_tmpl = """
[Instruction]
You are tasked with performing Named Entity Recognition (NER) specifically for diseases in a given medical case description to help with healthcare tasks (eg. readmission, motality, length of stay, drug prediction).  Follow the instructions below:
1. Input: You will receive a medical case description in the [Input].
2. NER Task: Focus on extracting the names of diseases as the target entity.
3. Output: Provide the extracted disease names in JSON format.

Ensure that the JSON output only includes the names of diseases mentioned in the provided [Input], excluding any additional content. The goal is to perform NER exclusively on disease names within the given text.

Example:
[Input]
 1:19 pm abdomen ( supine and erect ) clip  reason : sbo medical condition : 63 year old woman with reason for this examination : sbo final report indication : 63-year-old woman with small bowel obstruction . findings : supine and upright abdominal radiographs are markedly limited due to the patient 's body habitus . on the supine radiograph , there is a large distended loop of bowel in the right mid-abdomen , which extends towards the left . this measures 8 cm . it is difficult to accurately ascertain whether this is small or large bowel . if this is large bowel , a volvulus should be considered . if this is small bowel , it is markedly dilated . other prominent loops of small bowel are seen . additionally on the supine radiograph centered lower , there is a small amount of gas centered over the left femoral head . this could represent an incarcerated left inguinal hernia . impression : limited radiographs due to patient 's body habitus . this is consistent with an obstruction which may or may not involve a large bowel volvulus . these findings were telephoned immediately to the emergency room physician , . , caring for the patient .\",\"12:56 am chest ( portable ap ) clip reason : placement of cvl- r/o ptx , check position admitting diagnosis : small bowel obstruction medical condition : 63 year old woman with reason placement of cvl- r/o ptx , check position admitting diagnosis : small bowel obstruction medical condition : 63 year old woman with reason for this examination : placement of cvl- r/o ptx , check position final report indication : central venous line placement . views : single supine ap view , no prior studies . findings : the endotracheal tube is in satisfactory position approximately 4 cm from the carina . the right internal jugular central venous line is in satisfactory position with tip at the proximal superior vena cava . the study is limited by a lordotic position . low lung volumes are present bilaterally . the heart size appears enlarged . the pulmonary vascularity is difficult to assess . no pneumothorax is identified . no definite pulmonary infiltrates are present . the right costophrenic angle is sharp . the left costophrenic angle is excluded from the study . a nasogastric tube is seen which is looped within the fundus of the stomach with the tip pointing caudad within the distal stomach.
 
[Answer]
```json
{
"entities": ["small bowel obstruction",
            "large bowel volvulus",
            "incarcerated left inguinal hernia",
            "lordotic position",
            "low lung volumes",
            "enlarged heart",
            "pneumothorax"]
}
```

[Input]
{input}

[Answer]
"""

def parse_json(s: str) -> dict:
    lines = s.split('\n')
    p_start, p_end = -1, -1
    for idx, line in enumerate(lines):
        if '```' in line:
            if p_start == -1: p_start = idx
            else: p_end = idx
    try:
        if p_start == -1:
            p_start = 0
            p_end = len(lines)-1
        obj = json.loads('\n'.join(lines[p_start+1: p_end]))
    except:
        obj = {}
    return obj

def LLM_single(prompt: str):
    data = {
        "inputs": prompt
    }
    json_data = json.dumps(data)
    response = ask(user=json_data, llm_name="gpt-oss:20b")
    response_data = response.json()
    item = response_data["outputs"]
    obj = parse_json(item)
    try:
        entities = obj.get("entities", [])
    except:
        entities = []
    if len(entities) > 0:
        return entities
    return []

def extract_dataset(input_pkl, output_json):
    window_thresh = 2000  # qwen 8192,留一些生成文本的空间
    lines = pd.read_pickle(input_pkl)
    with open(output_json, 'a+', encoding='utf8') as fout:
        for idx, obj in enumerate(tqdm(lines)):
            words = obj['Texts'].split()
            records = [' '.join(words[i:i+window_thresh]) for i in range(0, len(words), window_thresh)]
            print(f"split into {len(records)} chunks")
            sample_entities = []
            if len(records) > 4:
                records = random.sample(records, 4)  # 太多了，推理巨慢
            for r in records:  # for each record
                ner_prompt = ner_prompt_tmpl.replace('{input}', r)
                record_entities = LLM_single(ner_prompt)
                sample_entities += record_entities  # 核心代码
            new_obj = {
                'PatientID': obj['PatientID'],
                'Entities': sample_entities,
                'Texts': obj['Texts']
            }
            l = json.dumps(new_obj, ensure_ascii=False)
            print(l, file=fout, flush=True)



if __name__ == "__main__":
    extract_dataset('./mimic4_all/ts_note_all.pkl', './mimic4_all/output6000.json', start_idx=6000+2895, end_idx=9000)