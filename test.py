import requests
import json

url = "http://192.168.100.205:11434/api/generate" 
data = { "model": "deepseek-v2:16b", "prompt": "Xin chào từ Ollama", "stream": False } 
resp = requests.post(url, json=data) 
raw_text = resp.text
data = json.loads(raw_text)
print("Model:", data["model"])
print("Created At:", data["created_at"])
print("Response:", data["response"])
print("Done:", data["done"])
print("Done Reason:", data["done_reason"])
print("Context length:", len(data["context"]))
print("First 5 context tokens:", data["context"][:5])
print("Total Duration:", data["total_duration"])