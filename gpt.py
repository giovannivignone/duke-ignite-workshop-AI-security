import openai
import os
from dotenv import load_dotenv
import json


# Load your OpenAI API key from an environment variable or secret management service
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Configure the OpenAI library with your API key
openai.api_key = api_key

def query_gpt_35_turbo(prompt):
    response = openai.ChatCompletion.create(
        messages=[{"role": "system",
                   "content": "You are a classification engine with the ability to determine whether network logs appear suspicious."},
                  {"role": "user", "content": prompt}],
        model="gpt-3.5-turbo"
    )
    text = response.choices[0].message.content
    return text.lower().replace("\n", "").replace(".", "")

def classify_network_log(log):
    prompt = f"Given the following network log, classify it as normal, suspicious, or malicious. Only return a single word: {log}"
    response = query_gpt_35_turbo(prompt)
    return response

if __name__ == "__main__":
    network_logs_path = 'network_log_example.json'
    network_logs = None
    with open(network_logs_path, 'r') as f:
        network_logs = json.load(f)
    attack_log = network_logs[0]
    normal_log = network_logs[1]
    print(classify_network_log(attack_log))
    print(classify_network_log(normal_log))
    pass