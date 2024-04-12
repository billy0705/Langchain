import requests


url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q3_K_S.gguf"


response = requests.get(url, stream=True)
response.raise_for_status()

with open("llama-2-7b-chat.Q3_K_S.gguf", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("模型已成功下載！")