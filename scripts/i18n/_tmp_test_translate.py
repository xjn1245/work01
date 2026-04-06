import os,json
from openai import OpenAI
api=os.getenv("DASHSCOPE_API_KEY")
client=OpenAI(api_key=api, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
texts=["会话ID设置为:","会话字段按数字人形象"]
langs=["en","ja","ko"]
payload={"texts":texts,"targets":langs}
system=("你是一名专业翻译器。任务：把输入的中文字符串分别翻译成 English / Japanese / Korean。要求：只输出合法 JSON，不要输出任何解释。"
        "JSON 结构：{"\n  \"en\": {\"中文原文\": \"英文翻译\"},"\n  \"ja\": {\"中文原文\": \"日本語翻訳\"},"\n  \"ko\": {\"中文原文\": \"한국어 번역\"}"\n}")
user="请翻译并返回 JSON：\n"+json.dumps(payload,ensure_ascii=False)
resp=client.chat.completions.create(model="qwen-turbo",messages=[{"role":"system","content":system},{"role":"user","content":user}],stream=False,temperature=0.0,max_tokens=800)
content=resp.choices[0].message.content or ""
print(content.encode("unicode_escape").decode("ascii"))
