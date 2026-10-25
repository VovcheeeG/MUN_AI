import os
import re
import torch
import faiss
import numpy as np
import io
import PyPDF2
import sqlite3
import secrets
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from config_lora import BASE_MODEL_PATH, OUTPUT_DIR

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ============================================================
#  БАЗА ДАННЫХ ДЛЯ АГЕНТОВ
# ============================================================
def init_db():
    conn = sqlite3.connect('mun_agents.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS agents
                 (api_key TEXT PRIMARY KEY, agent_name TEXT, system_prompt TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

def create_agent(agent_name: str, system_prompt: str):
    api_key = f"mun-{secrets.token_hex(16)}"
    conn = sqlite3.connect('mun_agents.db')
    c = conn.cursor()
    c.execute("INSERT INTO agents (api_key, agent_name, system_prompt) VALUES (?, ?, ?)",
              (api_key, agent_name, system_prompt))
    conn.commit()
    conn.close()
    return api_key

def get_agent(api_key: str):
    conn = sqlite3.connect('mun_agents.db')
    c = conn.cursor()
    c.execute("SELECT agent_name, system_prompt FROM agents WHERE api_key = ?", (api_key,))
    row = c.fetchone()
    conn.close()
    return row

# ============================================================
#  КЛАССИФИКАТОР
# ============================================================
LEGAL_WORDS = [
    'закон', 'право', 'статья', 'суд', 'кодекс', 'юрист', 'штраф',
    'ответственность', 'иск', 'договор', 'претензи', 'жалоб',
    'трудовой', 'уголовн', 'административн', 'гражданск',
    'потребител', 'возврат', 'гаранти', 'ущерб', 'банкрот',
    'зп', 'з/п', 'аванс', 'премия', 'оклад', 'зарплат', 'увольнен', 'отпуск', 'наследств', 'алимент',
    'работодатель', 'начальник', 'аренда', 'арендодатель', 'квартира', 'жилье', 'выселить',
    'дтп', 'авария', 'магазин', 'телефон', 'ноутбук',
    'могу ли я', 'имею ли я', 'имеет ли право', 'можно ли', 'законно ли',
    'задерживать', 'задержать', 'не платят', 'не выплачивают', 'обманули',
    'налог', 'усн', 'ндфл', 'ндс', 'ип', 'предприниматель', 'деклараци', 'бухгалтер'
]

def is_legal_question(prompt: str) -> bool:
    p = prompt.lower()
    return any(re.search(rf'\b{word}', p) for word in LEGAL_WORDS)

# ============================================================
#  RAG (ПОИСК)
# ============================================================
print('Загрузка поиска по законам...')
embed_model = SentenceTransformer('intfloat/multilingual-e5-large')
index = faiss.read_index('law_index.faiss')

with open('law_metadata.tsv', 'r', encoding='utf-8', errors='ignore') as f:
    law_meta = [line.strip().split('\t') for line in f]
    law_meta = [m for m in law_meta if len(m) == 4]

print(f'Загружено статей: {len(law_meta)}')

def search_law(query, k=10, mode="normal"):
    # === ФИКС: в обычном режиме RAG не используем вообще ===
    if mode != "lawyer":
        return []

    query_vec = embed_model.encode([f"query: {query}"], normalize_embeddings=True).astype('float32')
    distances, indices = index.search(query_vec, k)

    q = query.lower()

    # 1. Банкротство
    is_bankrot_query = any(w in q for w in ['банкрот', 'долг', 'кредит', 'физлицо', 'должник'])
    is_physical = any(w in q for w in ['физлицо', 'обычный человек', 'гражданин', 'физ. лицо', 'физлиц'])
    if is_bankrot_query:
        bankrot_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(law_meta):
                code, article, link, text = law_meta[idx]
                if '127-ФЗ' in code:
                    if is_physical and article.startswith('213'):
                        bankrot_results.insert(0, {'code': code, 'article': article, 'link': link, 'text': text})
                    else:
                        bankrot_results.append({'code': code, 'article': article, 'link': link, 'text': text})
        if bankrot_results:
            return bankrot_results[:3]

    # 2. Налоги
    is_tax_query = any(w in q for w in ['налог', 'усн', 'ндфл', 'ндс', 'ип', 'предприниматель', 'деклараци', 'бухгалтер'])
    if is_tax_query:
        tax_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(law_meta):
                code, article, link, text = law_meta[idx]
                if 'Налоговый кодекс' in code:
                    if any(w in q for w in ['усн', 'упрощен', '6%']) and article.startswith('346'):
                        tax_results.insert(0, {'code': code, 'article': article, 'link': link, 'text': text})
                    else:
                        tax_results.append({'code': code, 'article': article, 'link': link, 'text': text})
        if tax_results:
            return tax_results[:3]

    # 3. Медосвидетельствование — только если явно про 12.26
    is_med_query = any(w in q for w in ['12.26', 'статья 12.26', 'медосвидетельствование'])
    if is_med_query:
        for m in law_meta:
            if 'Кодекс об административных правонарушениях' in m[0] and m[1] == '12.26':
                return [{'code': m[0], 'article': m[1], 'link': m[2], 'text': m[3]}]

    # 4. Кредит
    is_credit_query = any(w in q for w in ['кредит', 'займ', 'банк'])
    if is_credit_query:
        for m in law_meta:
            if 'Гражданский кодекс' in m[0] and m[1] == '819':
                return [{'code': m[0], 'article': m[1], 'link': m[2], 'text': m[3]}]

    # 5. Обычный поиск
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(law_meta) and distances[0][i] < 0.65:
            code, article, link, text = law_meta[idx]
            good_chars = len(re.findall(r'[а-яА-Яa-zA-Z0-9\s.,;:!?()\[\]{}"\'«»\-—]', text))
            if len(text) > 0 and (good_chars / len(text)) > 0.7:
                results.append({'code': code, 'article': article, 'link': link, 'text': text})

    return results[:3]

# ============================================================
#  ЗАГРУЗКА МОДЕЛИ
# ============================================================
print('Загрузка токенизатора...')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print(f'Загрузка базы в 4-bit: {BASE_MODEL_PATH}')
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, quantization_config=bnb_config, device_map='cpu', trust_remote_code=True)

print(f'Загрузка LoRA: {OUTPUT_DIR}')
model = PeftModel.from_pretrained(model, OUTPUT_DIR)
model.eval()

print('Модель загружена.')

# ============================================================
#  ПРОМПТЫ
# ============================================================
GENERAL_PROMPT = """Ты — Мун, универсальный ассистент. Создатель — Шагин Владимир Олегович.
ПРАВИЛА:
1. Отвечай на русском языке.
2. Будь краток, но по делу.
3. НЕ ВЫДУМЫВАЙ факты. Если не уверен, скажи: "Я не уверен, лучше уточнить".
4. Не давай опасных советов.
5. Если вопрос про политику или религию — вежливо откажись.
"""

# ============================================================
#  API
# ============================================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    mode: str = "normal"

class GenResponse(BaseModel):
    text: str
    source_url: Optional[str] = None
    source_title: Optional[str] = None

class AgentCreateRequest(BaseModel):
    name: str
    prompt: str

@app.post('/generate', response_model=GenResponse)
def generate(req: GenRequest, authorization: str = Header(None)):
    try:
        agent_prompt = GENERAL_PROMPT
        if authorization:
            api_key = authorization.replace("Bearer ", "")
            agent = get_agent(api_key)
            if agent:
                agent_name, agent_prompt = agent
            else:
                raise HTTPException(status_code=401, detail="Неверный API-ключ")

        # === ФИКС: НЕ приводим весь промпт к lower, иначе не найдём "Пользователь:" ===
        prompt_raw = re.sub(r'^мун:\s*', '', req.prompt, flags=re.IGNORECASE)

        if req.mode == "lawyer":
            legal = True
            query_for_search = prompt_raw.lower()
        else:
            # Ищем разделитель в исходном регистре
            if "Пользователь:" in prompt_raw:
                query_for_search = prompt_raw.split("Пользователь:")[-1].strip()
            elif "пользователь:" in prompt_raw.lower():
                # fallback: режем по нижнему регистру
                low = prompt_raw.lower()
                idx = low.rfind("пользователь:")
                query_for_search = prompt_raw[idx + len("пользователь:"):].strip()
            else:
                query_for_search = prompt_raw
            legal = is_legal_question(query_for_search.lower())

        # РЕЖИМ 1: ЮРИДИЧЕСКИЙ
        if legal:
            found_laws = search_law(query_for_search, k=10, mode=req.mode)
            if not found_laws:
                return GenResponse(text="К сожалению, в базе законов не найдено подходящей статьи. Рекомендую обратиться к юристу.")

            answer_parts = []
            for law in found_laws:
                source_link = f"https://pravo.gov.ru/proxy/ips/?docbody=&nd={law['link']}"
                answer_parts.append(
                    f"Согласно **{law['code']}**, **Статья {law['article']}**:\n\n"
                    f"> {law['text']}\n\n"
                    f"**Источник:** {law['code']}, Статья {law['article']}\n"
                    f"🔗 [Проверить на pravo.gov.ru]({source_link})"
                )
            return GenResponse(text="\n\n---\n\n".join(answer_parts))

        # РЕЖИМ 2: ОБЫЧНЫЙ
        messages = [{'role': 'system', 'content': agent_prompt}, {'role': 'user', 'content': req.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p, do_sample=True, pad_token_id=tokenizer.eos_token_id)

        answer = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        answer = re.sub(r'^мун:\s*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'[^\x00-\x7F\u0400-\u04FF\s.,;:!?()\[\]{}"\'«»\-—\n]', '', answer)

        return GenResponse(text=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/create_agent')
def api_create_agent(req: AgentCreateRequest):
    key = create_agent(req.name, req.prompt)
    return {"api_key": key, "agent_name": req.name}

@app.get('/list_agents')
def api_list_agents():
    conn = sqlite3.connect('mun_agents.db')
    c = conn.cursor()
    c.execute("SELECT api_key, agent_name, created_at FROM agents ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"api_key": r[0], "agent_name": r[1], "created_at": r[2]} for r in rows]

@app.post('/analyze_pdf')
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(contents))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        risks = []
        if "неустойка" in text.lower():
            risks.append("⚠️ В договоре есть пункт о неустойке. Проверьте, не противоречит ли он ст. 333 ГК РФ.")
        if "расторжение" in text.lower():
            risks.append("⚠️ В договоре есть условия о расторжении. Проверьте, не ущемляют ли они ваши права.")
        if "штраф" in text.lower():
            risks.append("⚠️ В договоре есть штрафы. Убедитесь, что они соразмерны.")

        if not risks:
            risks.append("✅ Явных рисков по ключевым словам не найдено. Но рекомендую показать договор юристу.")

        return {"text": "\n".join(risks)}
    except Exception as e:
        return {"text": f"❌ Ошибка чтения PDF: {e}"}

@app.get('/health')
def health():
    return {'status': 'ok', 'articles': len(law_meta), 'agents': 'active'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)