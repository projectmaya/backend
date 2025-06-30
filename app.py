import os, json, uvicorn, hashlib, asyncio, aiofiles
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq, Groq
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(APP_DIR, 'history.json')
LOGIN_FILE = os.path.join(APP_DIR, 'login.json')
CONFIG_FILE = os.path.join(APP_DIR, 'info.json')
history_lock, login_lock, info_lock = asyncio.Lock(), asyncio.Lock(), asyncio.Lock()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

async def async_read_json(filepath):
    if not os.path.exists(filepath): return [] if 'login' in filepath else {}
    try:
        async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
            return json.loads(await f.read())
    except: return [] if 'login' in filepath else {}

async def async_write_json(filepath, data, lock):
    async with lock:
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2))

def load_app_config():
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: exit(1)
CONFIG = load_app_config()
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()
def load_users():
    users = CONFIG.get('users', {})
    return {u.lower(): {'password_hash': hash_password(d['password']), 'displayName': d['displayName'], 'ai_settings': d.get('ai_settings', {})} for u, d in users.items()}
USERS = load_users()

class ChatRequest(BaseModel): username: str; conversation_id: str; message: str; image_data: Optional[str] = None
class LoginRequest(BaseModel): username: str; password: str
class LogoutRequest(BaseModel): username: str
class AISettings(BaseModel):
    text_system_prompt: Optional[str] = "You are MAYA, a helpful AI assistant."
    text_temperature: Optional[float] = 0.7
    text_top_p: Optional[float] = 0.9
    image_system_prompt: Optional[str] = "Analyze this image in detail and then answer the user's question based on your analysis."
    image_temperature: Optional[float] = 0.5
    image_top_p: Optional[float] = 0.95
class AddUserRequest(BaseModel): username: str; password: str; displayName: str; ai_settings: AISettings

class AIController:
    def __init__(self, config):
        self.client = AsyncGroq(api_key=config['api_config']['groq_api_key'])
        self.text_model = config['api_config']['text_model']
        self.image_model = config['api_config']['image_model']
        self.sync_client = Groq(api_key=config['api_config']['groq_api_key'])
    async def analyze_image(self, prompt, image_data, image_settings):
        if image_data and not image_data.startswith('data:'):
            image_data = f"data:image/jpeg;base64,{image_data}"
        system_prompt = image_settings.get('image_system_prompt')
        messages = [{"role": "user", "content": [{"type": "text", "text": (system_prompt) + f"\n\nUser Question: {prompt}"}, {"type": "image_url", "image_url": {"url": image_data}}]}]
        try:
            loop = asyncio.get_running_loop()
            chat_completion = await loop.run_in_executor(None, lambda: self.sync_client.chat.completions.create(messages=messages, model=self.image_model, temperature=image_settings.get('image_temperature'), top_p=image_settings.get('image_top_p')))
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"ImageAI Error: {e}")
            return "Error: Could not analyze the image."
    async def stream_text_response(self, history, settings):
        try:
            stream = await self.client.chat.completions.create(messages=history, model=self.text_model, temperature=settings.get('text_temperature'), top_p=settings.get('text_top_p'), stream=True)
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content: yield content
        except Exception as e:
            print(f"TextAI Error: {e}")
            yield "An error occurred with the text model."
ai_controller = AIController(CONFIG)

def _reconstruct_ai_history(simplified_history):
    ai_history = []
    for turn in simplified_history:
        user_content = f"Based on the following image analysis, answer the user's query.\n\n--- IMAGE ANALYSIS ---\n{turn['image_ai']}\n\n--- USER QUERY ---\n{turn['user']}" if turn.get('image_ai') else turn['user']
        ai_history.append({"role": "user", "content": user_content})
        ai_history.append({"role": "assistant", "content": turn['text_ai']})
    return ai_history
async def log_login_event(username, action):
    log = await async_read_json(LOGIN_FILE)
    log.append({"username": username, "time": datetime.now().isoformat(), "action": action})
    await async_write_json(LOGIN_FILE, log, login_lock)

@app.post("/login")
async def login(request: LoginRequest):
    user = USERS.get(request.username.lower())
    if not user or hash_password(request.password) != user["password_hash"]:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    await log_login_event(request.username, "login")
    return {"status": "success", "displayName": user["displayName"], "username": request.username}

@app.post("/logout")
async def logout(request: LogoutRequest):
    await log_login_event(request.username, "logout")
    return {"status": "success"}

@app.get('/history/{username}')
async def get_user_history(username):
    db = await async_read_json(HISTORY_FILE)
    user_conversations = db.get(username, {})
    summary = [{"id": conv_id, "title": data[0]['user'][:50] if data else "Empty Chat"} for conv_id, data in user_conversations.items()]
    return sorted(summary, key=lambda x: x['id'], reverse=True)

@app.get('/history/{username}/{conversation_id}')
async def get_conversation(username, conversation_id):
    db = await async_read_json(HISTORY_FILE)
    conversation = db.get(username, {}).get(conversation_id)
    if conversation is None: raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@app.delete('/history/{username}/{conversation_id}')
async def delete_conversation(username, conversation_id):
    db = await async_read_json(HISTORY_FILE)
    if db.get(username, {}).get(conversation_id) is not None:
        del db[username][conversation_id]
        await async_write_json(HISTORY_FILE, db, history_lock)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Conversation not found")

@app.post('/chat')
async def chat(request: ChatRequest):
    user_data = USERS.get(request.username.lower())
    if not user_data: raise HTTPException(status_code=403, detail="Invalid user")
    user_ai_settings = user_data.get('ai_settings', {})
    async def stream_and_save():
        db = await async_read_json(HISTORY_FILE)
        if request.username not in db: db[request.username] = {}
        if request.conversation_id not in db[request.username]: db[request.username][request.conversation_id] = []
        conversation_log = db[request.username][request.conversation_id]
        image_ai_output = await ai_controller.analyze_image(request.message, request.image_data, user_ai_settings) if request.image_data else ""
        ai_history = _reconstruct_ai_history(conversation_log)
        if user_ai_settings.get('text_system_prompt'):
            ai_history.insert(0, {"role": "system", "content": user_ai_settings['text_system_prompt']})
        final_user_prompt = f"Based on the following image analysis, answer the user's query.\n\n--- IMAGE ANALYSIS ---\n{image_ai_output}\n\n--- USER QUERY ---\n{request.message}" if image_ai_output else request.message
        ai_history.append({"role": "user", "content": final_user_prompt})
        full_text_ai_reply = ""
        stream = ai_controller.stream_text_response(ai_history, user_ai_settings)
        async for chunk in stream:
            full_text_ai_reply += chunk
            yield chunk
        new_turn = {"user": request.message, "image_ai": image_ai_output, "text_ai": full_text_ai_reply}
        conversation_log.append(new_turn)
        await async_write_json(HISTORY_FILE, db, history_lock)
    return StreamingResponse(stream_and_save(), media_type="text/plain")

@app.get("/api/console/data/{filename}")
async def get_console_data(filename):
    if filename == "info": return JSONResponse(content=await async_read_json(CONFIG_FILE))
    if filename == "history": return JSONResponse(content=await async_read_json(HISTORY_FILE))
    if filename == "logs": return JSONResponse(content=await async_read_json(LOGIN_FILE))
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/console/save/{filename}")
async def save_console_data(filename, request: Request):
    try: data = await request.json()
    except: raise HTTPException(status_code=400, detail="Invalid JSON format")
    if filename == "info":
        if not isinstance(data, dict): raise HTTPException(status_code=400, detail="Invalid data structure for info.json")
        await async_write_json(CONFIG_FILE, data, info_lock)
        global CONFIG, USERS
        CONFIG = data
        USERS = load_users()
    elif filename == "history":
        if not isinstance(data, dict): raise HTTPException(status_code=400, detail="Invalid data structure for history.json")
        await async_write_json(HISTORY_FILE, data, history_lock)
    elif filename == "logs":
        if not isinstance(data, list): raise HTTPException(status_code=400, detail="Invalid data structure for login.json")
        await async_write_json(LOGIN_FILE, data, login_lock)
    else: raise HTTPException(status_code=404, detail="File not found")
    return {"status": "success", "detail": f"{filename}.json saved successfully."}

@app.post("/api/console/add_user")
async def add_new_user_console(request: AddUserRequest):
    global CONFIG, USERS
    async with info_lock:
        current_config_data = await async_read_json(CONFIG_FILE)
        if not isinstance(current_config_data, dict): current_config_data = {}
        if 'users' not in current_config_data: current_config_data['users'] = {}
        if request.username.lower() in (u.lower() for u in current_config_data.get('users', {})):
            raise HTTPException(status_code=409, detail=f"Username '{request.username}' already exists.")
        current_config_data['users'][request.username] = {"password": request.password, "displayName": request.displayName, "ai_settings": request.ai_settings.dict()}
        try:
            async with aiofiles.open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(current_config_data, indent=2))
        except Exception as e:
            print(f"FATAL: Could not write updated config to {CONFIG_FILE} during add user: {e}")
            raise HTTPException(status_code=500, detail="Failed to save new user configuration to file.")
        CONFIG = current_config_data
        USERS = load_users()
    return {"status": "success", "message": f"User '{request.username}' added successfully."}

if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)