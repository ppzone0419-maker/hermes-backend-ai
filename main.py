from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid, asyncio, json
from datetime import datetime
import google.generativeai as genai

app = FastAPI(title="HERMES PRO API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_STORE: dict = {}

AGENTS = {
    "pm": {
        "name": "專案經理",
        "emoji": "🎯",
        "system": "你是資深金融科技專案經理。你會把用戶需求拆解成具體任務清單，分析核心目標與技術挑戰，列出3-5個關鍵問題。用繁體中文，條列清晰。"
    },
    "strategist": {
        "name": "SMC策略師",
        "emoji": "📈",
        "system": "你是專精SMC、ICT、CRT的資深交易系統開發者。你熟悉Order Block、FVG、BOS/CHoCH、流動性獵取、Killzone等概念，能撰寫高品質Pine Script v5代碼。用繁體中文說明，英文代碼注釋。"
    },
    "critic": {
        "name": "風控審計官",
        "emoji": "🔍",
        "system": "你是資深量化風控專家，職責是找出問題。你會從過度擬合、回測偏差、停損缺陷、程式bug、極端行情處理等角度審查，列出至少3個問題並要求修正。用繁體中文，必須具體嚴格。"
    },
    "executor": {
        "name": "執行秘書",
        "emoji": "⚡",
        "system": "你是專業技術文件撰寫員。將討論結果整理成結構清晰的最終報告：包含策略概覽、完整Pine Script代碼（用```pinescript包裹）、風險提示、部署建議。用繁體中文，Markdown格式。"
    }
}

async def call_gemini(api_key: str, system: str, prompt: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=system
    )
    response = await asyncio.to_thread(
        model.generate_content, prompt,
        generation_config={"max_output_tokens": 2000, "temperature": 0.7}
    )
    return response.text

async def run_discussion(task_id: str, user_request: str, api_key: str):
    steps = [
        ("pm",         "pm",         lambda _: f"用戶需求：{user_request}\n\n請分析此需求，拆解任務清單，列出關鍵問題。"),
        ("strategist", "strategist", lambda prev: f"用戶需求：{user_request}\n\nPM分析結果：\n{prev}\n\n請根據以上設計完整交易策略並撰寫Pine Script v5代碼。"),
        ("critic",     "critic",     lambda prev: f"以下是策略師設計的策略：\n{prev}\n\n請嚴格審查，找出至少3個問題並提出修正建議。"),
        ("strategist", "strategist", lambda prev: f"審計官的問題：\n{prev}\n\n請逐一修正並提供優化後的完整Pine Script代碼。"),
        ("executor",   "executor",   lambda prev: f"用戶需求：{user_request}\n\n以上所有討論結果：\n{prev}\n\n請整理成最終報告。"),
    ]

    step_names = ["PM分析", "策略設計", "風控審計", "策略修正", "最終報告"]
    outputs = []
    last_output = ""

    try:
        for i, (agent_key, _, prompt_fn) in enumerate(steps):
            TASK_STORE[task_id]["current_step"] = i
            TASK_STORE[task_id]["status"] = "running"

            agent = AGENTS[agent_key]
            prompt = prompt_fn(last_output)

            output = await call_gemini(api_key, agent["system"], prompt)
            last_output = output
            outputs.append({
                "agent": step_names[i],
                "agent_key": agent_key,
                "name": agent["name"],
                "emoji": agent["emoji"],
                "content": output
            })
            TASK_STORE[task_id]["outputs"] = outputs

        TASK_STORE[task_id]["status"] = "done"
        TASK_STORE[task_id]["completed_at"] = datetime.now().isoformat()
        TASK_STORE[task_id]["result"] = {
            "final_report": last_output,
            "task_outputs": outputs
        }

    except Exception as e:
        TASK_STORE[task_id]["status"] = "error"
        TASK_STORE[task_id]["error"] = str(e)

class DiscussionRequest(BaseModel):
    user_request: str
    gemini_api_key: str

@app.get("/")
def root():
    return {"message": "HERMES PRO API", "version": "1.0.0", "status": "running"}

@app.post("/api/discussion/start")
async def start_discussion(req: DiscussionRequest, bg: BackgroundTasks):
    task_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    TASK_STORE[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "created_at": now,
        "current_step": -1,
        "outputs": [],
        "result": None,
        "error": None
    }
    bg.add_task(run_discussion, task_id, req.user_request, req.gemini_api_key)
    return {"task_id": task_id, "status": "pending", "created_at": now}

@app.get("/api/discussion/{task_id}")
def get_discussion(task_id: str):
    if task_id not in TASK_STORE:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")
    return TASK_STORE[task_id]

@app.get("/api/discussion/{task_id}/stream")
async def stream_discussion(task_id: str):
    async def generator():
        sent_steps = 0
        while True:
            if task_id not in TASK_STORE:
                break
            t = TASK_STORE[task_id]
            outputs = t.get("outputs", [])
            if len(outputs) > sent_steps:
                for o in outputs[sent_steps:]:
                    yield f"data: {json.dumps({'type':'step','data':o}, ensure_ascii=False)}\n\n"
                sent_steps = len(outputs)
            if t["status"] in ("done", "error"):
                yield f"data: {json.dumps({'type':'done','status':t['status'],'error':t.get('error')}, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(1)
    return StreamingResponse(generator(), media_type="text/event-stream")

@app.post("/webhook/line")
async def line_webhook(payload: dict):
    return {"status": "ok", "message": "LINE webhook 預留"}

@app.post("/webhook/discord")
async def discord_webhook(payload: dict):
    return {"status": "ok", "message": "Discord webhook 預留"}

@app.post("/webhook/telegram")
async def
