from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid, asyncio, json, httpx
from datetime import datetime

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
        "system": "你是資深金融科技專案經理。把用戶需求拆解成具體任務清單，分析核心目標與技術挑戰，列出3-5個關鍵問題。用繁體中文，條列清晰。"
    },
    "strategist": {
        "name": "SMC策略師",
        "emoji": "📈",
        "system": "你是專精SMC、ICT、CRT的資深交易系統開發者。熟悉Order Block、FVG、BOS/CHoCH、流動性獵取、Killzone等概念，能撰寫高品質Pine Script v5代碼。繁體中文說明，英文代碼注釋。"
    },
    "critic": {
        "name": "風控審計官",
        "emoji": "🔍",
        "system": "你是資深量化風控專家，職責是找出問題。從過度擬合、回測偏差、停損缺陷、程式bug、極端行情處理等角度審查，列出至少3個問題並要求修正。繁體中文，必須具體嚴格。"
    },
    "executor": {
        "name": "執行秘書",
        "emoji": "⚡",
        "system": "你是專業技術文件撰寫員。將討論結果整理成最終報告：策略概覽、完整Pine Script代碼（用```pinescript包裹）、風險提示、部署建議。繁體中文，Markdown格式。"
    }
}

STEP_NAMES = ["PM分析", "策略設計", "風控審計", "策略修正", "最終報告"]
STEP_AGENTS = ["pm", "strategist", "critic", "strategist", "executor"]

async def call_gemini(api_key: str, system: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.7}
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

async def run_discussion(task_id: str, user_request: str, api_key: str):
    outputs = []
    last_output = ""

    prompts = [
        f"用戶需求：{user_request}\n\n請分析此需求，拆解任務清單，列出關鍵問題。",
        f"用戶需求：{user_request}\n\nPM分析：\n{last_output}\n\n請設計完整交易策略並撰寫Pine Script v5代碼。",
        f"策略師設計：\n{last_output}\n\n請嚴格審查，找出至少3個問題並提出修正建議。",
        f"審計問題：\n{last_output}\n\n請逐一修正並提供優化後的完整Pine Script代碼。",
        f"用戶需求：{user_request}\n\n所有討論結果已完成，請整理成最終報告。",
    ]

    try:
        for i, agent_key in enumerate(STEP_AGENTS):
            TASK_STORE[task_id]["current_step"] = i
            TASK_STORE[task_id]["status"] = "running"

            # rebuild prompt with latest last_output
            if i == 0:
                prompt = f"用戶需求：{user_request}\n\n請分析此需求，拆解任務清單，列出3-5個關鍵問題與開發方向。"
            elif i == 1:
                prompt = f"用戶需求：{user_request}\n\nPM分析結果：\n{last_output}\n\n請根據以上設計完整交易策略並撰寫Pine Script v5代碼，含完整注釋。"
            elif i == 2:
                prompt = f"以下是策略師的設計：\n{last_output}\n\n請嚴格審查，找出至少3個潛在問題（策略漏洞、代碼bug、風控缺失），並提出具體修正建議。"
            elif i == 3:
                prompt = f"審計官提出的問題：\n{last_output}\n\n請逐一回應並修正，提供優化後的完整Pine Script v5代碼，用注釋標記所有修改處。"
            elif i == 4:
                prompt = f"用戶需求：{user_request}\n\n以上討論已完成所有輪次。請整理成最終報告，包含：策略概覽、完整代碼、風險提示、TradingView部署建議。"

            agent = AGENTS[agent_key]
            output = await call_gemini(api_key, agent["system"], prompt)
            last_output = output

            step_data = {
                "agent": STEP_NAMES[i],
                "agent_key": agent_key,
                "name": agent["name"],
                "emoji": agent["emoji"],
                "content": output,
                "completed_at": datetime.now().isoformat()
            }
            outputs.append(step_data)
            TASK_STORE[task_id]["outputs"] = list(outputs)

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
        sent = 0
        while True:
            if task_id not in TASK_STORE:
                break
            t = TASK_STORE[task_id]
            outputs = t.get("outputs", [])
            for o in outputs[sent:]:
                yield f"data: {json.dumps({'type':'step','data':o}, ensure_ascii=False)}\n\n"
                sent += 1
            if t["status"] in ("done", "error"):
                yield f"data: {json.dumps({'type':'done','status':t['status'],'error':t.get('error')}, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(1)
    return StreamingResponse(generator(), media_type="text/event-stream")

@app.post("/webhook/line")
async def line_webhook(payload: dict):
    return {"status": "ok"}

@app.post("/webhook/discord")
async def discord_webhook(payload: dict):
    return {"status": "ok"}

@app.post("/webhook/telegram")
async def telegram_webhook(payload: dict):
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
