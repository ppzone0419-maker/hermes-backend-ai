from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uuid, asyncio, json, httpx
from datetime import datetime

app = FastAPI(title="HERMES PRO API", version="3.0.0")

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
        "system": "你是資深金融科技專案經理，名字叫「PM」。把用戶需求拆解成具體任務清單，分析核心目標與技術挑戰。你會主動點名其他專家發表意見，並且在團隊有分歧時負責協調。用繁體中文，條列清晰，語氣專業但不死板。"
    },
    "strategist": {
        "name": "SMC策略師",
        "emoji": "📈",
        "system": "你是專精SMC、ICT、CRT的資深交易系統開發者，名字叫「策略師」。熟悉Order Block、FVG、BOS/CHoCH、流動性獵取、Killzone等概念，能撰寫高品質Pine Script v5代碼。當審計官質疑你時，你會據理力爭或虛心修正，不會無條件投降。繁體中文說明，英文代碼注釋。"
    },
    "critic": {
        "name": "風控審計官",
        "emoji": "🔍",
        "system": "你是資深量化風控專家，名字叫「審計官」，職責是找出問題、質疑一切。從過度擬合、回測偏差、停損缺陷、程式bug、極端行情處理等角度審查，語氣犀利直接，不留情面，但論點要有依據。如果策略師回應你的質疑，你要評估對方的回應是否真的解決問題，可以繼續追問或表示認可。繁體中文。"
    },
    "executor": {
        "name": "執行秘書",
        "emoji": "⚡",
        "system": "你是專業技術文件撰寫員，名字叫「執行秘書」。將討論結果整理成最終報告：策略概覽、完整Pine Script代碼（用```pinescript包裹）、風險提示、部署建議。繁體中文，Markdown格式，簡潔精準。"
    }
}

GROQ_MODEL = "llama-3.3-70b-versatile"

async def call_groq(api_key: str, system: str, messages: List[dict]) -> str:
    """messages: list of {"role": "user"/"assistant", "content": str} — 支援多輪上下文"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    full_messages = [{"role": "system", "content": system}] + messages
    payload = {
        "model": GROQ_MODEL,
        "messages": full_messages,
        "max_tokens": 1200,
        "temperature": 0.75,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            raise Exception(f"Groq API 錯誤 {r.status_code}: {r.text[:300]}")
        data = r.json()
        return data["choices"][0]["message"]["content"]

def build_shared_history(task: dict) -> str:
    """把目前為止所有 outputs 組成一段給每個 agent 看的共享上下文"""
    parts = []
    for o in task.get("outputs", []):
        parts.append(f"【{o['name']}】說：\n{o['content']}")
    return "\n\n---\n\n".join(parts)

# ── 主要多輪辯論流程（5步驟，agent間真正互相回應） ─────────────────────────
async def run_discussion(task_id: str, user_request: str, api_key: str):
    outputs = []
    STEP_AGENTS = ["pm", "strategist", "critic", "strategist", "executor"]
    STEP_NAMES = ["PM分析", "策略設計", "風控審計", "策略修正", "最終報告"]

    try:
        for i, agent_key in enumerate(STEP_AGENTS):
            # 檢查是否被用戶中斷插話（pause）
            while TASK_STORE[task_id].get("paused"):
                await asyncio.sleep(1)
                if TASK_STORE[task_id].get("cancelled"):
                    return

            TASK_STORE[task_id]["current_step"] = i
            TASK_STORE[task_id]["status"] = "running"

            shared_ctx = build_shared_history(TASK_STORE[task_id])
            # 取得使用者中途插話的訊息（如果有）
            injections = TASK_STORE[task_id].get("injections", [])
            inject_text = ""
            if injections:
                inject_text = "\n\n【用戶中途補充意見】：\n" + "\n".join(injections)
                TASK_STORE[task_id]["injections"] = []  # 清空已讀取的

            if i == 0:
                prompt = f"用戶需求：{user_request}{inject_text}\n\n請分析此需求，拆解任務清單，列出3-5個關鍵問題與開發方向，並點名策略師接下來該做什麼。"
            elif i == 1:
                prompt = f"用戶需求：{user_request}\n\n以下是團隊目前的討論：\n{shared_ctx}{inject_text}\n\n請根據PM的分析設計完整交易策略並撰寫Pine Script v5代碼，含完整注釋。"
            elif i == 2:
                prompt = f"以下是團隊目前的討論：\n{shared_ctx}{inject_text}\n\n請針對策略師最新提出的設計，嚴格審查，找出至少3個潛在問題（策略漏洞、代碼bug、風控缺失），直接點名策略師回應。"
            elif i == 3:
                prompt = f"以下是團隊目前的討論：\n{shared_ctx}{inject_text}\n\n審計官對你的設計提出質疑，請逐一回應（可以反駁或修正），並提供優化後的完整Pine Script v5代碼，用注釋標記修改處。"
            elif i == 4:
                prompt = f"用戶需求：{user_request}\n\n以下是完整團隊討論記錄：\n{shared_ctx}{inject_text}\n\n請整理成最終報告，包含：策略概覽、完整代碼、風險提示、TradingView部署建議。"

            agent = AGENTS[agent_key]
            output = await call_groq(api_key, agent["system"], [{"role": "user", "content": prompt}])

            step_data = {
                "step_id": str(uuid.uuid4()),
                "agent": STEP_NAMES[i],
                "agent_key": agent_key,
                "name": agent["name"],
                "emoji": agent["emoji"],
                "content": output,
                "completed_at": datetime.now().isoformat()
            }
            outputs.append(step_data)
            TASK_STORE[task_id]["outputs"] = list(outputs)

            # 步驟間延遲，避免連續高 token 請求撞到 Groq 每分鐘速率限制
            await asyncio.sleep(3)

        TASK_STORE[task_id]["status"] = "done"
        TASK_STORE[task_id]["completed_at"] = datetime.now().isoformat()
        TASK_STORE[task_id]["result"] = {
            "final_report": outputs[-1]["content"] if outputs else "",
            "task_outputs": outputs
        }

    except Exception as e:
        TASK_STORE[task_id]["status"] = "error"
        TASK_STORE[task_id]["error"] = str(e)

# ── 單獨追問特定 Agent ──────────────────────────────────────────────────
async def run_followup(task_id: str, agent_key: str, question: str, api_key: str):
    try:
        followup_id = str(uuid.uuid4())
        followups = TASK_STORE[task_id].setdefault("followups", [])
        followups.append({"id": followup_id, "agent_key": agent_key, "question": question, "status": "running", "answer": None})

        agent = AGENTS[agent_key]
        shared_ctx = build_shared_history(TASK_STORE[task_id])

        # 找出這個 agent 之前在討論串裡是否已有發言過，組成對話歷史讓他「記得自己說過什麼」
        prior_messages = []
        for o in TASK_STORE[task_id].get("outputs", []):
            if o["agent_key"] == agent_key:
                prior_messages.append({"role": "assistant", "content": o["content"]})

        # 也包含之前的追問紀錄（同一個agent）
        for f in followups[:-1]:
            if f["agent_key"] == agent_key and f.get("answer"):
                prior_messages.append({"role": "user", "content": f["question"]})
                prior_messages.append({"role": "assistant", "content": f["answer"]})

        context_prompt = f"完整團隊討論紀錄：\n{shared_ctx}\n\n---\n\n用戶現在直接問你（{agent['name']}）：\n{question}\n\n請直接、針對性地回答，不用重複整個討論，用繁體中文。"

        messages = prior_messages + [{"role": "user", "content": context_prompt}]
        answer = await call_groq(api_key, agent["system"], messages)

        for f in followups:
            if f["id"] == followup_id:
                f["status"] = "done"
                f["answer"] = answer
                f["answered_at"] = datetime.now().isoformat()

    except Exception as e:
        for f in TASK_STORE[task_id].get("followups", []):
            if f["id"] == followup_id:
                f["status"] = "error"
                f["answer"] = f"⚠️ {str(e)}"

# ── Pydantic Models ────────────────────────────────────────────────────
class DiscussionRequest(BaseModel):
    user_request: str
    gemini_api_key: str  # 欄位保留相容前端，實際放 Groq Key

class InjectRequest(BaseModel):
    message: str

class FollowupRequest(BaseModel):
    agent_key: str  # pm / strategist / critic / executor
    question: str
    gemini_api_key: str

@app.get("/")
def root():
    return {"message": "HERMES PRO API", "version": "3.0.0", "llm": "Groq Llama 3.3 70B", "status": "running"}

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
        "followups": [],
        "injections": [],
        "paused": False,
        "cancelled": False,
        "result": None,
        "error": None,
        "api_key": req.gemini_api_key,
    }
    bg.add_task(run_discussion, task_id, req.user_request, req.gemini_api_key)
    return {"task_id": task_id, "status": "pending", "created_at": now}

@app.get("/api/discussion/{task_id}")
def get_discussion(task_id: str):
    if task_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="Task not found")
    t = dict(TASK_STORE[task_id])
    t.pop("api_key", None)  # 不外洩 key
    return t

@app.post("/api/discussion/{task_id}/inject")
def inject_message(task_id: str, req: InjectRequest):
    """中途插話：下一個 agent 發言時會看到這段補充意見"""
    if task_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="Task not found")
    TASK_STORE[task_id].setdefault("injections", []).append(req.message)
    return {"status": "injected", "message": req.message}

@app.post("/api/discussion/{task_id}/followup")
async def followup_agent(task_id: str, req: FollowupRequest, bg: BackgroundTasks):
    """單獨追問某個 Agent，不影響主討論流程"""
    if task_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="Task not found")
    if req.agent_key not in AGENTS:
        raise HTTPException(status_code=400, detail="無效的 agent_key")
    bg.add_task(run_followup, task_id, req.agent_key, req.question, req.gemini_api_key)
    return {"status": "asked", "agent_key": req.agent_key}

@app.post("/api/discussion/{task_id}/pause")
def pause_discussion(task_id: str):
    if task_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="Task not found")
    TASK_STORE[task_id]["paused"] = True
    return {"status": "paused"}

@app.post("/api/discussion/{task_id}/resume")
def resume_discussion(task_id: str):
    if task_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="Task not found")
    TASK_STORE[task_id]["paused"] = False
    return {"status": "resumed"}

@app.get("/api/discussion/{task_id}/stream")
async def stream_discussion(task_id: str):
    async def generator():
        sent_steps = 0
        sent_followups = 0
        while True:
            if task_id not in TASK_STORE:
                break
            t = TASK_STORE[task_id]
            outputs = t.get("outputs", [])
            for o in outputs[sent_steps:]:
                yield f"data: {json.dumps({'type':'step','data':o}, ensure_ascii=False)}\n\n"
                sent_steps += 1
            followups = t.get("followups", [])
            for f in followups[sent_followups:]:
                if f.get("status") in ("done", "error"):
                    yield f"data: {json.dumps({'type':'followup','data':f}, ensure_ascii=False)}\n\n"
                    sent_followups += 1
            if t["status"] in ("done", "error") and sent_followups >= len(followups):
                yield f"data: {json.dumps({'type':'done','status':t['status'],'error':t.get('error')}, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(1)
    return StreamingResponse(generator(), media_type="text/event-stream")

@app.delete("/api/discussion/{task_id}")
def delete_task(task_id: str):
    TASK_STORE.pop(task_id, None)
    return {"deleted": task_id}

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
