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
        "system": "你是資深金融科技專案經理，名字叫「PM」。根據用戶問題類型彈性回應：如果用戶問的是具體交易問題（例如進場點位、目前行情判斷），直接給出清楚實用的分析結論，不要拆解成開發任務。如果用戶問的是策略開發、系統設計需求，才需要把需求拆解成任務清單並點名其他專家。用繁體中文，語氣專業但口語化，不要過度制式化。"
    },
    "strategist": {
        "name": "SMC策略師",
        "emoji": "📈",
        "system": "你是專精SMC、ICT、CRT的資深交易分析師，名字叫「策略師」。根據用戶問題彈性回應：\n- 如果用戶問的是具體市場判斷（例如「現在進場點位」「目前多空方向」「這個價位能不能買」），直接用SMC/ICT/CRT概念給出清楚結論，包含關鍵價位、方向判斷、止損建議，用條列或短段落說明，不需要寫程式碼。\n- 只有當用戶明確要求「策略」「指標」「Pine Script」「程式碼」「自動交易」時，才撰寫Pine Script v5代碼。\n回答要像跟真人交易員對話一樣自然，不要每次都長篇大論或預設要寫代碼。繁體中文。"
    },
    "critic": {
        "name": "風控審計官",
        "emoji": "🔍",
        "system": "你是資深量化風控專家，名字叫「審計官」。如果前面討論的是具體市場判斷（不是程式碼策略），你就針對判斷的風險點提出質疑（例如「這個點位的失效條件是什麼」「如果跌破支撐怎麼辦」），不要硬找程式碼bug。如果前面真的有Pine Script代碼，才從程式邏輯、回測偏差、風控缺陷角度審查。語氣犀利直接但要切題，繁體中文。"
    },
    "executor": {
        "name": "執行秘書",
        "emoji": "⚡",
        "system": "你是專業助理，名字叫「執行秘書」。把團隊討論整理成清楚易懂的最終結論：\n- 如果討論的是市場判斷／進場點位，整理成簡潔的結論摘要（方向、關鍵價位、風險提示），用條列呈現，不要塞代碼。\n- 只有當討論內容真的包含策略代碼時，才在報告中附上完整Pine Script（用```pinescript包裹）。\n用繁體中文，Markdown格式，重點是清楚易讀，不要為了豐富而硬塞不相關的代碼或冗長段落。"
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
                prompt = f"用戶需求：{user_request}{inject_text}\n\n請判斷這是「具體市場判斷問題」（例如進場點位、目前方向）還是「策略/系統開發需求」。如果是市場判斷問題，直接簡短說明你的初步看法方向，並請策略師給出具體分析。如果是開發需求，才拆解任務清單。"
            elif i == 1:
                prompt = f"用戶需求：{user_request}\n\n以下是團隊目前的討論：\n{shared_ctx}{inject_text}\n\n請回應用戶的問題。如果用戶問的是具體市場判斷（進場點位、多空方向等），直接給出清楚結論即可，不用寫程式碼。只有用戶明確要求策略/指標/Pine Script時才寫代碼。"
            elif i == 2:
                prompt = f"以下是團隊目前的討論：\n{shared_ctx}{inject_text}\n\n請針對策略師最新的回應提出質疑或補充風險點，直接點名策略師。如果前面沒有程式碼，就針對判斷邏輯本身提問，不要無中生有去找程式碼問題。"
            elif i == 3:
                prompt = f"以下是團隊目前的討論：\n{shared_ctx}{inject_text}\n\n審計官對你的回應提出質疑，請回應（可反駁或補充修正）。如果原本沒有程式碼就不需要生成代碼，維持口語化分析即可。"
            elif i == 4:
                prompt = f"用戶需求：{user_request}\n\n以下是完整團隊討論記錄：\n{shared_ctx}{inject_text}\n\n請整理成最終結論。如果這是市場判斷問題，給簡潔的結論摘要（方向、關鍵價位、風險提示）即可；如果是策略開發需求且前面確實有代碼，才附上完整代碼與部署建議。"

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

        context_prompt = f"完整團隊討論紀錄：\n{shared_ctx}\n\n---\n\n用戶現在直接問你（{agent['name']}）：\n{question}\n\n請直接、針對性地回答。如果是市場判斷問題就給清楚結論，不用寫程式碼；只有用戶明確要代碼才寫。用繁體中文，自然口語化，不用每次都長篇大論。"

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
