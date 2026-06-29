"""
HERMES PRO — FastAPI + CrewAI 後端
AI 討論室 Multi-Agent 核心骨架
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid, asyncio, os, json
from datetime import datetime

# CrewAI
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

app = FastAPI(title="HERMES PRO API", version="1.0.0")

# ── CORS（允許前端 HTML 呼叫）─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 任務狀態暫存（生產環境可換成 Redis）──────────────────────────────────
TASK_STORE: dict[str, dict] = {}

# ── Gemini LLM 設定 ───────────────────────────────────────────────────────
def get_gemini_llm(api_key: str) -> LLM:
    return LLM(
        model="gemini/gemini-2.0-flash",
        api_key=api_key,
        temperature=0.7,
        max_tokens=2000,
    )

# ── Agent 定義 ────────────────────────────────────────────────────────────
def build_agents(llm: LLM) -> dict:
    pm = Agent(
        role="專案經理 (PM / Moderator)",
        goal="拆解用戶需求、協調各專家、統整最終決策報告",
        backstory=(
            "你是資深金融科技專案經理，擅長需求分析與多方協調。"
            "你會把用戶需求拆解成具體任務，指派給對應專家，"
            "並在最後統整出清晰的行動計畫與結論報告。"
        ),
        llm=llm,
        verbose=True,
    )

    strategist = Agent(
        role="SMC/ICT 策略師 & Pine Script 開發者",
        goal="設計高勝率交易策略並撰寫高品質 Pine Script 或 Python 程式碼",
        backstory=(
            "你是專精 SMC（Smart Money Concepts）、ICT（Inner Circle Trader）"
            "與 CRT（Candle Range Theory）的資深交易系統開發者。"
            "你熟悉 Order Block、FVG、BOS/CHoCH、流動性獵取、Killzone 等概念，"
            "能將這些框架轉化為可執行的 Pine Script v5 或 Python 策略代碼。"
        ),
        llm=llm,
        verbose=True,
    )

    critic = Agent(
        role="魔鬼代言人 / 風控審計官 (Critic & Auditor)",
        goal="找出策略漏洞、程式 bug、風控缺陷，強制要求修正",
        backstory=(
            "你是資深量化風控專家，你的職責是質疑一切、找出問題。"
            "你會從以下角度審查：過度擬合風險、回測偏差、停損設計缺陷、"
            "流動性風險、程式邏輯錯誤、極端行情處理。"
            "你不接受「差不多就好」，每個問題都必須被明確指出並要求修正。"
        ),
        llm=llm,
        verbose=True,
    )

    executor = Agent(
        role="執行秘書 / 輸出格式師 (Executor)",
        goal="將最終討論成果整理成結構清晰、可直接使用的報告與代碼",
        backstory=(
            "你是專業技術文件撰寫員，負責將討論結果格式化輸出。"
            "你會確保代碼有完整註解、報告有清晰段落、重點用 Markdown 標記。"
            "最終輸出必須讓用戶可以直接複製使用。"
        ),
        llm=llm,
        verbose=True,
    )

    return {"pm": pm, "strategist": strategist, "critic": critic, "executor": executor}

# ── Task 建構 ─────────────────────────────────────────────────────────────
def build_tasks(user_request: str, agents: dict) -> list[Task]:
    t1_analyze = Task(
        description=(
            f"用戶需求：{user_request}\n\n"
            "請以專案經理身分：\n"
            "1. 分析此需求的核心目標與技術挑戰\n"
            "2. 列出需要解決的 3-5 個關鍵問題\n"
            "3. 制定開發任務清單，指派給策略師\n"
            "用繁體中文，條列清晰。"
        ),
        agent=agents["pm"],
        expected_output="結構化的需求分析報告與任務清單",
    )

    t2_design = Task(
        description=(
            "根據 PM 的任務清單，作為 SMC/ICT 策略師：\n"
            "1. 設計完整的交易策略邏輯（入場條件、出場條件、止損止盈）\n"
            "2. 撰寫對應的 Pine Script v5 代碼（含完整注釋）\n"
            "3. 說明策略的理論依據（SMC/ICT/CRT 哪個框架）\n"
            "確保代碼可直接在 TradingView 運行。繁體中文說明 + 英文代碼注釋。"
        ),
        agent=agents["strategist"],
        expected_output="完整的策略說明 + Pine Script v5 代碼",
        context=[t1_analyze],
    )

    t3_audit = Task(
        description=(
            "作為魔鬼代言人審計官，對策略師的設計進行嚴格審查：\n"
            "1. 找出至少 3 個潛在問題或風險（策略邏輯、代碼 bug、風控）\n"
            "2. 對每個問題提出具體修正建議\n"
            "3. 評估此策略在不同市場情境下的失效條件\n"
            "必須具體、嚴格，不接受模糊回答。繁體中文。"
        ),
        agent=agents["critic"],
        expected_output="審計報告：問題清單 + 具體修正建議",
        context=[t2_design],
    )

    t4_revise = Task(
        description=(
            "根據審計官的反饋，作為策略師進行修正：\n"
            "1. 逐一回應每個問題並說明修正方式\n"
            "2. 提供修正後的完整 Pine Script v5 代碼\n"
            "3. 標記所有修改的地方（用注釋說明 // [修正] 原因）\n"
            "繁體中文說明 + 代碼。"
        ),
        agent=agents["strategist"],
        expected_output="修正後的策略說明 + 優化版 Pine Script 代碼",
        context=[t3_audit],
    )

    t5_output = Task(
        description=(
            "作為執行秘書，將整個討論過程整理成最終報告：\n"
            "1. 【需求摘要】用戶原始需求\n"
            "2. 【策略概覽】核心邏輯（3-5點）\n"
            "3. 【最終代碼】修正後的完整 Pine Script（代碼塊格式）\n"
            "4. 【風險提示】審計發現的主要風險\n"
            "5. 【部署建議】如何在 TradingView 設定此策略\n"
            "用 Markdown 格式輸出，代碼用 ```pinescript 包裹。繁體中文。"
        ),
        agent=agents["executor"],
        expected_output="完整的 Markdown 格式最終報告",
        context=[t4_revise],
    )

    return [t1_analyze, t2_design, t3_audit, t4_revise, t5_output]

# ── Pydantic Models ───────────────────────────────────────────────────────
class DiscussionRequest(BaseModel):
    user_request: str
    gemini_api_key: str
    session_id: Optional[str] = None

class TaskStatus(BaseModel):
    task_id: str
    status: str          # pending | running | done | error
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None

# ── Background task runner ────────────────────────────────────────────────
async def run_crew_task(task_id: str, request: DiscussionRequest):
    try:
        TASK_STORE[task_id]["status"] = "running"
        TASK_STORE[task_id]["logs"] = []

        llm = get_gemini_llm(request.gemini_api_key)
        agents = build_agents(llm)
        tasks  = build_tasks(request.user_request, agents)

        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        # 收集各 task 輸出
        task_outputs = []
        for i, task in enumerate(tasks):
            role_map = ["PM 分析", "策略設計", "風控審計", "策略修正", "最終報告"]
            task_outputs.append({
                "agent": role_map[i] if i < len(role_map) else f"Task {i+1}",
                "content": str(task.output) if hasattr(task, "output") else "",
            })

        TASK_STORE[task_id]["status"]       = "done"
        TASK_STORE[task_id]["completed_at"] = datetime.now().isoformat()
        TASK_STORE[task_id]["result"]       = {
            "final_report": str(result),
            "task_outputs": task_outputs,
        }

    except Exception as e:
        TASK_STORE[task_id]["status"] = "error"
        TASK_STORE[task_id]["error"]  = str(e)

# ── API Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "HERMES PRO API", "version": "1.0.0", "status": "running"}

@app.post("/api/discussion/start", response_model=TaskStatus)
async def start_discussion(req: DiscussionRequest, bg: BackgroundTasks):
    """啟動 AI 討論室多代理任務"""
    task_id = str(uuid.uuid4())
    now     = datetime.now().isoformat()

    TASK_STORE[task_id] = {
        "task_id":    task_id,
        "status":     "pending",
        "created_at": now,
        "completed_at": None,
        "result":     None,
        "error":      None,
        "request":    req.user_request,
    }

    bg.add_task(run_crew_task, task_id, req)

    return TaskStatus(task_id=task_id, status="pending", created_at=now)

@app.get("/api/discussion/{task_id}", response_model=TaskStatus)
def get_discussion(task_id: str):
    """查詢任務狀態與結果"""
    if task_id not in TASK_STORE:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")
    t = TASK_STORE[task_id]
    return TaskStatus(**{k: v for k, v in t.items() if k in TaskStatus.model_fields})

@app.get("/api/discussion/{task_id}/stream")
async def stream_discussion(task_id: str):
    """SSE 串流：即時推送討論進度到前端"""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        while True:
            if task_id not in TASK_STORE:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                break
            t = TASK_STORE[task_id]
            yield f"data: {json.dumps({'status': t['status'], 'result': t.get('result')})}\n\n"
            if t["status"] in ("done", "error"):
                break
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.delete("/api/discussion/{task_id}")
def delete_task(task_id: str):
    """清除任務記錄"""
    TASK_STORE.pop(task_id, None)
    return {"deleted": task_id}

# Webhook 預留接口（LINE / Discord / Telegram）
@app.post("/webhook/line")
async def line_webhook(payload: dict, bg: BackgroundTasks):
    """LINE Bot Webhook 預留接口"""
    # TODO: 驗證 LINE signature
    # events = payload.get("events", [])
    # for event in events:
    #     if event["type"] == "message":
    #         msg = event["message"]["text"]
    #         reply_token = event["replyToken"]
    #         # 啟動 discussion, 回傳結果給 LINE
    return {"status": "ok", "message": "LINE webhook 預留，待串接"}

@app.post("/webhook/discord")
async def discord_webhook(payload: dict):
    """Discord Webhook 預留接口"""
    return {"status": "ok", "message": "Discord webhook 預留，待串接"}

@app.post("/webhook/telegram")
async def telegram_webhook(payload: dict):
    """Telegram Webhook 預留接口"""
    return {"status": "ok", "message": "Telegram webhook 預留，待串接"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
