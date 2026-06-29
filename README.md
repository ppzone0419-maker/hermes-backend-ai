# HERMES PRO — 後端 API

## 本地運行
```bash
pip install -r requirements.txt
uvicorn main:app --reload
# API 文件：http://localhost:8000/docs
```

## 部署到 Render（免費）
1. 把這個資料夾推到 GitHub
2. 去 render.com → New Web Service → 連接 GitHub repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. 環境變數加入 `GEMINI_API_KEY`
6. 部署完成後得到網址（如 https://hermes-pro.onrender.com）

## API 端點
- POST /api/discussion/start — 啟動 AI 討論室
- GET  /api/discussion/{id}  — 查詢任務狀態
- GET  /api/discussion/{id}/stream — SSE 即時串流
- POST /webhook/line         — LINE Bot 預留
- POST /webhook/discord      — Discord 預留
- POST /webhook/telegram     — Telegram 預留
