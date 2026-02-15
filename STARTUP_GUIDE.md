# RAG Assistant - Startup Guide

Complete guide to start the entire RAG Assistant application with frontend, backend, and all required services.

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.9+** installed
- **Node.js 16+** and npm installed
- **PostgreSQL** installed and running
- **LM Studio** installed (for local embeddings)
- All dependencies installed

---

## 🚀 Quick Start (Step-by-Step)

### Step 1: Start PostgreSQL Database

**Check if PostgreSQL is running:**
```powershell
Get-Service -Name "*postgresql*"
```

**If not running, start it:**
```powershell
Start-Service postgresql-x64-18
```

**Verify database exists:**
```powershell
psql -U postgres -d rag_db -c "\dt"
```

If database doesn't exist, create it:
```powershell
psql -U postgres -c "CREATE DATABASE rag_db;"
```

---

### Step 2: Configure Environment Variables

**Edit `.env` file in the project root:**

```bash
# Location: C:\VS Code\Projects\RAG_Assistant\Enterprise-ai-assistant\.env
```

**Required settings:**

```env
# Database (REQUIRED)
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_db
POSTGRES_PASSWORD=YOUR_PASSWORD

# LM Studio for Local Embeddings (REQUIRED)
EMBEDDING_PROVIDER=lm-studio
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=YOUR_KEY_HERE

# LLM Provider (choose one)
# Option 1: OpenRouter (has free tier)
LLM_PROVIDER=openrouter
LLM_MODEL=deepseek/deepseek-r1-0528:free
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY

# Option 2: Google Gemini (recommended - free & reliable)
# LLM_PROVIDER=gemini
# LLM_MODEL=gemini-2.0-flash-exp
# GEMINI_API_KEY=YOUR_GEMINI_KEY

# Get free Gemini key: https://aistudio.google.com/app/apikey
```

---

### Step 3: Start LM Studio (for Embeddings)

**1. Open LM Studio application**

**2. Load Embedding Model:**
   - Go to the **Models** tab
   - Search for: `text-embedding-bge-m3`
   - Download if not already downloaded
   - Click **Load Model** in the Server tab

**3. Start Local Server:**
   - Go to **Local Server** tab
   - Click **Start Server**
   - Verify it's running on `http://127.0.0.1:1234`

**4. Test the server:**
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:1234/v1/models" -UseBasicParsing | Select-Object -ExpandProperty Content
```

---

### Step 4: Install Python Dependencies

**Navigate to project directory:**
```powershell
cd "C:\VS Code\Projects\RAG_Assistant\Enterprise-ai-assistant"
```

**Install dependencies (if not already done):**
```powershell
pip install -r requirements.txt
```

---

### Step 5: Start Backend Server (FastAPI)

**Option A - Simple start:**
```powershell
$env:PYTHONPATH="C:\VS Code\Projects\RAG_Assistant\Enterprise-ai-assistant"
python -m uvicorn backend.main:app --port 8000 --reload
```

**Option B - Background process:**
```powershell
$env:PYTHONPATH="C:\VS Code\Projects\RAG_Assistant\Enterprise-ai-assistant"
Start-Process python -ArgumentList "-m", "uvicorn", "backend.main:app", "--port", "8000", "--reload" -WindowStyle Hidden
```

**Verify backend is running:**
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing | Select-Object -ExpandProperty Content
```

Expected output: `{"status":"ok","llm_loaded":true}`

---

### Step 6: Install Frontend Dependencies

**Navigate to frontend directory:**
```powershell
cd frontend
```

**Install npm packages (first time only):**
```powershell
npm install
```

---

### Step 7: Start Frontend Development Server

**Start Vite dev server:**
```powershell
npm run dev
```

**Expected output:**
```
VITE v6.4.1  ready in 684 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

---

## ✅ Verification Checklist

After all steps, verify everything is running:

| Service | Status Check | Expected Result |
|---------|--------------|-----------------|
| **PostgreSQL** | `Get-Service postgresql-x64-18` | Status: Running |
| **LM Studio** | Open http://127.0.0.1:1234 in browser | Server page loads |
| **Backend API** | Open http://localhost:8000/docs | FastAPI docs page |
| **Frontend** | Open http://localhost:5173 | RAG Assistant UI loads |

---

## 🎯 Using the Application

1. **Open your browser:** http://localhost:5173
2. **Upload a document:**
   - Click the paperclip icon
   - Select a file (PDF, DOCX, TXT, etc.)
   - Click "Upload All"
3. **Process the document:**
   - Click "Process" button
   - Wait for processing to complete
4. **Ask questions:**
   - Type your question in the input field
   - Press Enter or click Send
   - Get AI-powered answers from your documents!

---

## 🛠️ Troubleshooting

### Backend won't start
**Error:** `ModuleNotFoundError: No module named 'backend'`

**Fix:**
```powershell
$env:PYTHONPATH="C:\VS Code\Projects\RAG_Assistant\Enterprise-ai-assistant"
```

---

### Database connection error
**Error:** `DATABASE_URL environment variable is required`

**Fix:** Check `.env` file has correct DATABASE_URL setting

---

### Frontend shows "connection refused"
**Issue:** Backend not running

**Fix:** Start backend on port 8000 (see Step 5)

---

### LM Studio embeddings fail
**Error:** `Connection refused to localhost:1234`

**Fix:**
1. Open LM Studio
2. Load embedding model
3. Start Local Server
4. Verify with: `curl http://127.0.0.1:1234/v1/models`

---

### Rate limit errors (OpenRouter)
**Error:** `429 Too Many Requests`

**Fix:** Switch to Gemini in `.env`:
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash-exp
GEMINI_API_KEY=your_key_here
```

Get free key: https://aistudio.google.com/app/apikey

---

## 🔄 Stopping the Services

**Stop Frontend:**
```
Press Ctrl+C in the terminal running npm
```

**Stop Backend:**
```powershell
# Find the process
Get-Process python | Where-Object {$_.Path -like "*python*"}

# Kill by port
$proc = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($proc) { Stop-Process -Id $proc.OwningProcess -Force }
```

**Stop LM Studio:**
- Click "Stop Server" in LM Studio app

**Stop PostgreSQL (optional):**
```powershell
Stop-Service postgresql-x64-18
```

---

## 📁 Directory Structure

```
Enterprise-ai-assistant/
├── backend/              # FastAPI backend
│   ├── main.py          # Entry point
│   ├── api/             # API routes
│   └── services/        # Business logic
├── frontend/            # React frontend
│   ├── src/             # Source code
│   └── package.json     # npm config
├── data/                # Data storage
│   ├── documents/       # Uploaded files (central)
│   ├── sessions/        # Session-isolated storage
│   └── vector_store/    # FAISS indexes
├── .env                 # Environment variables
└── requirements.txt     # Python dependencies
```

---

## 📞 Need Help?

- Check logs in `logs/` directory
- Backend logs: Terminal where uvicorn is running
- Frontend logs: Browser console (F12)
- Database logs: PostgreSQL logs

---

## 🎉 You're Ready!

Your RAG Assistant is now running:
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

Start uploading documents and asking questions! 🚀
