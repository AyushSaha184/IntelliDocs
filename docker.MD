# Render.com Deployment Guide for RAG Assistant

## 📋 Prerequisites

Before deploying to Render.com, you'll need:

1. **Render.com Account** (Free tier available)
   - Sign up at https://dashboard.render.com/

2. **GitHub/GitLab Repository**
   - Push your code to GitHub or GitLab
   - Render will deploy directly from your git repository

3. **OpenRouter API Key** (Required for LLM)
   - Get from https://openrouter.ai/keys
   - This is the ONLY API key you need!
   - No HuggingFace token required - embedding model downloads locally

---

## 🚀 Deployment Steps

### Method 1: Using render.yaml (Recommended - Infrastructure as Code)

1. **Add Files to Your Project Root**
   ```
   your-project/
   ├── Dockerfile
   ├── .dockerignore
   ├── render.yaml
   └── ... (rest of your project files)
   ```

2. **Push to GitHub**
   ```bash
   git add Dockerfile .dockerignore render.yaml
   git commit -m "Add Render deployment configuration"
   git push origin main
   ```

3. **Deploy on Render**
   - Go to https://dashboard.render.com/
   - Click **"New +"** → **"Blueprint"**
   - Connect your GitHub/GitLab repository
   - Select your repository
   - Render will automatically detect `render.yaml`
   - Click **"Apply"**

4. **Set Environment Variables**
   After deployment starts, go to your web service dashboard and set:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key

5. **Wait for Deployment**
   - Initial deployment takes 15-20 minutes (includes model download)
   - The embedding model (BAAI/bge-m3) downloads during build (~2-3 GB)
   - Watch the build logs in the Render dashboard
   - Once deployed, you'll get a URL like: `https://rag-assistant.onrender.com`

---

### Method 2: Manual Deployment (More Control)

#### Step 1: Create PostgreSQL Database

1. In Render Dashboard, click **"New +"** → **"PostgreSQL"**
2. Configure:
   - **Name**: `rag-assistant-db`
   - **Database**: `rag_db`
   - **User**: `rag_user`
   - **Region**: Choose closest to you
   - **Plan**: Free (or paid for production)
3. Click **"Create Database"**
4. **Save the connection details** (Internal Database URL)

#### Step 2: Create Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub/GitLab repository
3. Configure:
   - **Name**: `rag-assistant`
   - **Region**: Same as database
   - **Branch**: `main`
   - **Root Directory**: Leave blank
   - **Environment**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: Free (or paid for production)

#### Step 3: Configure Environment Variables

In the **Environment** tab, add these variables:

**Database (Copy from your PostgreSQL service):**
```
DATABASE_URL=<from PostgreSQL service>
POSTGRES_HOST=<from PostgreSQL service>
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=<from PostgreSQL service>
```

**Embeddings (Local Model - Downloaded During Build):**
```
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_MODEL_PATH=/app/models
EMBEDDING_NORMALIZE=true
```

**LLM (OpenRouter):**
```
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-3.5-sonnet
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4000
OPENROUTER_API_KEY=<your-openrouter-api-key>
OPENROUTER_SITE_URL=https://rag-assistant.onrender.com
OPENROUTER_SITE_NAME=RAG_Assistant
```

**Other Settings:**
```
USE_RERANKER=false
RETRIEVAL_MODE=hybrid
FUSION_STRATEGY=rrf
PYTHONUNBUFFERED=1
PYTHONPATH=/app
```

#### Step 4: Add Persistent Disk

1. In the **"Disks"** tab, click **"Add Disk"**
2. Configure:
   - **Name**: `rag-data`
   - **Mount Path**: `/app/data`
   - **Size**: 1 GB (free tier limit)
3. Click **"Save"**

#### Step 5: Deploy

1. Click **"Create Web Service"**
2. Wait for the build to complete (10-15 minutes)
3. Monitor logs in the **"Logs"** tab

---

## 🔧 Important Notes

### 1. **Local Embedding Model**
   - The embedding model (BAAI/bge-m3) is downloaded during Docker build
   - Model size: ~2.3 GB (stored in container, NOT persistent disk)
   - Cached in `/app/models` directory inside the container
   - No API calls needed for embeddings - runs entirely on Render
   - Model persists across container restarts

### 2. **Storage Architecture**
   - **Container Storage**: ~3GB for embedding model (part of container image)
   - **Persistent Disk**: 1-5GB for uploaded documents and vector store (mounted at /app/data)
   - Both are separate - model doesn't count against persistent disk quota

### 3. **OpenRouter for LLM**
   - Uses OpenRouter API for language model inference
   - You can change the model in environment variables (e.g., `gpt-4o`, `claude-3.5-sonnet`)
   - Check available models at https://openrouter.ai/models
   - Costs depend on model used (see OpenRouter pricing)

### 4. **Free Tier Considerations**
   - **RAM**: 512MB (tight but works - upgrade to 1-2GB recommended)
   - **Persistent Disk**: 1GB free (upgrade to 5GB for $0.50/month for more documents)
   - **Sleeps after 15 min of inactivity** (spins up in 30-60 seconds)
   - **750 hours/month** of runtime
   
   **Recommendation**: Start with free tier, upgrade RAM to 1GB ($7/month) for better performance.

### 5. **Build Time**
   - Initial build takes 15-20 minutes (downloads embedding model)
   - Rebuilds are faster if model is cached
   - Be patient during first deployment!

### 6. **File Uploads**
   - Use the persistent disk (/app/data) for uploaded documents
   - Start with 1GB free, upgrade as needed
   - Files persist across deployments and restarts

---

## 📁 File Upload Locations

Place these files in your **project root directory**:

```
Enterprise-ai-assistant/          ← Your project root
├── Dockerfile                     ← Upload HERE
├── .dockerignore                  ← Upload HERE  
├── render.yaml                    ← Upload HERE
├── requirements.txt               ← Already exists
├── backend/                       ← Already exists
│   └── main.py
├── frontend/                      ← Already exists
│   ├── package.json
│   └── src/
├── config/
├── data/
└── src/
```

---

## ✅ Verification Steps

After deployment:

1. **Check Health Endpoint**
   ```
   https://your-app.onrender.com/api/health
   ```
   Should return: `{"status":"ok","llm_loaded":true}`

2. **Access Frontend**
   ```
   https://your-app.onrender.com
   ```
   You should see the RAG Assistant UI

3. **Test Upload**
   - Upload a small document
   - Click "Process"
   - Ask a question

---

## 🎨 Customization Options

### Choosing a Different LLM Model

OpenRouter supports many models. Update the `LLM_MODEL` environment variable to any of:

**Most Popular:**
- `anthropic/claude-3.5-sonnet` (Best quality, higher cost)
- `anthropic/claude-3-haiku` (Fast and cheap)
- `openai/gpt-4o` (OpenAI's latest)
- `openai/gpt-3.5-turbo` (Cheap and fast)
- `google/gemini-pro` (Good balance)
- `meta-llama/llama-3-70b-instruct` (Open source)

**Check all available models:** https://openrouter.ai/models

### Using a Smaller Embedding Model

If you have disk space issues, use a smaller model in Dockerfile:

```python
# Replace BAAI/bge-m3 with a smaller model:
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/models'); \
    print('Model downloaded successfully')"
```

Then update render.yaml:
```yaml
- key: EMBEDDING_MODEL
  value: sentence-transformers/all-MiniLM-L6-v2
```

**Model Sizes:**
- `all-MiniLM-L6-v2`: ~90MB (fast, decent quality)
- `all-mpnet-base-v2`: ~420MB (better quality)
- `BAAI/bge-m3`: ~2.3GB (best quality, multilingual)

---

## 🐛 Troubleshooting

### Build Fails
- Check logs in Render dashboard
- Verify all files are committed to git
- Ensure `requirements.txt` has all dependencies (especially `sentence-transformers`)

### Model Download Issues
- If build times out, the model download may be slow
- Check build logs for "Model downloaded successfully" message
- If model download fails, Render may need more time or better network
- Consider using a smaller model like `all-MiniLM-L6-v2` for faster builds

### App Crashes on Startup
- Check if model is correctly loaded (look for errors in logs)
- Verify `EMBEDDING_MODEL_PATH=/app/models` is set
- Ensure enough RAM (upgrade from 512MB to 1GB or 2GB)

### "Out of Disk Space" Error
- This refers to **persistent disk** (/app/data), not container storage
- The embedding model (~2.3GB) is in container, doesn't use persistent disk
- Free tier persistent disk: 1GB - upgrade to 5GB for $0.50/month
- Use for uploaded documents and vector store

### OpenRouter API Errors
- Verify `OPENROUTER_API_KEY` is set correctly
- Check API key has credits at https://openrouter.ai/credits
- Verify model name is correct (check https://openrouter.ai/models)

### Database Connection Issues
- Ensure `DATABASE_URL` is set from PostgreSQL service
- Check database is in same region as web service

### Slow Performance
- Free tier has limited resources (512MB RAM)
- Upgrade to Starter plan (2GB RAM) for better performance
- Check if app is sleeping (cold start takes 30-60 seconds)

---

## 💰 Cost Optimization

**Free Tier (For Testing):**
- PostgreSQL: Free forever
- Web Service: Free (512MB RAM)
- Persistent Disk: 1GB free (can increase to 5GB for $0.50/month)
- Container Storage: Includes ~3GB for embedding model (part of service)
- OpenRouter API: Pay-per-use (varies by model)
- Total: $0-0.50/month + API costs

**Recommended Production Setup:**
- PostgreSQL: Starter ($7/month)
- Web Service: Starter ($7/month) - 2GB RAM, better performance
- Persistent Disk: 5GB ($0.50/month) - for documents and vector store
- OpenRouter API: Pay-per-use (varies by model)
- Total: ~$14.50/month + API costs

**Budget Option:**
- Use smaller embedding model (`all-MiniLM-L6-v2`)
- Use cheaper LLM (`gpt-3.5-turbo` or `claude-3-haiku`)
- Stay on free tier for personal projects

---

## 🔄 Updating Your App

**Method 1: Git Push (Automatic)**
```bash
git add .
git commit -m "Update application"
git push origin main
```
Render automatically rebuilds and redeploys.

**Method 2: Manual Deploy**
- Go to Render dashboard
- Click **"Manual Deploy"** → **"Deploy latest commit"**

---

## 📊 Monitoring

Access these in Render Dashboard:

1. **Metrics Tab**: CPU, Memory, Request rate
2. **Logs Tab**: Application logs in real-time
3. **Events Tab**: Deployment history
4. **Shell Tab**: Direct access to container (paid plans)

---

## 🎯 Next Steps

1. ✅ Upload the 3 files to your project root
2. ✅ Get your OpenRouter API key from https://openrouter.ai/keys
3. ✅ Push to GitHub/GitLab
4. ✅ Deploy on Render using Method 1 (render.yaml)
5. ✅ Set `OPENROUTER_API_KEY` environment variable
6. ✅ Wait for build (15-20 min for first build with model download)
7. ✅ Test your deployment

**Your app will be live at:** `https://rag-assistant.onrender.com` (or your custom URL)

### 🔑 Key Benefits of This Setup

✅ **No HuggingFace API required** - Embedding model runs locally
✅ **Fast embeddings** - No API latency, all computed on your server  
✅ **OpenRouter flexibility** - Switch between any LLM model easily  
✅ **Cost-effective** - Only pay for LLM usage, embeddings are free  
✅ **Private** - Your documents never leave your server for embedding

---

## 📞 Support

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- Check logs first when debugging issues

Good luck with your deployment! 🚀