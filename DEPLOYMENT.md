# Deployment Guide — Smart Insurance Advisor V2.0

This guide covers multiple deployment options from local development to cloud-hosted production.

## Table of Contents

1. [Local Development](#1-local-development)
2. [Docker Deployment](#2-docker-deployment)
3. [Render Deployment (Free Tier)](#3-render-deployment-free-tier)
4. [Railway Deployment](#4-railway-deployment)
5. [Environment Variables](#5-environment-variables)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Local Development

### Prerequisites
- Python 3.11 or 3.12
- pip
- (Optional) An Anthropic API key for Claude AI reports

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/Dutchy-O-o/DataMiningInsurance.git
cd DataMiningInsurance

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set Claude API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > DataSet/.env

# Run the app
cd DataSet
python app.py
```

The app will open automatically at `http://localhost:5000`.

### Running Tests

```bash
pytest tests/ -v
```

---

## 2. Docker Deployment

### Prerequisites
- Docker Desktop installed and running

### Single-Command Deployment

```bash
# Build the image
docker build -t smart-insurance-advisor .

# Run the container
docker run -p 5000:5000 smart-insurance-advisor
```

### Using Docker Compose (recommended)

```bash
# Start with compose (creates .env if needed)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

With the API key:

```bash
# Create .env at the project root
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
docker-compose up -d
```

---

## 3. Render Deployment (Free Tier)

Render offers free hosting for web services with automatic Git deployment.

### Steps

1. **Push your code to GitHub** (make sure `requirements.txt`, `Dockerfile`, and `.dockerignore` are committed).

2. **Sign up at [render.com](https://render.com)** and connect your GitHub account.

3. **Create a new Web Service:**
   - Click "New +" → "Web Service"
   - Select your DataMiningInsurance repo
   - Configure:
     - **Name:** smart-insurance-advisor
     - **Region:** Frankfurt (or closest to you)
     - **Branch:** main
     - **Runtime:** Docker
     - **Instance Type:** Free
   - Render auto-detects the `Dockerfile`.

4. **Add environment variables** (optional, for Claude AI):
   - Click "Environment" → "Add Environment Variable"
   - Key: `ANTHROPIC_API_KEY`
   - Value: your API key

5. **Click "Create Web Service".** Render will build and deploy automatically.

6. **Your live URL** will be something like `https://smart-insurance-advisor.onrender.com`

### Notes on Free Tier
- Free services spin down after 15 minutes of inactivity
- First request after idle takes ~30 seconds to wake up
- 750 hours/month free compute time

---

## 4. Railway Deployment

Railway offers $5/month free credits for new accounts.

### Steps

1. **Go to [railway.app](https://railway.app)** and sign up with GitHub.

2. **Start a new project:**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select DataMiningInsurance

3. **Railway auto-detects Dockerfile** and starts building.

4. **Add environment variables** (Settings → Variables):
   - `ANTHROPIC_API_KEY` (optional)
   - `PORT` (Railway sets this automatically, but check)

5. **Generate public URL:**
   - Settings → Networking → "Generate Domain"
   - You'll get `your-app.up.railway.app`

### Important: Bind to Railway's PORT

Railway injects a `$PORT` environment variable. Update `app.py` if needed:

```python
# At the bottom of app.py, replace:
app.run(host="0.0.0.0", port=5000)

# With:
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
```

---

## 5. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | No | Enables Claude AI personalized reports. Without it, predictions still work. |
| `PORT` | No | Port to bind to (default: 5000). Cloud platforms often inject this. |
| `FLASK_ENV` | No | Set to `production` on cloud hosts. |

### Getting a Claude API Key

1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Go to API Keys → Create Key
3. Copy the key (starts with `sk-ant-`)
4. Add $5+ credits to your account for usage

Claude Haiku pricing: ~$0.0003 per 1K tokens. A single insurance report uses ~1,000 tokens (~$0.0003).

---

## 6. Troubleshooting

### "No module named 'shap'" during Docker build
Ensure `libgomp1` is installed in the Dockerfile (already included).

### "Connection refused" after deployment
Check that the app binds to `0.0.0.0` not `localhost`. Cloud platforms need external binding.

### Predictions return 500 error
Verify all `.joblib` model files are committed to the repo. They should be in `DataSet/saved_models/`.

### AI reports not appearing
This is expected behavior when no `ANTHROPIC_API_KEY` is set. The app gracefully degrades — predictions and SHAP still work.

### Container OOM on Render free tier
Render free tier has 512MB RAM. If the app crashes:
- Remove matplotlib/seaborn imports from app.py (only needed for plot generation)
- Consider upgrading to the $7/month Starter tier

### Tests fail in CI but pass locally
Ensure `DataSet/saved_models/*.joblib` files are committed. Git LFS may be needed for large model files.

---

## Post-Deployment Checklist

- [ ] `/` returns the HTML UI
- [ ] `/api/stats` returns dataset JSON
- [ ] `/predict` accepts POST with valid payload
- [ ] `/batch_predict` accepts CSV uploads
- [ ] `/similar` returns 5 similar patients
- [ ] Claude AI reports generate (if API key set)
- [ ] Mobile view is responsive
- [ ] HTTPS is enabled (automatic on Render/Railway)
