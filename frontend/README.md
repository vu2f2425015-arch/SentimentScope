# SentimentScope Frontend (Vercel)

This directory contains the standalone web client for **SentimentScope**, designed for 1-click deployment on [Vercel](https://vercel.com).

## Key Features
- **Zero Build Time**: Lightweight Single-Page Application (HTML5, Vanilla JS, Tailwind CSS, Chart.js).
- **DotField Interactive Engine**: 60fps canvas particle mesh with interactive tab behaviors and dynamic light/dark themes.
- **Dynamic Cloud Backend Switcher**: Interactive modal to connect to your Vercel Serverless backend API or test locally on `http://127.0.0.1:8000`.
- **Automatic Fallback & Persistence**: Backend URL is saved in browser `localStorage`.

---

## Deploy to Vercel (2 Options)

### Option A: Via Vercel Web Dashboard (Recommended)
1. Push your repository to GitHub.
2. Log into [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Import your `SentimentScope` repository.
4. In the **Project Settings**:
   - Set **Root Directory** to `frontend`.
   - Leave **Framework Preset** as **Other**.
   - (Optional) Under **Environment Variables**, you can add `VITE_API_URL` or let the UI auto-connect.
5. Click **Deploy**. Your frontend is live instantly worldwide with global edge CDN caching!

### Option B: Via Vercel CLI
```bash
# Navigate to frontend folder
cd frontend

# Deploy preview
vercel

# Deploy to production
vercel --prod
```

---

## Local Development
To run the frontend locally:
```bash
# Option 1: Using npx serve
npx serve -l 3000 .

# Option 2: Python simple HTTP server
python -m http.server 3000
```
Then open `http://localhost:3000` in your browser.
