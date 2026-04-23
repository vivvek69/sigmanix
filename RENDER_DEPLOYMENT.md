# Render Deployment Guide - Sigmanix Tech Chatbot

## Quick Deployment Steps

### 1. **Prepare Your Repository**

Push all code to GitHub (if not already done):
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. **Create Render Account & Connect GitHub**

1. Go to [Render.com](https://render.com)
2. Sign up with GitHub
3. Grant Render access to your GitHub repositories
4. Click "New +" → "Web Service"
5. Select your chatbot repository

### 3. **Configure on Render**

**Basic Settings:**
- **Name:** `sigmanix-chatbot` (or your preferred name)
- **Environment:** `Python 3.11`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn chatbot_production:app`
- **Instance Type:** `Free` or `Starter` (recommended)

**Environment Variables** (Critical!):
```
GROQ_API_KEY=your_groq_api_key_from_console.groq.com
SECRET_KEY=generate_a_random_secure_string_here
FLASK_ENV=production
CORS_ORIGINS=https://your-render-url.onrender.com,https://yourdomain.com
DATABASE_PATH=chatbot_database.db
```

### 4. **Get Your Groq API Key**

1. Visit [console.groq.com](https://console.groq.com)
2. Create/Log into account
3. Go to API Keys section
4. Generate new API key
5. Copy and paste into Render's `GROQ_API_KEY` environment variable

### 5. **Deploy**

1. Click "Create Web Service"
2. Render will automatically deploy when you push to GitHub
3. Wait for deployment to complete (5-10 minutes)
4. Your chatbot will be live at: `https://sigmanix-chatbot.onrender.com`

## Important Notes

### Database Management
- Render's file system is ephemeral (temporary)
- Your SQLite database will be reset if the service stops
- **Recommendation:** Use PostgreSQL for production data persistence
  ```yaml
  # Add to render.yaml for PostgreSQL
  databases:
    - name: chatbot_db
      databaseName: sigmanix_chatbot
      user: chatbot_user
  ```

### Cold Start Issue
- Free tier experiences ~30-second cold starts when idle
- Upgrade to `Starter` or `Standard` tier for better performance

### Production Checklist
- ✅ GROQ_API_KEY set
- ✅ SECRET_KEY is strong and random
- ✅ CORS_ORIGINS includes your Render URL
- ✅ FLASK_ENV=production
- ✅ requirements.txt includes all dependencies

## Troubleshooting

**"ModuleNotFoundError" error:**
- Check that all imports are in `requirements.txt`
- Verify Python version matches (3.11 recommended)

**CORS errors:**
- Add your Render URL to `CORS_ORIGINS` environment variable
- Include both `https://` version and any custom domain

**Groq API errors:**
- Verify GROQ_API_KEY is correct
- Check Groq console hasn't hit rate limits
- Test API key locally first

**Database connection errors:**
- Ensure database file path is writable
- Consider switching to PostgreSQL for production

## Auto-Redeploy with GitHub

Any push to your main branch will automatically trigger a new deployment on Render!

## Support

- Render Docs: https://render.com/docs
- Groq API Docs: https://console.groq.com/docs
- Report issues in your repository
# 🌟 RENDER DEPLOYMENT GUIDE - Sigmanix Tech Chatbot

## Status: ✅ READY FOR RENDER DEPLOYMENT

**Render URL:** https://render.com  
**Repository:** https://github.com/vivvek69/sigmanix-tech-chatbot  
**Estimated Time:** 5 minutes

---

## 🎯 STEP-BY-STEP DEPLOYMENT

### **Step 1: Sign Up on Render**

1. Go to: https://render.com
2. Click: **"Sign up"**
3. Choose: **"Sign up with GitHub"** (easiest)
4. Authorize Render to access your GitHub account
5. Done! ✅

---

### **Step 2: Create New Web Service**

1. Click: **"New +"** (top right)
2. Select: **"Web Service"**
3. Choose: **"Public Git repository"**
4. Paste: **`https://github.com/vivvek69/sigmanix-tech-chatbot.git`**
5. Click: **"Continue"**

---

### **Step 3: Configure Web Service**

Fill in the following:

**Basic Settings:**
- **Name:** `sigmanix-chatbot`
- **Environment:** Python 3
- **Region:** Choose closest to you (e.g., Oregon, Frankfurt)
- **Branch:** `main`

**Build & Deployment:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn chatbot_production:app`

**Instance Type:**
- Select: **"Free"** (recommended to start)
- Or upgrade to: **"Starter"** ($7/month) for better performance

---

### **Step 4: Add Environment Variables**

1. Click: **"Add Environment Variable"**

2. **First Variable:**
   - **Key:** `GROQ_API_KEY`
   - **Value:** [Get from step below]
   - Click: **"Add"**

3. **Optional Variables:**
   - `FLASK_ENV` = `production`
   - `FLASK_SECRET_KEY` = [generate random string]

---

### **Step 5: Get Your Groq API Key**

Before clicking "Deploy", you need your API key:

1. Open new tab: https://console.groq.com
2. Sign up (free)
3. In Dashboard:
   - Click: **"API Keys"**
   - Click: **"Create New API Key"**
   - Copy the key
4. Return to Render
5. Paste key in `GROQ_API_KEY` field

---

### **Step 6: Deploy!**

1. Click: **"Create Web Service"**
2. Render will:
   - Build your application
   - Install dependencies
   - Start your server
   - Assign public URL

3. **Wait:** 3-5 minutes for first deployment

---

### **Step 7: Access Your Chatbot**

After deployment:

1. Render dashboard shows: **"Service is live"** ✅
2. Your URL appears in top section: `https://sigmanix-chatbot.onrender.com`
3. Click the URL to open your chatbot
4. **Share with anyone!** 🌐

---

## ✅ WHAT YOU GET

- ✅ **Live URL:** `https://sigmanix-chatbot.onrender.com`
- ✅ **Always On:** 24/7 hosting
- ✅ **Auto-Deploy:** Updates on GitHub push
- ✅ **Custom Domain:** Add your domain anytime
- ✅ **SSL/HTTPS:** Automatic
- ✅ **Logs:** Real-time error tracking
- ✅ **Monitoring:** Built-in dashboard
- ✅ **Scaling:** Auto-scales with traffic

---

## 🔄 AUTO-DEPLOYMENT FROM GITHUB

After initial setup, every push to `main` branch auto-deploys:

```bash
git add .
git commit -m "Update chatbot features"
git push origin main
# Render automatically redeploys!
```

---

## 📊 RENDER DASHBOARD

After deployment, use Render dashboard 

1. **View Logs:** Real-time server output
2. **Monitor:** CPU, memory, bandwidth usage
3. **Restart:** Manual restart if needed
4. **Settings:** Update environment variables
5. **Connect Domain:** Add custom domain

---

## 🐛 TROUBLESHOOTING

### **Issue: "Build failed"**
```
Solution: 
1. Check logs for error messages
2. Verify requirements.txt is in root directory
3. Ensure data.txt exists
4. Check GROQ_API_KEY is set
```

### **Issue: "Service not starting"**
```
Solution:
1. Check Start Command: gunicorn chatbot_production:app
2. Verify GROQ_API_KEY environment variable is set
3. Check logs for Python errors
```

### **Issue: "502 Bad Gateway"**
```
Solution:
1. Service might still be starting (wait 1-2 min)
2. Check if chatbot_production.py has errors
3. Ensure port 5000 binding is correct
```

### **Issue: "CORS error"**
```
Solution:
1. Check chatbot_production.py CORS configuration
2. Add your frontend domain to REACT_DOMAINS
3. Update on Render dashboard
```

---

## 📱 MOBILE ACCESS

After deployment:

1. Get your URL: `https://sigmanix-chatbot.onrender.com`
2. Open on phone/tablet
3. Works perfectly on mobile! 📱
4. Share link with team

---

## 💰 PRICING

**Free Tier:**
- Cost: $0/month
- Always on: Yes (on Free instances, no auto-sleep)
- Compute: Shared
- Suitable for: Testing, demos, low-traffic

**Starter Tier:**
- Cost: $7/month
- Always on: Yes (guaranteed)
- Compute: Dedicated
- Suitable for: Production, reliability

---

## 🔐 SECURITY

Render provides:
- ✅ HTTPS/SSL automatic
- ✅ DDoS protection
- ✅ Environment variable encryption
- ✅ GitHub OAuth security
- ✅ Auto-backups

---

## 📈 MONITORING & LOGS

**View Logs:**
1. Render Dashboard → Your Service
2. Click: **"Logs"** tab
3. See real-time server output

**Monitor Metrics:**
1. Click: **"Metrics"** tab
2. View: CPU, Memory, Bandwidth usage
3. Track performance over time

---

## 🚀 NEXT STEPS

After deployment:

1. ✅ Test your chatbot: Ask questions
2. ✅ Share the URL with team
3. ✅ Set up custom domain (optional)
4. ✅ Monitor logs for errors
5. ✅ Add teammates as collaborators

---

## 🔗 USEFUL LINKS

- **Render Dashboard:** https://dashboard.render.com
- **Groq Console:** https://console.groq.com
- **Your Repository:** https://github.com/vivvek69/sigmanix-tech-chatbot
- **Render Docs:** https://render.com/docs

---

## 📞 SUPPORT

**For Render Issues:**
- Email: support@render.com
- Docs: https://render.com/docs

**For Chatbot Issues:**
- Check logs in Render dashboard
- Review chatbot_production.py for errors
- Test locally first: `python chatbot_production.py`

---

**🎉 YOUR CHATBOT IS NOW LIVE ON RENDER!**

**Your URL:** `https://sigmanix-chatbot.onrender.com`

Share it with your team and users! 🌐