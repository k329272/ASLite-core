# Free ASLite Backend Demo Path

This route avoids using your laptop as the server. It runs ASLite on a free hosted notebook runtime and exposes it through a temporary HTTPS tunnel for the Vercel frontend.

## 1. Start a free runtime
Use Google Colab with GPU if available. Clone this repo into the runtime, then run Kevin's model script as the source of truth:

```bash
git clone <ASLite-core repo url>
cd ASLite-core
bash ./download
pip install -r requirements.txt fastapi uvicorn[standard] pydantic pyngrok
```

## 2. Run the API server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 7860
```

## 3. Expose it with a tunnel
In another Colab cell:

```python
from pyngrok import ngrok
print(ngrok.connect(7860).public_url)
```

Copy the HTTPS URL. That becomes the frontend API URL.

## 4. Point Vercel frontend to it
In the frontend project:

```bash
npx vercel env add NEXT_PUBLIC_SIGNBRIDGE_API_URL production
npx vercel --prod
```

Paste the tunnel URL when Vercel asks for the env value.

## Notes
- This is good for a hackathon demo, but the URL changes whenever the notebook restarts.
- Hugging Face Docker Spaces would be cleaner, but free Docker/Gradio Space hosting required Pro in our CLI test.
- A real MVP should move this to a persistent backend host once budget/auth is available.
