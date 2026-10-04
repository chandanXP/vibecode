# main.py
import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables for local testing (Render uses the Environment tab)
load_dotenv()

app = FastAPI(title="Nemotron Ultra Coder")

# Initialize OpenAI async client pointing to NVIDIA's NIM API
client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    payload = await request.json()
    
    async def event_generator():
        try:
            stream = await client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b",
                messages=payload.get("messages", []),
                temperature=float(payload.get("temperature", 0.7)),
                top_p=float(payload.get("top_p", 0.95)),
                max_tokens=int(payload.get("max_tokens", 16384)),
                extra_body={"chat_template_kwargs": {"enable_thinking": payload.get("enable_thinking", True)}},
                stream=True
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # Intercept Nemotron's specific reasoning trace
                reasoning = getattr(delta, "reasoning_content", None)
                content = delta.content
                
                # Stream via Server-Sent Events (SSE)
                if reasoning:
                    yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning})}\n\n"
                if content:
                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")
