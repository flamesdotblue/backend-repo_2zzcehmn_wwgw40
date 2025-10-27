import os
import re
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    url: HttpUrl
    tone: str = Field(default="Professional", description="Tone of voice for the LinkedIn post")
    hashtag_count: int = Field(default=5, ge=0, le=10)
    include_emojis: bool = Field(default=True)


class SummarizeResponse(BaseModel):
    title: Optional[str] = None
    post: str
    summary: str
    hook: str
    hashtags: List[str]
    used_provider: str


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# ---------- Content Extraction & Summarization Utilities ----------
from bs4 import BeautifulSoup


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")


def extract_main_content(html: str) -> Dict[str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")

    # Try to get title
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title.get("content").strip()

    # Remove scripts/styles/nav/footer/aside
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    # Prefer article tag text, else main, else all paragraphs
    candidates = []
    for selector in ["article", "main"]:
        node = soup.find(selector)
        if node:
            text = "\n".join(p.get_text(" ", strip=True) for p in node.find_all(["p", "li"]))
            if len(text.split()) > 100:
                candidates.append(text)
    if not candidates:
        # Fallback: take all paragraphs
        text = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        candidates.append(text)

    content = max(candidates, key=lambda t: len(t)) if candidates else ""

    # Basic cleanup
    content = re.sub(r"\s+", " ", content).strip()
    return {"title": title, "content": content}


def make_hashtags(text: str, count: int = 5) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text.lower())
    stop = set(
        "the of and to in a is for on with that as at by an be are from it this will you your our about into not have has their more can new using over under after before out up down why how what when where which who whose into than these those many much very also just being been was were".split()
    )
    freq: Dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        key = w.replace("-", "")
        freq[key] = freq.get(key, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    tags = []
    for w, _ in ranked[: count * 2]:
        if w.isalpha():
            tags.append("#" + (w if len(w) < 20 else w[:20]))
        if len(tags) >= count:
            break
    # Ensure LinkedIn-friendly basics
    baseline = ["#Leadership", "#Strategy", "#AI", "#Innovation", "#Career"]
    for b in baseline:
        if len(tags) >= count:
            break
        if b.lower() not in [t.lower() for t in tags]:
            tags.append(b)
    return tags[:count]


def simple_summarize(text: str, max_sentences: int = 4) -> str:
    # Basic frequency-based summarization
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= max_sentences:
        return text[:1200]

    words = re.findall(r"\w+", text.lower())
    stop = set(
        "the of and to in a is for on with that as at by an be are from it this will you your our about into not have has their more can new using over under after before out up down why how what when where which who whose into than these those many much very also just being been was were".split()
    )
    freq: Dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1

    scores: List[float] = []
    for s in sentences:
        score = 0
        for w in re.findall(r"\w+", s.lower()):
            if w in freq:
                score += freq[w]
        scores.append(score / (len(s.split()) + 1))

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    best = [s for s, _ in ranked[: max_sentences]]
    # Preserve original order
    best_set = set(best)
    ordered = [s for s in sentences if s in best_set]
    summary = " ".join(ordered)
    return summary[:1200]


def craft_hook(title: Optional[str], summary: str, tone: str, include_emojis: bool) -> str:
    emoji = "🚀" if include_emojis else ""
    if title:
        return f"{emoji} {title.strip()} — here's what matters:".strip()
    # else derive a strong opening from summary
    first = summary.split(".")[0].strip()
    if len(first) > 0:
        return f"{emoji} {first}"
    return f"{emoji} Key takeaways you can use today:"


def craft_linkedin_post(title: Optional[str], summary: str, tone: str, hashtags: List[str], include_emojis: bool) -> str:
    hook = craft_hook(title, summary, tone, include_emojis)
    bullets = []
    for s in re.split(r"(?<=[.!?])\s+", summary)[:4]:
        s = s.strip()
        if not s:
            continue
        bullets.append(f"• {s}")
    flair = "\n" + ("✨ " if include_emojis else "") + {
        "Professional": "Actionable insights",
        "Insightful": "Why it matters",
        "Bold": "My take",
    }.get(tone, "Takeaways")
    body = "\n".join(bullets)
    tag_line = " ".join(hashtags)
    return f"{hook}\n\n{body}{flair}\n\n{tag_line}"


def try_openai_summarize(text: str, title: Optional[str], tone: str, hashtag_count: int, include_emojis: bool) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        # Use new OpenAI client if available
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        system = (
            "You are a helpful assistant that rewrites article content into a concise, high-signal LinkedIn post. "
            "Keep it scannable with short sentences. Avoid clickbait."
        )
        user_prompt = (
            f"Article title: {title or 'N/A'}\n\n"
            f"Content (trimmed):\n{text[:6000]}\n\n"
            f"Write a LinkedIn-ready post in a {tone} tone. Include a 3-5 bullet summary, a strong opening hook, and {hashtag_count} relevant hashtags. "
            f"Emojis: {'use sparingly' if include_emojis else 'do not use emojis'}."
        )
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        content = chat.choices[0].message.content
        # Heuristic split
        hashtags = [t for t in re.findall(r"#[A-Za-z0-9_]+", content)][:hashtag_count]
        # Try to extract a first line as hook
        first_line = content.splitlines()[0].strip() if content else ""
        if not hashtags:
            hashtags = make_hashtags(text, hashtag_count)
        post = content
        summary = simple_summarize(text, max_sentences=4)
        hook = first_line or craft_hook(title, summary, tone, include_emojis)
        return {
            "title": title,
            "post": post,
            "summary": summary,
            "hook": hook,
            "hashtags": hashtags,
            "used_provider": "openai",
        }
    except Exception:
        return None


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    html = fetch_html(str(req.url))
    extracted = extract_main_content(html)
    content = extracted.get("content") or ""
    title = extracted.get("title")
    if not content or len(content.split()) < 30:
        raise HTTPException(status_code=422, detail="Couldn't extract enough content from the provided URL.")

    # Try OpenAI first if key available
    ai_result = try_openai_summarize(
        text=content,
        title=title,
        tone=req.tone,
        hashtag_count=req.hashtag_count,
        include_emojis=req.include_emojis,
    )
    if ai_result:
        return ai_result

    # Fallback heuristic summarization
    summary = simple_summarize(content, max_sentences=4)
    hashtags = make_hashtags(content, req.hashtag_count)
    hook = craft_hook(title, summary, req.tone, req.include_emojis)
    post = craft_linkedin_post(title, summary, req.tone, hashtags, req.include_emojis)

    return {
        "title": title,
        "post": post,
        "summary": summary,
        "hook": hook,
        "hashtags": hashtags,
        "used_provider": "heuristic",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
