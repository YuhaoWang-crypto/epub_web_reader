import os
import asyncio
import base64
import hashlib
import io
import json
import posixpath
import re
import threading
import wave
import zipfile
import xml.etree.ElementTree as ET

import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup

# ============================================================
# Optional dependencies
# ============================================================
try:
    import edge_tts  # pip install edge-tts
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

try:
    from openai import OpenAI  # pip install openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from google import genai  # pip install google-genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# ============================================================
# App config
# ============================================================
st.set_page_config(page_title="EPUB Web Reader", layout="wide")


# ============================================================
# Helpers
# ============================================================
def normalize_zip_path(path: str) -> str:
    path = (path or "").replace("\\", "/")
    path = re.sub(r"^\./", "", path)
    return posixpath.normpath(path)


def resolve_href(base_dir: str, href: str):
    """
    Resolve a relative href (possibly with #fragment / ?query) to a zip internal path.
    Returns: (zip_path or None, fragment)
    """
    href = (href or "").strip()
    if not href:
        return None, ""
    if re.match(r"^[a-zA-Z]+://", href):
        return None, ""  # external link
    href_no_frag = href.split("#", 1)[0].split("?", 1)[0]
    fragment = href.split("#", 1)[1] if "#" in href else ""
    target = normalize_zip_path(posixpath.join(base_dir, href_no_frag))
    return target, fragment


def decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "utf-16"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin-1", errors="ignore")


def guess_mime(path: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "svg": "image/svg+xml",
        "webp": "image/webp",
        "css": "text/css",
    }.get(ext, "application/octet-stream")


def soup_html(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def first_child_text(parent, tag_suffix: str):
    if parent is None:
        return None
    for el in parent.iter():
        if isinstance(el.tag, str) and el.tag.endswith(tag_suffix):
            if el.text and el.text.strip():
                return el.text.strip()
    return None


def run_coro(coro):
    """
    Run async coroutine safely in Streamlit (handles already-running event loops).
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        out = {}

        def runner():
            out["value"] = asyncio.run(coro)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join()
        return out.get("value")


def get_secret(*names: str) -> str:
    """
    Read secrets from Streamlit Secrets first, then environment variables.
    """
    for n in names:
        if hasattr(st, "secrets") and n in st.secrets and str(st.secrets.get(n, "")).strip():
            return str(st.secrets.get(n)).strip()
        if str(os.environ.get(n, "")).strip():
            return str(os.environ.get(n)).strip()
    return ""


# ============================================================
# TOC parsing
# ============================================================
def parse_nav_toc(z: zipfile.ZipFile, nav_path: str):
    html = decode_bytes(z.read(nav_path))
    soup = soup_html(html)

    nav = soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav", attrs={"role": "doc-toc"})
    if not nav:
        for cand in soup.find_all("nav"):
            t = cand.get("epub:type") or cand.get("type") or ""
            if "toc" in str(t).lower():
                nav = cand
                break

    toc_dir = posixpath.dirname(nav_path)
    entries = []
    if nav:
        for a in nav.select("a[href]"):
            title = a.get_text(" ", strip=True) or "Untitled"
            href = a.get("href", "")
            path, frag = resolve_href(toc_dir, href)
            if path:
                entries.append({"title": title, "href": href, "path": path, "fragment": frag})
    return entries


def parse_ncx_toc(z: zipfile.ZipFile, ncx_path: str):
    root = ET.fromstring(z.read(ncx_path))
    toc_dir = posixpath.dirname(ncx_path)
    entries = []

    def walk(node):
        for child in list(node):
            if isinstance(child.tag, str) and child.tag.endswith("navPoint"):
                title = None
                src = None
                for el in child.iter():
                    if title is None and isinstance(el.tag, str) and el.tag.endswith("text"):
                        if el.text and el.text.strip():
                            title = el.text.strip()
                    if src is None and isinstance(el.tag, str) and el.tag.endswith("content"):
                        src = el.attrib.get("src")
                if src:
                    path, frag = resolve_href(toc_dir, src)
                    if path:
                        entries.append({"title": title or "Untitled", "href": src, "path": path, "fragment": frag})
                walk(child)

    navmap = next((e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("navMap")), root)
    walk(navmap)
    return entries


# ============================================================
# EPUB parsing (DRM-free)
# ============================================================
@st.cache_data(show_spinner=False)
def parse_epub(epub_bytes: bytes):
    z = zipfile.ZipFile(io.BytesIO(epub_bytes))
    file_list = set(z.namelist())

    if "META-INF/container.xml" not in file_list:
        raise ValueError("EPUB 缺少 META-INF/container.xml，文件可能损坏或不是标准 EPUB。")

    container_root = ET.fromstring(z.read("META-INF/container.xml"))
    opf_path = None
    for el in container_root.iter():
        if isinstance(el.tag, str) and el.tag.endswith("rootfile"):
            opf_path = el.attrib.get("full-path")
            break
    if not opf_path:
        raise ValueError("无法在 container.xml 中找到 rootfile (OPF)。")

    opf_path = normalize_zip_path(opf_path)
    if opf_path not in file_list:
        raise ValueError(f"OPF 文件不存在：{opf_path}")

    opf_dir = posixpath.dirname(opf_path)
    opf_root = ET.fromstring(z.read(opf_path))

    metadata_el = next((e for e in opf_root.iter() if isinstance(e.tag, str) and e.tag.endswith("metadata")), None)
    title = first_child_text(metadata_el, "title") or "Untitled"
    creator = first_child_text(metadata_el, "creator") or ""
    language = first_child_text(metadata_el, "language") or ""

    manifest_el = next((e for e in opf_root.iter() if isinstance(e.tag, str) and e.tag.endswith("manifest")), None)
    manifest = {}
    if manifest_el is not None:
        for item in list(manifest_el):
            if not (isinstance(item.tag, str) and item.tag.endswith("item")):
                continue
            iid = item.attrib.get("id")
            href = item.attrib.get("href")
            media_type = item.attrib.get("media-type", "")
            props = item.attrib.get("properties", "")
            if not iid or not href:
                continue
            path = normalize_zip_path(posixpath.join(opf_dir, href))
            manifest[iid] = {"id": iid, "href": href, "path": path, "media_type": media_type, "properties": props}

    mime_by_path = {m["path"]: (m.get("media_type") or "") for m in manifest.values()}

    spine_el = next((e for e in opf_root.iter() if isinstance(e.tag, str) and e.tag.endswith("spine")), None)
    spine_idrefs = []
    ncx_id = spine_el.attrib.get("toc") if spine_el is not None else None
    if spine_el is not None:
        for itemref in list(spine_el):
            if isinstance(itemref.tag, str) and itemref.tag.endswith("itemref"):
                idref = itemref.attrib.get("idref")
                if idref:
                    spine_idrefs.append(idref)

    def is_doc(media_type: str) -> bool:
        mt = (media_type or "").lower()
        return ("xhtml" in mt) or ("html" in mt)

    spine_paths = []
    for idref in spine_idrefs:
        m = manifest.get(idref)
        if m and is_doc(m.get("media_type", "")):
            p = m["path"]
            if p in file_list:
                spine_paths.append(p)

    if not spine_paths:
        raise ValueError("未能从 spine 中解析到可阅读章节（可能是非常规 EPUB 或受保护文件）。")

    nav_path = None
    for m in manifest.values():
        if "nav" in (m.get("properties") or "").split():
            nav_path = m["path"]
            break

    ncx_path = None
    if ncx_id and ncx_id in manifest:
        ncx_path = manifest[ncx_id]["path"]
    if not ncx_path:
        for m in manifest.values():
            mt = (m.get("media_type") or "").lower()
            if "dtbncx" in mt or (m.get("href", "").lower().endswith(".ncx")):
                ncx_path = m["path"]
                break

    cover_path = None
    for m in manifest.values():
        props = (m.get("properties") or "").split()
        if "cover-image" in props and (m.get("media_type") or "").startswith("image/"):
            cover_path = m["path"]
            break
    if cover_path and cover_path not in file_list:
        cover_path = None

    toc_entries = []
    if nav_path and nav_path in file_list:
        toc_entries = parse_nav_toc(z, nav_path)
    elif ncx_path and ncx_path in file_list:
        toc_entries = parse_ncx_toc(z, ncx_path)

    path_to_index = {p: i for i, p in enumerate(spine_paths)}
    chapter_titles = [None] * len(spine_paths)
    for e in toc_entries:
        p = e.get("path")
        if p in path_to_index:
            idx = path_to_index[p]
            if not chapter_titles[idx]:
                chapter_titles[idx] = e.get("title") or None
    for i in range(len(chapter_titles)):
        if not chapter_titles[i]:
            chapter_titles[i] = f"第 {i+1} 章"

    return {
        "title": title,
        "creator": creator,
        "language": language,
        "spine_paths": spine_paths,
        "chapter_titles": chapter_titles,
        "mime_by_path": mime_by_path,
        "cover_path": cover_path,
        "file_list": file_list,
    }


# ============================================================
# Chapter extraction & rendering
# ============================================================
def embed_images_in_body(body, chapter_path: str, z: zipfile.ZipFile, mime_by_path: dict, file_list: set, max_images: int = 300):
    base_dir = posixpath.dirname(chapter_path)
    count = 0
    for img in body.find_all("img"):
        if count >= max_images:
            break
        src = img.get("src")
        if not src:
            continue
        target, _frag = resolve_href(base_dir, src)
        if not target or target not in file_list:
            continue
        raw = z.read(target)
        mime = mime_by_path.get(target) or guess_mime(target)
        b64 = base64.b64encode(raw).decode("ascii")
        img["src"] = f"data:{mime};base64,{b64}"
        count += 1


def extract_chapter_text_and_html(epub_bytes: bytes, book: dict, chapter_idx: int, embed_images: bool):
    z = zipfile.ZipFile(io.BytesIO(epub_bytes))
    chapter_path = book["spine_paths"][chapter_idx]
    raw = z.read(chapter_path)
    html = decode_bytes(raw)

    soup = soup_html(html)
    body = soup.body or soup
    for s in body.find_all("script"):
        s.decompose()

    text = body.get_text("\n", strip=True)

    if embed_images:
        embed_images_in_body(body, chapter_path, z, book["mime_by_path"], book["file_list"])
    body_html = "".join(str(x) for x in body.contents)
    return text, body_html


def extract_chapter_blocks(epub_bytes: bytes, book: dict, chapter_idx: int):
    z = zipfile.ZipFile(io.BytesIO(epub_bytes))
    chapter_path = book["spine_paths"][chapter_idx]
    html = decode_bytes(z.read(chapter_path))
    soup = soup_html(html)
    body = soup.body or soup
    for s in body.find_all("script"):
        s.decompose()

    blocks = []
    for el in body.find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]):
        txt = el.get_text(" ", strip=True)
        if txt:
            blocks.append({"text": txt, "html": str(el)})

    if not blocks:
        txt = body.get_text("\n", strip=True)
        parts = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        blocks = [{"text": p, "html": f"<p>{p}</p>"} for p in parts]
    return blocks


def paginate_blocks(blocks, per_page: int):
    return [blocks[i:i + per_page] for i in range(0, len(blocks), per_page)]


# ============================================================
# HTML wrapper (Reader + optional browser TTS)
# ============================================================
def wrap_reader_html(body_html: str, font_size: int, line_height: float, max_width: int, theme: str,
                     enable_web_speech: bool, speech_lang: str):
    if theme == "Dark":
        bg = "#0f1115"
        fg = "#e6e6e6"
        subtle = "#b6b6b6"
        panel = "rgba(255,255,255,.06)"
        border = "rgba(255,255,255,.14)"
    else:
        bg = "#ffffff"
        fg = "#111111"
        subtle = "#555555"
        panel = "rgba(0,0,0,.04)"
        border = "rgba(0,0,0,.12)"

    tts_toolbar = ""
    tts_script = ""

    if enable_web_speech:
        tts_toolbar = f"""
        <div class="ttsbar">
          <div class="row">
            <button id="ttsStartTop">从本章开头朗读</button>
            <button id="ttsStartSel">从选区/光标处朗读</button>
            <button id="ttsPause">暂停</button>
            <button id="ttsResume">继续</button>
            <button id="ttsStop">停止</button>
          </div>
          <div class="row">
            <label>语速</label>
            <input id="ttsRate" type="range" min="0.6" max="1.4" step="0.05" value="1.0"/>
            <span id="ttsRateVal">1.00</span>
            <label style="margin-left:14px;">声音</label>
            <select id="ttsVoice"></select>
            <span class="tip">提示：直接点击任意段落即可从该段开始读</span>
          </div>
        </div>
        """

        tts_script = f"""
        <script>
        (function() {{
          const LANG = {json.dumps(speech_lang)};

          function $(id) {{ return document.getElementById(id); }}

          const rate = $("ttsRate");
          const rateVal = $("ttsRateVal");
          const voiceSel = $("ttsVoice");

          rate.addEventListener("input", () => {{
            rateVal.textContent = Number(rate.value).toFixed(2);
          }});

          function collectSpeakables() {{
            const nodes = Array.from(document.querySelectorAll(
              ".reader p, .reader li, .reader blockquote, .reader h1, .reader h2, .reader h3, .reader h4, .reader h5, .reader h6"
            ));
            const speakables = nodes
              .map(n => ({{ el: n, text: (n.innerText || "").trim() }}))
              .filter(x => x.text.length > 0);

            speakables.forEach((x, i) => {{
              x.el.dataset.ttsIndex = String(i);
              x.el.classList.add("tts-clickable");
              x.el.addEventListener("click", (evt) => {{
                if (evt.target && evt.target.closest && evt.target.closest("a")) return;
                startFrom(i);
              }});
            }});
            return speakables;
          }}

          let speakables = collectSpeakables();
          let speaking = false;

          function clearHighlight() {{
            document.querySelectorAll(".tts-speaking").forEach(n => n.classList.remove("tts-speaking"));
          }}
          function highlight(i) {{
            clearHighlight();
            const el = speakables[i]?.el;
            if (!el) return;
            el.classList.add("tts-speaking");
            el.scrollIntoView({{ block: "center", behavior: "smooth" }});
          }}

          function loadVoices() {{
            const voices = speechSynthesis.getVoices() || [];
            const filtered = voices.filter(v => (v.lang || "").toLowerCase().startsWith(LANG.toLowerCase()));
            const list = (filtered.length ? filtered : voices);

            voiceSel.innerHTML = "";
            list.forEach(v => {{
              const opt = document.createElement("option");
              opt.value = v.name;
              opt.textContent = `${{v.name}} (${{v.lang}})`;
              voiceSel.appendChild(opt);
            }});
          }}
          speechSynthesis.onvoiceschanged = loadVoices;
          loadVoices();

          function getSelectedVoice() {{
            const name = voiceSel.value;
            const voices = speechSynthesis.getVoices() || [];
            return voices.find(v => v.name === name) || null;
          }}

          function stopAll() {{
            speechSynthesis.cancel();
            speaking = false;
            clearHighlight();
          }}

          function speakOne(i) {{
            if (i >= speakables.length) {{
              stopAll();
              return;
            }}
            speaking = true;
            highlight(i);

            const u = new SpeechSynthesisUtterance(speakables[i].text);
            u.lang = LANG;
            u.rate = Number(rate.value);

            const v = getSelectedVoice();
            if (v) u.voice = v;

            u.onend = () => {{
              if (!speaking) return;
              speakOne(i + 1);
            }};
            u.onerror = () => {{
              if (!speaking) return;
              speakOne(i + 1);
            }};
            speechSynthesis.speak(u);
          }}

          function startFrom(i) {{
            stopAll();
            speakOne(i);
          }}

          function startFromSelection() {{
            const sel = window.getSelection();
            if (!sel || sel.rangeCount === 0) return;
            let node = sel.anchorNode;
            if (!node) return;
            if (node.nodeType === 3) node = node.parentElement;
            while (node && !node.dataset.ttsIndex) node = node.parentElement;
            if (!node) return;
            const idx = parseInt(node.dataset.ttsIndex, 10);
            if (!isNaN(idx)) startFrom(idx);
          }}

          $("ttsStartTop").addEventListener("click", () => startFrom(0));
          $("ttsStartSel").addEventListener("click", () => startFromSelection());
          $("ttsPause").addEventListener("click", () => speechSynthesis.pause());
          $("ttsResume").addEventListener("click", () => speechSynthesis.resume());
          $("ttsStop").addEventListener("click", () => stopAll());

          window.addEventListener("load", () => {{
            speakables = collectSpeakables();
          }});
        }})();
        </script>
        """

    top_padding = "72px" if enable_web_speech else "24px"
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body {{
    background: {bg};
    color: {fg};
    margin: 0;
    padding: 0;
    font-size: {font_size}px;
    line-height: {line_height};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
  }}
  .reader {{
    max-width: {max_width}px;
    margin: 0 auto;
    padding: {top_padding} 18px 64px 18px;
    word-wrap: break-word;
    overflow-wrap: anywhere;
  }}
  img {{ max-width: 100%; height: auto; }}
  a {{ color: inherit; text-decoration: underline; }}
  hr {{ border: none; border-top: 1px solid rgba(127,127,127,.35); }}
  blockquote {{
    margin: 16px 0;
    padding: 10px 14px;
    border-left: 3px solid rgba(127,127,127,.6);
    color: {subtle};
    background: rgba(127,127,127,.08);
  }}

  .ttsbar {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    backdrop-filter: blur(8px);
    background: {panel};
    border-bottom: 1px solid {border};
    padding: 10px 12px;
    z-index: 9999;
    font-size: 14px;
  }}
  .ttsbar .row {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 6px 0;
  }}
  .ttsbar button {{
    padding: 6px 10px;
    border: 1px solid {border};
    background: transparent;
    color: {fg};
    border-radius: 6px;
    cursor: pointer;
  }}
  .ttsbar select {{
    padding: 6px 8px;
    border: 1px solid {border};
    background: transparent;
    color: {fg};
    border-radius: 6px;
  }}
  .ttsbar input[type="range"] {{
    width: 180px;
  }}
  .ttsbar .tip {{
    margin-left: 10px;
    color: {subtle};
  }}

  .tts-clickable {{
    cursor: pointer;
  }}
  .tts-speaking {{
    outline: 2px solid rgba(255, 80, 80, .55);
    outline-offset: 3px;
    border-radius: 6px;
  }}
</style>
</head>
<body>
  {tts_toolbar}
  <div class="reader">{body_html}</div>
  {tts_script}
</body>
</html>"""


# ============================================================
# Edge TTS helpers (MP3)
# ============================================================
def chunk_text(text: str, max_chars: int = 3000):
    paras = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf += "\n\n" + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    out = []
    for c in chunks:
        if len(c) <= max_chars:
            out.append(c)
        else:
            for i in range(0, len(c), max_chars):
                out.append(c[i:i + max_chars])
    return out


def strip_id3_if_present(mp3_bytes: bytes) -> bytes:
    if len(mp3_bytes) < 10:
        return mp3_bytes
    if mp3_bytes[:3] != b"ID3":
        return mp3_bytes
    size_bytes = mp3_bytes[6:10]
    size = ((size_bytes[0] & 0x7F) << 21) | ((size_bytes[1] & 0x7F) << 14) | ((size_bytes[2] & 0x7F) << 7) | (size_bytes[3] & 0x7F)
    start = 10 + size
    return mp3_bytes[start:] if start < len(mp3_bytes) else mp3_bytes


async def edge_tts_mp3_from_text(text: str, voice: str, rate: str, pitch: str, volume: str) -> bytes:
    audio = bytearray()
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch, volume=volume)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


async def edge_tts_mp3_long(text: str, voice: str, rate: str, pitch: str, volume: str, max_chars: int = 3000) -> bytes:
    parts = chunk_text(text, max_chars=max_chars)
    out = bytearray()
    for i, part in enumerate(parts):
        mp3_part = await edge_tts_mp3_from_text(part, voice, rate, pitch, volume)
        if i > 0:
            mp3_part = strip_id3_if_present(mp3_part)
        out.extend(mp3_part)
    return bytes(out)


# ============================================================
# Google Gemini TTS helpers (WAV in memory)
# ============================================================
def pcm16_to_wav_bytes(pcm: bytes, rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def gemini_tts_wav(text: str, voice_name: str, model: str, style: str = "") -> bytes:
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-genai 未安装")

    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    contents = text if not style.strip() else f"{style.strip()}\n\n{text}"

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            )
        )
    )

    data = resp.candidates[0].content.parts[0].inline_data.data
    if isinstance(data, str):
        pcm = base64.b64decode(data)
    else:
        pcm = data
    return pcm16_to_wav_bytes(pcm, rate=24000)


# ============================================================
# Translation helpers
# ============================================================
@st.cache_data(show_spinner=False)
def translate_en_to_zh_openai(text: str, model: str) -> str:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai 未安装")
    client = OpenAI()
    prompt = (
        "请把下面英文翻译成简体中文。\n"
        "要求：\n"
        "1) 尽量逐段对应（保留段落换行）。\n"
        "2) 不要添加解释、注释或多余内容。\n"
        "3) 保持人名/地名一致。\n"
        "4) 语气自然，保留原文风格。\n\n"
        f"{text}"
    )
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text.strip()


@st.cache_data(show_spinner=False)
def translate_en_to_zh_gemini(text: str, model: str = "gemini-2.0-flash", style: str = "") -> str:
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-genai 未安装")

    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    prompt = (
        "请把下面英文翻译成简体中文。\n"
        "要求：\n"
        "1) 尽量逐段对应（保留段落换行）。\n"
        "2) 不要添加解释、注释或多余内容。\n"
        "3) 保持人名/地名一致。\n"
        "4) 语气自然，保留原文风格。\n"
    )
    if style.strip():
        prompt += f"\n额外风格要求：{style.strip()}\n"
    prompt += "\n---\n" + text

    resp = client.models.generate_content(model=model, contents=prompt)
    out = getattr(resp, "text", None)
    if out and out.strip():
        return out.strip()
    try:
        return resp.candidates[0].content.parts[0].text.strip()
    except Exception:
        raise RuntimeError("Gemini 返回格式异常，未能提取文本。")


# ============================================================
# UI
# ============================================================
st.title("EPUB 在线阅读 + 在线朗读 + 对照翻译（Streamlit）")

with st.sidebar:
    st.header("打开 EPUB")
    uploaded = st.file_uploader("上传 EPUB（建议 DRM-free）", type=["epub"])

    st.divider()
    st.subheader("阅读设置")
    theme = st.radio("主题", ["Light", "Dark"], horizontal=True, index=1)
    font_size = st.slider("字号", 14, 28, 18, 1)
    line_height = st.slider("行距", 1.2, 2.2, 1.7, 0.05)
    max_width = st.slider("版心宽度（px）", 520, 1100, 780, 10)

    view_mode = st.radio("显示模式", ["排版（HTML）", "纯文本", "对照翻译（英->中）"], index=0)
    embed_images = st.checkbox("嵌入图片（有插图时更完整）", value=True)

    st.divider()
    st.subheader("在线朗读（浏览器）")
    enable_web_speech = st.checkbox("启用：点击段落从该处开始读（不生成MP3）", value=True)
    st.caption("说明：浏览器朗读依赖系统/浏览器语音包；音色由浏览器决定。")

    st.divider()
    st.caption("提示：仅用于你有合法权限阅读的 EPUB；DRM 书籍可能无法解析。")

if not uploaded:
    st.info("在左侧上传 EPUB 文件后即可开始阅读。")
    st.stop()

epub_bytes = uploaded.getvalue()
book_hash = hashlib.sha256(epub_bytes).hexdigest()

try:
    book = parse_epub(epub_bytes)
except Exception as e:
    st.error(f"解析失败：{e}")
    st.stop()

if "book_hash" not in st.session_state or st.session_state.book_hash != book_hash:
    st.session_state.book_hash = book_hash
    st.session_state.chapter_idx = 0
    st.session_state.page_idx = 0

meta_cols = st.columns([1, 4])
with meta_cols[0]:
    if book.get("cover_path"):
        try:
            ztmp = zipfile.ZipFile(io.BytesIO(epub_bytes))
            cover_bytes = ztmp.read(book["cover_path"])
            st.image(cover_bytes, use_container_width=True)
        except Exception:
            pass

with meta_cols[1]:
    st.subheader(book.get("title", "Untitled"))
    if book.get("creator"):
        st.write(book["creator"])
    if book.get("language"):
        st.caption(f"Language: {book['language']}")

chapter_count = len(book["spine_paths"])
labels = book["chapter_titles"]

selected = st.sidebar.selectbox(
    "目录（按章节顺序）",
    options=list(range(chapter_count)),
    index=st.session_state.chapter_idx,
    format_func=lambda i: labels[i] if 0 <= i < len(labels) else f"第 {i + 1} 章",
)
if selected != st.session_state.chapter_idx:
    st.session_state.chapter_idx = int(selected)
    st.session_state.page_idx = 0

nav_cols = st.columns([1, 1, 6])
with nav_cols[0]:
    if st.button("上一章", disabled=(st.session_state.chapter_idx <= 0), use_container_width=True):
        st.session_state.chapter_idx -= 1
        st.session_state.page_idx = 0
        st.rerun()
with nav_cols[1]:
    if st.button("下一章", disabled=(st.session_state.chapter_idx >= chapter_count - 1), use_container_width=True):
        st.session_state.chapter_idx += 1
        st.session_state.page_idx = 0
        st.rerun()
with nav_cols[2]:
    st.markdown(f"### {labels[st.session_state.chapter_idx]}")

chapter_text, chapter_body_html = extract_chapter_text_and_html(
    epub_bytes=epub_bytes,
    book=book,
    chapter_idx=st.session_state.chapter_idx,
    embed_images=(embed_images and view_mode == "排版（HTML）"),
)

if view_mode == "纯文本":
    safe_text = chapter_text.replace("\n", "  \n")
    st.markdown(safe_text)

elif view_mode == "对照翻译（英->中）":
    blocks = extract_chapter_blocks(epub_bytes, book, st.session_state.chapter_idx)
    per_page = st.sidebar.slider("每页段落数（决定对照粒度）", 4, 18, 8, 1)
    pages = paginate_blocks(blocks, per_page=per_page)

    if not pages:
        st.warning("本章无可分页内容。")
    else:
        max_page = len(pages)
        page_idx = st.sidebar.slider("页", 1, max_page, min(max_page, st.session_state.page_idx + 1), 1) - 1
        st.session_state.page_idx = int(page_idx)

        page_blocks = pages[page_idx]
        left_html = "".join(b["html"] for b in page_blocks)
        src_text = "\n\n".join(b["text"] for b in page_blocks)

        # best-effort english detection
        is_english = book.get("language", "").lower().startswith("en")
        if not is_english:
            ascii_ratio = sum(1 for c in src_text if ord(c) < 128) / max(1, len(src_text))
            is_english = ascii_ratio > 0.7

        colL, colR = st.columns(2, gap="large")

        with colL:
            st.markdown("#### 原文")
            full_html = wrap_reader_html(
                body_html=left_html,
                font_size=font_size,
                line_height=line_height,
                max_width=max_width,
                theme=theme,
                enable_web_speech=False,
                speech_lang="en-US",
            )
            components.html(full_html, height=780, scrolling=True)

        with colR:
            st.markdown("#### 中文翻译")
            if not is_english:
                st.info("检测到本页不像英文，未自动翻译。")
                st.text_area("内容", src_text, height=780, key="not_en_text_area")
            else:
                provider = st.radio("翻译引擎", ["Gemini", "OpenAI"], horizontal=True, index=0)
                style = st.text_input("翻译风格（可选）", value="")

                if provider == "Gemini":
                    if not GEMINI_AVAILABLE:
                        st.error("未安装 google-genai：请 pip install google-genai")
                    else:
                        gemini_model = st.text_input("Gemini 翻译模型", value="gemini-2.0-flash")
                        if st.button("翻译当前页（Gemini）", use_container_width=True):
                            try:
                                with st.spinner("正在翻译（Gemini）…"):
                                    zh = translate_en_to_zh_gemini(src_text, model=gemini_model, style=style)
                                st.text_area("翻译结果（可复制）", zh, height=780, key="zh_gemini")
                            except Exception as e:
                                st.error(f"翻译失败：{e}")
                                st.caption("检查：是否已设置 GEMINI_API_KEY / GOOGLE_API_KEY（环境变量或 Streamlit Secrets）？")
                else:
                    if not OPENAI_AVAILABLE:
                        st.error("未安装 openai：请 pip install openai")
                    else:
                        openai_model = st.text_input("OpenAI 翻译模型", value="gpt-4o-mini")
                        if st.button("翻译当前页（OpenAI）", use_container_width=True):
                            try:
                                with st.spinner("正在翻译（OpenAI）…"):
                                    zh = translate_en_to_zh_openai(src_text, model=openai_model)
                                st.text_area("翻译结果（可复制）", zh, height=780, key="zh_openai")
                            except Exception as e:
                                st.error(f"翻译失败：{e}")
                                st.caption("检查：是否已设置 OPENAI_API_KEY？")

else:
    speech_lang = "en-US" if (book.get("language", "").lower().startswith("en")) else "zh-CN"
    full_html = wrap_reader_html(
        body_html=chapter_body_html,
        font_size=font_size,
        line_height=line_height,
        max_width=max_width,
        theme=theme,
        enable_web_speech=enable_web_speech,
        speech_lang=speech_lang,
    )
    components.html(full_html, height=900, scrolling=True)

# ============================================================
# Optional: TTS (Edge -> MP3)
# ============================================================
with st.expander("生成本章 MP3（Edge TTS，可下载）", expanded=False):
    if not EDGE_TTS_AVAILABLE:
        st.info("未安装 edge-tts：运行 `pip install edge-tts` 后重启 Streamlit。")
    else:
        tts_cols = st.columns([2, 2, 2, 2])
        with tts_cols[0]:
            voice = st.selectbox(
                "中文声音（Edge TTS）",
                options=[
                    "zh-CN-XiaoxiaoNeural",
                    "zh-CN-YunxiNeural",
                    "zh-CN-YunyangNeural",
                    "zh-CN-XiaoyiNeural",
                    "zh-CN-liaoning-XiaobeiNeural",
                    "zh-TW-HsiaoChenNeural",
                    "zh-TW-YunJheNeural",
                    "zh-HK-HiuMaanNeural",
                ],
                index=0,
            )
        with tts_cols[1]:
            rate_pct = st.slider("语速（%）", -40, 40, 0, 5)
        with tts_cols[2]:
            pitch_hz = st.slider("音高（Hz）", -20, 20, 0, 1)
        with tts_cols[3]:
            vol_pct = st.slider("音量（%）", -20, 20, 0, 1)

        max_chars = st.slider("单段最大字符（越小越稳）", 1500, 5000, 3000, 250)
        rate = f"{rate_pct:+d}%"
        pitch = f"{pitch_hz:+d}Hz"
        volume = f"{vol_pct:+d}%"

        st.caption("提示：章节很长时会自动分段合成并拼接为一个 MP3。")

        if st.button("生成本章 MP3", use_container_width=True):
            if not chapter_text.strip():
                st.warning("本章内容为空或无法提取文本。")
            else:
                with st.spinner("正在生成音频…"):
                    audio_bytes = run_coro(
                        edge_tts_mp3_long(
                            text=chapter_text,
                            voice=voice,
                            rate=rate,
                            pitch=pitch,
                            volume=volume,
                            max_chars=max_chars,
                        )
                    )
                st.audio(audio_bytes, format="audio/mp3")
                safe_name = re.sub(r'[\\/:*?"<>|]+', "_", labels[st.session_state.chapter_idx])
                filename = f"{book.get('title','book')}-{safe_name}.mp3"
                st.download_button(
                    "下载本章 MP3",
                    data=audio_bytes,
                    file_name=filename,
                    mime="audio/mpeg",
                    use_container_width=True,
                )

# ============================================================
# Optional: Google Gemini TTS (Kore etc.) -> WAV playback
# ============================================================
with st.expander("在线朗读（Google Gemini TTS：Kore 等，播放 WAV，不生成 MP3）", expanded=False):
    if not GEMINI_AVAILABLE:
        st.info("未安装 google-genai：运行 `pip install google-genai` 后重启。")
    else:
        st.caption("需要设置 GEMINI_API_KEY 或 GOOGLE_API_KEY（环境变量或 Streamlit Secrets）。")
        voice = st.selectbox("Google 声音（预置）", ["Kore", "Zephyr", "Puck", "Charon", "Fenrir"], index=0)
        model = st.text_input("Gemini TTS 模型", value="gemini-2.5-flash-preview-tts")
        style = st.text_input("风格指令（可选）", value="请用温柔、自然、略慢的语气朗读：")

        blocks = extract_chapter_blocks(epub_bytes, book, st.session_state.chapter_idx)
        if not blocks:
            st.warning("本章无可朗读段落。")
        else:
            start_idx = st.number_input("从第几段开始", min_value=1, max_value=len(blocks), value=1, step=1)
            count = st.number_input("朗读多少段", min_value=1, max_value=min(40, len(blocks)), value=min(8, len(blocks)), step=1)
            tts_text = "\n\n".join(b["text"] for b in blocks[int(start_idx) - 1: int(start_idx) - 1 + int(count)])

            if st.button("生成并播放（Gemini TTS）", use_container_width=True):
                try:
                    with st.spinner("正在生成音频（WAV）…"):
                        wav_bytes = gemini_tts_wav(tts_text, voice_name=voice, model=model, style=style)
                    st.audio(wav_bytes, format="audio/wav")
                except Exception as e:
                    st.error(f"生成失败：{e}")
                    st.caption("检查：Key 是否设置、模型名称是否可用。")
