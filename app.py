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
st.set_page_config(page_title="EPUB AI Reader", layout="wide")

# ============================================================
# Helpers
# ============================================================
def normalize_zip_path(path: str) -> str:
    path = (path or "").replace("\\", "/")
    path = re.sub(r"^\./", "", path)
    return posixpath.normpath(path)

def resolve_href(base_dir: str, href: str):
    href = (href or "").strip()
    if not href:
        return None, ""
    if re.match(r"^[a-zA-Z]+://", href):
        return None, ""
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
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "gif": "image/gif", "svg": "image/svg+xml", "webp": "image/webp",
        "css": "text/css",
    }.get(ext, "application/octet-stream")

def soup_html(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

def first_child_text(parent, tag_suffix: str):
    if parent is None: return None
    for el in parent.iter():
        if isinstance(el.tag, str) and el.tag.endswith(tag_suffix):
            if el.text and el.text.strip():
                return el.text.strip()
    return None

def run_coro(coro):
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
    for n in names:
        if hasattr(st, "secrets") and n in st.secrets and str(st.secrets.get(n, "")).strip():
            return str(st.secrets.get(n)).strip()
        if str(os.environ.get(n, "")).strip():
            return str(os.environ.get(n)).strip()
    return ""

# ============================================================
# TOC & EPUB Parsing
# ============================================================
def parse_nav_toc(z: zipfile.ZipFile, nav_path: str):
    html = decode_bytes(z.read(nav_path))
    soup = soup_html(html)
    nav = soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav", attrs={"role": "doc-toc"})
    if not nav:
        for cand in soup.find_all("nav"):
            if "toc" in str(cand.get("epub:type") or cand.get("type") or "").lower():
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
                title, src = None, None
                for el in child.iter():
                    if not title and isinstance(el.tag, str) and el.tag.endswith("text"):
                        if el.text and el.text.strip(): title = el.text.strip()
                    if not src and isinstance(el.tag, str) and el.tag.endswith("content"):
                        src = el.attrib.get("src")
                if src:
                    path, frag = resolve_href(toc_dir, src)
                    if path:
                        entries.append({"title": title or "Untitled", "href": src, "path": path, "fragment": frag})
                walk(child)
    navmap = next((e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("navMap")), root)
    walk(navmap)
    return entries

@st.cache_data(show_spinner=False)
def parse_epub(epub_bytes: bytes):
    z = zipfile.ZipFile(io.BytesIO(epub_bytes))
    file_list = set(z.namelist())
    if "META-INF/container.xml" not in file_list:
        raise ValueError("无效 EPUB：缺少 META-INF/container.xml")
    
    container = ET.fromstring(z.read("META-INF/container.xml"))
    opf_path = None
    for el in container.iter():
        if isinstance(el.tag, str) and el.tag.endswith("rootfile"):
            opf_path = el.attrib.get("full-path")
            break
    if not opf_path or normalize_zip_path(opf_path) not in file_list:
        raise ValueError("无法找到 OPF 文件")
    
    opf_path = normalize_zip_path(opf_path)
    opf_dir = posixpath.dirname(opf_path)
    opf_root = ET.fromstring(z.read(opf_path))
    
    metadata = next((e for e in opf_root.iter() if isinstance(e.tag, str) and e.tag.endswith("metadata")), None)
    title = first_child_text(metadata, "title") or "Untitled"
    creator = first_child_text(metadata, "creator") or ""
    language = first_child_text(metadata, "language") or ""
    
    manifest = {}
    manifest_el = next((e for e in opf_root.iter() if isinstance(e.tag, str) and e.tag.endswith("manifest")), None)
    if manifest_el is not None:
        for item in list(manifest_el):
            if not (isinstance(item.tag, str) and item.tag.endswith("item")): continue
            iid, href = item.attrib.get("id"), item.attrib.get("href")
            if iid and href:
                path = normalize_zip_path(posixpath.join(opf_dir, href))
                manifest[iid] = {"id": iid, "href": href, "path": path, "media_type": item.attrib.get("media-type", ""), "properties": item.attrib.get("properties", "")}

    spine_el = next((e for e in opf_root.iter() if isinstance(e.tag, str) and e.tag.endswith("spine")), None)
    spine_paths = []
    if spine_el is not None:
        for itemref in list(spine_el):
            if isinstance(itemref.tag, str) and itemref.tag.endswith("itemref"):
                idref = itemref.attrib.get("idref")
                if idref and idref in manifest:
                    m = manifest[idref]
                    if "html" in (m.get("media_type") or "").lower():
                        if m["path"] in file_list:
                            spine_paths.append(m["path"])

    if not spine_paths: raise ValueError("未找到可阅读章节")
    
    nav_path = next((m["path"] for m in manifest.values() if "nav" in (m.get("properties") or "").split()), None)
    ncx_path = None
    ncx_id = spine_el.attrib.get("toc") if spine_el is not None else None
    if ncx_id and ncx_id in manifest: ncx_path = manifest[ncx_id]["path"]
    
    toc_entries = []
    if nav_path and nav_path in file_list: toc_entries = parse_nav_toc(z, nav_path)
    elif ncx_path and ncx_path in file_list: toc_entries = parse_ncx_toc(z, ncx_path)
    
    path_to_idx = {p: i for i, p in enumerate(spine_paths)}
    chapter_titles = [None] * len(spine_paths)
    for e in toc_entries:
        if e["path"] in path_to_idx:
            idx = path_to_idx[e["path"]]
            if not chapter_titles[idx]: chapter_titles[idx] = e.get("title")
    for i in range(len(chapter_titles)):
        if not chapter_titles[i]: chapter_titles[i] = f"第 {i+1} 章"
        
    cover_path = next((m["path"] for m in manifest.values() if "cover-image" in (m.get("properties") or "").split()), None)
    
    return {
        "title": title, "creator": creator, "language": language,
        "spine_paths": spine_paths, "chapter_titles": chapter_titles,
        "mime_by_path": {m["path"]: m.get("media_type") for m in manifest.values()},
        "cover_path": cover_path, "file_list": file_list
    }

# ============================================================
# Extraction & HTML
# ============================================================
def embed_images_in_body(body, chapter_path: str, z: zipfile.ZipFile, mime_by_path: dict, file_list: set):
    base_dir = posixpath.dirname(chapter_path)
    for img in body.find_all("img"):
        src = img.get("src")
        target, _ = resolve_href(base_dir, src)
        if target and target in file_list:
            raw = z.read(target)
            mime = mime_by_path.get(target) or guess_mime(target)
            b64 = base64.b64encode(raw).decode("ascii")
            img["src"] = f"data:{mime};base64,{b64}"

def extract_chapter_content(epub_bytes: bytes, book: dict, chapter_idx: int, embed_images: bool):
    z = zipfile.ZipFile(io.BytesIO(epub_bytes))
    path = book["spine_paths"][chapter_idx]
    html = decode_bytes(z.read(path))
    soup = soup_html(html)
    body = soup.body or soup
    for s in body.find_all("script"): s.decompose()
    
    if embed_images:
        embed_images_in_body(body, path, z, book["mime_by_path"], book["file_list"])
    
    # Extract blocks for TTS
    blocks = []
    # Identify paragraphs and headings for granular TTS
    for el in body.find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "div"]):
        # 只取直接包含文本的元素或常见的文本容器
        txt = el.get_text(" ", strip=True)
        if txt and len(txt) > 1: # 忽略太短的杂音
             # 简单的去重策略：如果父元素已经包含了这个文本，这里尽量避免重复，
             # 但为了简单，我们主要抓取叶子节点或近似叶子节点
             blocks.append({"text": txt, "tag": el.name})
    
    # Fallback if structure is weird
    if not blocks:
        txt = body.get_text("\n", strip=True)
        parts = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        blocks = [{"text": p, "tag": "p"} for p in parts]

    # Inject Markers into HTML for UI (visualizing paragraphs)
    # 这一步为了不破坏 HTML 结构比较复杂，我们简单地生成用于显示的 HTML，不强行注入 ID 到原 soup
    # 但为了让用户知道“第几段”，我们可以在 sidebar 显示段落预览
    
    return str(body), blocks

def wrap_reader_html(body_html: str, font_size: int, line_height: float, max_width: int, theme: str):
    bg, fg = ("#0f1115", "#e6e6e6") if theme == "Dark" else ("#ffffff", "#111111")
    subtle = "#b6b6b6" if theme == "Dark" else "#555555"
    
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{ background: {bg}; color: {fg}; font-size: {font_size}px; line-height: {line_height}; 
          font-family: sans-serif; margin: 0; padding: 20px; }}
  .reader {{ max-width: {max_width}px; margin: 0 auto; padding-bottom: 100px; }}
  img {{ max-width: 100%; height: auto; }}
  blockquote {{ border-left: 3px solid #777; padding-left: 10px; color: {subtle}; margin: 1em 0; }}
  a {{ color: inherit; }}
</style>
</head>
<body><div class="reader">{body_html}</div></body>
</html>"""

# ============================================================
# Gemini TTS Logic
# ============================================================
def pcm16_to_wav_bytes(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def gemini_tts_segment(text: str, voice_name: str, model: str) -> bytes:
    """生成单个片段的音频 (WAV)"""
    if not GEMINI_AVAILABLE: return b""
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key: return b""
    
    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=text,
            config=genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                )
            )
        )
        data = resp.candidates[0].content.parts[0].inline_data.data
        pcm = base64.b64decode(data) if isinstance(data, str) else data
        return pcm
    except Exception as e:
        print(f"Gemini TTS Error: {e}")
        return b""

def generate_chapter_audio(blocks: list, start_idx: int, count: int, voice: str, model: str):
    """组合多个段落生成一个连续的 WAV"""
    end_idx = min(start_idx + count, len(blocks))
    if start_idx >= end_idx:
        return None, start_idx
    
    # 提取文本
    segment_texts = [b["text"] for b in blocks[start_idx:end_idx]]
    # Gemini 对单次请求长度有限制，但这里我们通过 python 循环拼接 PCM 数据
    # 或者如果文本不太长，合并发送。为了稳妥，按段落组装发送，或者每 ~1000 字符发送一次。
    # 为了流畅度，我们尝试将文本合并，中间加停顿符。
    
    full_text = "\n\n".join(segment_texts)
    
    # 如果文本过长(>3000 chars)，简单切分一下防止超时
    # 这里简化处理：直接请求。Gemini 2.5 Flash 处理长文本能力较强。
    pcm_data = gemini_tts_segment(full_text, voice, model)
    
    if not pcm_data:
        return None, start_idx

    wav_bytes = pcm16_to_wav_bytes(pcm_data)
    return wav_bytes, end_idx

# ============================================================
# Custom Audio Player (HTML/JS) for Speed Control
# ============================================================
def render_custom_audio_player(audio_bytes: bytes, playback_rate: float, auto_play: bool = True):
    if not audio_bytes: return
    b64 = base64.b64encode(audio_bytes).decode()
    src = f"data:audio/wav;base64,{b64}"
    
    # Random ID to force re-render if needed
    player_id = f"audio_{hash(b64)[:8]}"
    
    html = f"""
    <div style="background: rgba(127,127,127,0.1); padding: 10px; border-radius: 8px; margin-bottom: 10px;">
        <div style="display:flex; align-items:center; justify-content: space-between; margin-bottom:4px;">
            <span style="font-weight:bold; font-size:0.9em;">AI 播放器 ({playback_rate}x)</span>
            <span style="font-size:0.8em; color: #777;">生成完毕</span>
        </div>
        <audio id="{player_id}" controls {"autoplay" if auto_play else ""} style="width: 100%;">
            <source src="{src}" type="audio/wav">
        </audio>
        <script>
            var aud = document.getElementById("{player_id}");
            aud.playbackRate = {playback_rate};
            
            // Re-apply rate on play just in case
            aud.addEventListener('play', function() {{
                this.playbackRate = {playback_rate};
            }});
        </script>
    </div>
    """
    components.html(html, height=70)


# ============================================================
# Main UI
# ============================================================
st.title("EPUB AI 阅读器 (Gemini 语音版)")

if not GEMINI_AVAILABLE:
    st.error("未检测到 `google-genai` 库。请运行 `pip install google-genai` 并重启应用。")

# --- Sidebar: File & Settings ---
with st.sidebar:
    st.header("1. 图书文件")
    uploaded = st.file_uploader("上传 EPUB", type=["epub"])
    
    st.divider()
    st.header("2. AI 朗读设置 (Gemini)")
    
    if not get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        st.warning("未检测到 API Key。请在 secrets.toml 或环境变量设置 GEMINI_API_KEY。")
        api_ready = False
    else:
        api_ready = True

    # Voice selection
    voice = st.selectbox("朗读声音", ["Kore", "Zephyr", "Puck", "Charon", "Fenrir"], index=1, 
                         help="Kore(女/沉稳), Zephyr(女/温柔), Puck(男/活泼), Charon(男/深沉), Fenrir(男/有力)")
    
    # Speed control (Client side)
    speed = st.slider("播放速度", 0.5, 2.0, 1.25, 0.1, help="由播放器端控制加速")
    
    # Generation config
    batch_size = st.number_input("单次生成段落数", 5, 50, 20, help="每次点击播放时生成的文本量。太长可能等待较久。")
    gemini_model = "gemini-2.5-flash-preview-tts" # Hardcoded for best TTS support currently

    st.divider()
    st.header("3. 阅读视觉")
    theme = st.radio("主题", ["Light", "Dark"], index=1, horizontal=True)
    font_size = st.slider("字号", 14, 32, 18)

if not uploaded:
    st.info("请先在左侧上传 EPUB 文件。")
    st.stop()

# --- Load Book ---
epub_bytes = uploaded.getvalue()
book_hash = hashlib.sha256(epub_bytes).hexdigest()

try:
    book = parse_epub(epub_bytes)
except Exception as e:
    st.error(f"解析出错: {e}")
    st.stop()

# Initialize Session State
if "book_hash" not in st.session_state or st.session_state.book_hash != book_hash:
    st.session_state.book_hash = book_hash
    st.session_state.chapter_idx = 0
    st.session_state.current_start_paragraph = 0 # 记录当前阅读/朗读到的段落索引
    st.session_state.audio_cache = None # 存储当前生成的音频

# --- Chapter Navigation ---
chapter_titles = book["chapter_titles"]
col_nav = st.columns([2, 5, 1])
with col_nav[0]:
    selected_chap = st.selectbox("选择章节", range(len(chapter_titles)), 
                                 format_func=lambda i: chapter_titles[i], 
                                 index=st.session_state.chapter_idx)
    if selected_chap != st.session_state.chapter_idx:
        st.session_state.chapter_idx = selected_chap
        st.session_state.current_start_paragraph = 0 # 换章重置段落
        st.session_state.audio_cache = None
        st.rerun()

with col_nav[2]:
    if st.button("下一章", disabled=st.session_state.chapter_idx >= len(chapter_titles)-1):
        st.session_state.chapter_idx += 1
        st.session_state.current_start_paragraph = 0
        st.session_state.audio_cache = None
        st.rerun()

# --- Content Extraction ---
# 我们需要同时获取 HTML 用于显示，和 Blocks 用于 TTS 定位
body_html, blocks = extract_chapter_content(epub_bytes, book, st.session_state.chapter_idx, embed_images=True)
total_paragraphs = len(blocks)

# --- AI Player Control Area (Top of Content) ---
st.markdown("### 🎧 AI 有声播放")

player_col1, player_col2 = st.columns([1, 2])

with player_col1:
    # 允许用户手动修改起始位置
    start_para = st.number_input(
        f"起始段落 (共 {total_paragraphs} 段)", 
        min_value=0, max_value=max(0, total_paragraphs-1), 
        value=st.session_state.current_start_paragraph
    )
    # 如果用户手动改了数字，更新状态
    if start_para != st.session_state.current_start_paragraph:
        st.session_state.current_start_paragraph = start_para
        # 清除旧音频，因为位置变了
        st.session_state.audio_cache = None 

with player_col2:
    st.write(" ") # Spacer
    st.write(" ")
    btn_col = st.columns([1, 1])
    with btn_col[0]:
        # 主要的播放按钮
        if st.button("▶️ 生成并播放", use_container_width=True, type="primary", disabled=not api_ready):
            if total_paragraphs == 0:
                st.warning("本章没有可朗读的文本。")
            else:
                with st.spinner("正在请求 Google Gemini 生成语音..."):
                    wav_data, new_end_idx = generate_chapter_audio(
                        blocks, 
                        start_idx=st.session_state.current_start_paragraph, 
                        count=batch_size, 
                        voice=voice, 
                        model=gemini_model
                    )
                    if wav_data:
                        st.session_state.audio_cache = wav_data
                        # 我们不自动更新 start_paragraph，直到用户点击“标记为已读”或者“下一段”
                        # 但为了方便连续播放，我们可以记录下一次生成的起点
                        st.session_state.next_start_paragraph = new_end_idx
                    else:
                        st.error("生成失败，请检查 API Key 或网络。")

    with btn_col[1]:
        # 能够让用户快速跳过已生成的段落
        if "next_start_paragraph" in st.session_state and st.session_state.next_start_paragraph > st.session_state.current_start_paragraph:
             if st.button(f"⏩ 跳至第 {st.session_state.next_start_paragraph} 段 (加载后续)", use_container_width=True):
                 st.session_state.current_start_paragraph = st.session_state.next_start_paragraph
                 st.session_state.audio_cache = None
                 st.rerun()

# --- Render Player if audio exists ---
if st.session_state.audio_cache:
    render_custom_audio_player(st.session_state.audio_cache, playback_rate=speed)
    
    # 预览即将朗读的文本片段
    end_p = min(st.session_state.current_start_paragraph + batch_size, total_paragraphs)
    preview_text = " ".join([b["text"] for b in blocks[st.session_state.current_start_paragraph : end_p]])[:150] + "..."
    st.caption(f"当前音频内容 ({st.session_state.current_start_paragraph} - {end_p} 段): {preview_text}")

st.divider()

# --- Reader View ---
# 我们在 HTML 中高亮当前的起始段落，或者简单的显示全文
full_html = wrap_reader_html(body_html, font_size, 1.6, 800, theme)
components.html(full_html, height=800, scrolling=True)

# 底部状态栏
st.caption(f"当前: {book.get('title')} - {chapter_titles[st.session_state.chapter_idx]}")
