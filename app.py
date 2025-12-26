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
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from google import genai
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
                        if el.text and el.text.strip():
                            title = el.text.strip()
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
        raise ValueError("EPUB 缺少 META-INF/container.xml")

    container_root = ET.fromstring(z.read("META-INF/container.xml"))
    opf_path = None
    for el in container_root.iter():
        if isinstance(el.tag, str) and el.tag.endswith("rootfile"):
            opf_path = el.attrib.get("full-path")
            break
    if not opf_path:
        raise ValueError("无法找到 rootfile (OPF)")

    opf_path = normalize_zip_path(opf_path)
    if opf_path not in file_list:
        raise ValueError(f"OPF 文件不存在: {opf_path}")

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
            if iid and href:
                path = normalize_zip_path(posixpath.join(opf_dir, href))
                manifest[iid] = {"id": iid, "href": href, "path": path, "media_type": media_type, "properties": props}

    spine_el = next((e for e in opf_root.iter() if isinstance(e.tag, str) and e.tag.endswith("spine")), None)
    spine_idrefs = []
    ncx_id = spine_el.attrib.get("toc") if spine_el is not None else None
    if spine_el is not None:
        for itemref in list(spine_el):
            if isinstance(itemref.tag, str) and itemref.tag.endswith("itemref"):
                idref = itemref.attrib.get("idref")
                if idref: spine_idrefs.append(idref)

    spine_paths = []
    for idref in spine_idrefs:
        m = manifest.get(idref)
        if m and ("html" in (m.get("media_type") or "").lower()) and m["path"] in file_list:
            spine_paths.append(m["path"])

    if not spine_paths:
        raise ValueError("未能解析到可阅读章节")

    nav_path = next((m["path"] for m in manifest.values() if "nav" in (m.get("properties") or "").split()), None)
    ncx_path = None
    if ncx_id and ncx_id in manifest:
        ncx_path = manifest[ncx_id]["path"]
    
    cover_path = next((m["path"] for m in manifest.values() if "cover-image" in (m.get("properties") or "").split()), None)
    if cover_path and cover_path not in file_list: cover_path = None

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
                chapter_titles[idx] = e.get("title")
    for i in range(len(chapter_titles)):
        if not chapter_titles[i]:
            chapter_titles[i] = f"第 {i+1} 章"

    return {
        "title": title,
        "creator": creator,
        "language": language,
        "spine_paths": spine_paths,
        "chapter_titles": chapter_titles,
        "mime_by_path": {m["path"]: m.get("media_type") for m in manifest.values()},
        "cover_path": cover_path,
        "file_list": file_list,
    }


# ============================================================
# Chapter & Block Extraction
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


def extract_chapter_blocks(epub_bytes: bytes, book: dict, chapter_idx: int, embed_images: bool):
    z = zipfile.ZipFile(io.BytesIO(epub_bytes))
    chapter_path = book["spine_paths"][chapter_idx]
    html = decode_bytes(z.read(chapter_path))
    soup = soup_html(html)
    body = soup.body or soup
    for s in body.find_all("script"):
        s.decompose()

    if embed_images:
        embed_images_in_body(body, chapter_path, z, book["mime_by_path"], book["file_list"])

    # 提取结构化块，用于对照翻译和TTS定位
    blocks = []
    # 查找所有可能有文本的块级元素
    for el in body.find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "div"]):
        # 简单过滤：如果该元素包含子级块元素，跳过（防止重复）
        if el.find(["p", "li", "h1", "h2", "h3"]):
            continue
        
        txt = el.get_text(" ", strip=True)
        if txt and len(txt) > 1: # 过滤掉只有符号的
            # 为每个块分配一个 ID，用于 HTML 锚点
            block_id = f"block_{len(blocks)}"
            blocks.append({
                "id": block_id,
                "text": txt,
                "html": str(el), # 保留原始标签样式
                "tag": el.name
            })
    
    # 兜底：如果没提取到
    if not blocks:
        txt = body.get_text("\n", strip=True)
        parts = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        for i, p in enumerate(parts):
            blocks.append({
                "id": f"block_{i}",
                "text": p,
                "html": f"<p>{p}</p>",
                "tag": "p"
            })
            
    return blocks


def paginate_blocks(blocks, per_page: int):
    return [blocks[i:i + per_page] for i in range(0, len(blocks), per_page)]


# ============================================================
# Translation Logic
# ============================================================
@st.cache_data(show_spinner=False)
def translate_en_to_zh_gemini(text: str, model: str = "gemini-2.0-flash", style: str = "") -> str:
    if not GEMINI_AVAILABLE: return "Error: google-genai not installed."
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    prompt = (
        "Translate the following text to Simplified Chinese.\n"
        "Requirements: Keep original paragraph structure. Natural flow.\n"
    )
    if style.strip(): prompt += f"Style: {style.strip()}\n"
    prompt += f"\n---\n{text}"

    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text.strip()
    except Exception as e:
        return f"Translation Error: {e}"


@st.cache_data(show_spinner=False)
def translate_en_to_zh_openai(text: str, model: str) -> str:
    if not OPENAI_AVAILABLE: return "Error: openai not installed."
    client = OpenAI()
    prompt = f"Translate to Simplified Chinese. Keep paragraphs.\n\n{text}"
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()


# ============================================================
# Gemini Audio Logic
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
def get_gemini_audio_for_blocks(text: str, voice: str, model: str) -> bytes:
    """调用 Gemini 生成音频"""
    if not GEMINI_AVAILABLE: return b""
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    
    # 限制长度，防止超时
    safe_text = text[:4500] 

    try:
        resp = client.models.generate_content(
            model=model,
            contents=safe_text,
            config=genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                )
            )
        )
        data = resp.candidates[0].content.parts[0].inline_data.data
        pcm = base64.b64decode(data) if isinstance(data, str) else data
        return pcm16_to_wav_bytes(pcm)
    except Exception as e:
        print(f"TTS Error: {e}")
        return b""


# ============================================================
# Reader HTML with "Click-to-Play" Injection
# ============================================================
def wrap_reader_html_interactive(blocks, font_size, line_height, max_width, theme, 
                                 current_playing_idx=None, speed=1.0):
    """
    生成 HTML，其中每个段落都包裹在可点击的 JS 中。
    点击段落 -> 改变 URL query param -> Streamlit 重新运行 -> 播放音频
    """
    if theme == "Dark":
        bg, fg, subtle, highlight = "#0f1115", "#e6e6e6", "#b6b6b6", "rgba(255, 200, 100, 0.2)"
    else:
        bg, fg, subtle, highlight = "#ffffff", "#111111", "#555555", "rgba(255, 200, 0, 0.3)"

    html_content = []
    
    # 注入一段 JS，用于处理点击和 URL 更新
    # 使用 window.parent.location.search 来更新 Streamlit 的 query params
    # 这会触发 Streamlit 刷新
    script = """
    <script>
    function playBlock(idx) {
        // 构建新的 URL，添加或更新 play_idx 参数
        const url = new URL(window.parent.location.href);
        url.searchParams.set('play_idx', idx);
        window.parent.location.href = url.toString();
    }
    </script>
    """
    html_content.append(script)

    for i, block in enumerate(blocks):
        # 如果当前正在播放这个段落，加高亮
        style_extra = ""
        if current_playing_idx is not None and i == current_playing_idx:
            style_extra = f"background-color: {highlight}; border-radius: 4px;"
        
        # 将原始 HTML 标签改为带 onclick 的 div 包装，或者直接在标签上加 onclick
        # 为了通用性，我们在外层包一个 div class="clickable-block"
        raw_html = block["html"] 
        # 简单的正则替换，给标签加 onclick，或者直接包裹
        # 包裹更安全
        wrapped = f"""
        <div class="block-wrapper" onclick="playBlock({i})" style="cursor: pointer; margin-bottom: 0.5em; padding: 4px; transition: background 0.2s; {style_extra}" title="点击朗读 (Gemini)">
            {raw_html}
        </div>
        """
        html_content.append(wrapped)

    body_inner = "".join(html_content)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{
    background: {bg}; color: {fg}; margin: 0; padding: 20px;
    font-size: {font_size}px; line-height: {line_height};
    font-family: sans-serif;
  }}
  .reader {{ max-width: {max_width}px; margin: 0 auto; padding-bottom: 120px; }}
  img {{ max-width: 100%; height: auto; }}
  .block-wrapper:hover {{ background-color: {subtle}22; }}
</style>
</head>
<body>
  <div class="reader">{body_inner}</div>
</body>
</html>"""


# ============================================================
# Main UI
# ============================================================
def main():
    # --- 1. Audio Auto-Play Logic (Must be at top) ---
    # 检查 URL 参数是否有 play_idx，如果有，生成音频并播放
    
    # Streamlit 1.30+ 使用 st.query_params
    query_params = st.query_params
    play_idx_str = query_params.get("play_idx", None)
    
    # 侧边栏设置
    with st.sidebar:
        st.header("1. 打开 EPUB")
        uploaded = st.file_uploader("上传文件", type=["epub"])
        
        st.divider()
        st.header("2. 朗读设置 (Gemini)")
        if not GEMINI_AVAILABLE:
            st.error("请安装 `google-genai`")
        
        # 声音设置
        voice = st.selectbox("声音", ["Kore", "Zephyr", "Puck", "Charon", "Fenrir"], index=0)
        speed = st.slider("语速 (变速)", 0.5, 2.0, 1.2, 0.1)
        auto_continue = st.checkbox("自动连续朗读", value=True, help="一段读完自动跳下一段")
        gemini_model = "gemini-2.5-flash-preview-tts"
        
        st.divider()
        st.header("3. 阅读/翻译设置")
        view_mode = st.radio("模式", ["排版（点击朗读）", "对照翻译（英->中）"], index=0)
        
        theme = st.radio("主题", ["Light", "Dark"], index=1, horizontal=True)
        font_size = st.slider("字号", 14, 30, 18)
        
        if view_mode == "对照翻译（英->中）":
             trans_provider = st.selectbox("翻译引擎", ["Gemini", "OpenAI"])
             trans_style = st.text_input("翻译风格", placeholder="例如：武侠风、学术风...")

    if not uploaded:
        st.info("👈 请在左侧上传 EPUB 文件开始阅读。")
        st.stop()

    # --- Load Book ---
    epub_bytes = uploaded.getvalue()
    book_hash = hashlib.sha256(epub_bytes).hexdigest()
    
    if "book_hash" not in st.session_state or st.session_state.book_hash != book_hash:
        try:
            st.session_state.book_data = parse_epub(epub_bytes)
            st.session_state.book_hash = book_hash
            st.session_state.chapter_idx = 0
            st.session_state.page_idx = 0
        except Exception as e:
            st.error(f"解析失败: {e}")
            st.stop()
            
    book = st.session_state.book_data
    
    # --- Navigation ---
    col_nav = st.columns([2, 4, 2])
    with col_nav[0]:
        if st.button("⬅️ 上一章", use_container_width=True):
            st.session_state.chapter_idx = max(0, st.session_state.chapter_idx - 1)
            st.session_state.page_idx = 0
            st.query_params.clear() # 换章清除播放状态
            st.rerun()
    with col_nav[1]:
        opts = book["chapter_titles"]
        curr = st.session_state.chapter_idx
        sel = st.selectbox("章节", range(len(opts)), index=curr, format_func=lambda i: opts[i], label_visibility="collapsed")
        if sel != curr:
            st.session_state.chapter_idx = sel
            st.session_state.page_idx = 0
            st.query_params.clear()
            st.rerun()
    with col_nav[2]:
        if st.button("下一章 ➡️", use_container_width=True):
            st.session_state.chapter_idx = min(len(opts)-1, st.session_state.chapter_idx + 1)
            st.session_state.page_idx = 0
            st.query_params.clear()
            st.rerun()

    # --- Extract Content ---
    blocks = extract_chapter_blocks(epub_bytes, book, st.session_state.chapter_idx, embed_images=True)
    
    # --- Audio Logic Handling ---
    audio_html = ""
    current_idx = int(play_idx_str) if play_idx_str is not None else None
    
    # 如果 URL 里有 play_idx，说明用户点击了某段，或者自动连播到了某段
    if current_idx is not None and 0 <= current_idx < len(blocks):
        target_block = blocks[current_idx]
        text_to_speak = target_block["text"]
        
        # 简单的预取策略：如果文本太短，可以多读一段（可选，这里保持精准对应）
        # 如果需要连贯，可以在这里合并 blocks[current_idx] 和 blocks[current_idx+1]
        
        if text_to_speak.strip():
            with st.spinner(f"正在生成语音 (段落 {current_idx+1})..."):
                wav_bytes = get_gemini_audio_for_blocks(text_to_speak, voice, gemini_model)
            
            if wav_bytes:
                b64 = base64.b64encode(wav_bytes).decode()
                # 自动播放逻辑
                # 使用 onended 事件触发下一个 URL 更新，实现连播
                next_js = ""
                if auto_continue and current_idx + 1 < len(blocks):
                    next_idx = current_idx + 1
                    # JS: 音频播放结束 -> 修改 URL play_idx -> Streamlit Rerun -> 播放下一段
                    next_js = f"""
                    aud.onended = function() {{
                        const url = new URL(window.parent.location.href);
                        url.searchParams.set('play_idx', {next_idx});
                        window.parent.location.href = url.toString();
                    }};
                    """
                
                # 这里的 Audio 播放器是隐藏的还是显示的？
                # 放在顶部作为一个迷你控制条
                audio_player_html = f"""
                <div style="position: fixed; bottom: 0; left: 0; right: 0; background: #222; color: #fff; padding: 10px; z-index: 9999; display: flex; align-items: center; justify-content: center; box-shadow: 0 -2px 10px rgba(0,0,0,0.5);">
                    <span style="margin-right: 10px; font-size: 0.9em;">正在朗读第 {current_idx+1} 段...</span>
                    <audio id="gemini_audio" controls autoplay style="height: 30px;">
                        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    </audio>
                    <script>
                        var aud = document.getElementById("gemini_audio");
                        aud.playbackRate = {speed};
                        {next_js}
                    </script>
                </div>
                """
                # 使用 components.html 注入播放器，注意高度
                components.html(audio_player_html, height=0) # 设为0高度隐藏iframe，但其中的fixed div会显示
                
                # 由于 iframe 隔离，fixed div 可能显示在 iframe 内部。
                # Streamlit component 的 iframe 很难跳出 sandbox。
                # 妥协方案：直接显示在页面顶部或底部的 Streamlit 原生位置，或者让 iframe 高度足够。
                # 更好的方案：直接把 audio player 放在页面顶部显眼位置。
                st.markdown(f"""
                <audio id="main_audio" controls autoplay style="width: 100%; margin-bottom: 10px;">
                    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                </audio>
                <script>
                    var aud = document.getElementById("main_audio");
                    if(aud) {{
                        aud.playbackRate = {speed};
                        {next_js.replace("window.parent.location", "window.location")} 
                        // 注意：st.markdown 中的 JS 是在同源下执行的吗？
                        // Streamlit 会过滤 script 标签。需要用 components.html。
                    }}
                </script>
                """, unsafe_allow_html=True) 
                # 上面 markdown unsafe_allow_html 的 script 不一定会执行连播跳转。
                # 我们必须依靠 components.html 来执行 JS 跳转。
                
                # 修正播放器显示：放在内容上方
                components.html(f"""
                    <div style="display:flex; align-items:center; background:rgba(0,0,0,0.05); padding:5px; border-radius:5px;">
                       <b style="margin-right:10px; font-size:14px;">▶ 正在朗读 ({current_idx+1}/{len(blocks)})</b>
                       <audio id="player" controls autoplay style="flex:1; height:30px;">
                           <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                       </audio>
                    </div>
                    <script>
                        var aud = document.getElementById("player");
                        aud.playbackRate = {speed};
                        aud.onended = function() {{
                             window.parent.location.search = '?play_idx={current_idx + 1}';
                        }};
                    </script>
                """, height=50)

    # --- Render Main View ---

    if view_mode == "排版（点击朗读）":
        st.info("提示：点击任意段落即可开始 Gemini 朗读（支持自动连播）。")
        # 渲染交互式 HTML
        html = wrap_reader_html_interactive(
            blocks, font_size, 1.6, 800, theme, 
            current_playing_idx=current_idx, speed=speed
        )
        components.html(html, height=800, scrolling=True)

    else: # 对照翻译模式
        st.info("对照模式：左侧原文（点击可读），右侧翻译。")
        per_page = 10
        pages = paginate_blocks(blocks, per_page)
        
        if not pages:
            st.write("本章无内容")
        else:
            total_pages = len(pages)
            page_idx = st.sidebar.number_input("页码", 1, total_pages, st.session_state.page_idx + 1) - 1
            st.session_state.page_idx = page_idx
            
            curr_blocks = pages[page_idx]
            src_text_all = "\n\n".join([b["text"] for b in curr_blocks])
            
            colL, colR = st.columns(2)
            
            with colL:
                # 左侧：原文，也做成可点击朗读
                # 我们需要计算当前页 block 在全局 blocks 的 index
                global_start_idx = page_idx * per_page
                
                # 重新构建这一页的 blocks 数据，带上全局 index 供播放用
                page_blocks_for_html = []
                for i, b in enumerate(curr_blocks):
                    real_idx = global_start_idx + i
                    page_blocks_for_html.append({
                        "html": b["html"],
                        "idx": real_idx # 仅用于追踪
                    })
                
                # 我们可以复用 wrap_reader_html_interactive，但需要传入这一页的切片
                # 这里的 current_playing_idx 需要对应切片的相对位置
                rel_play_idx = None
                if current_idx is not None:
                    if global_start_idx <= current_idx < global_start_idx + per_page:
                        rel_play_idx = current_idx - global_start_idx
                
                # 注意：wrap_reader_html_interactive 接收的是 list of dicts with 'html'
                # 并且它生成的 onclick 是 0, 1, 2... 
                # 我们需要它生成 onclick(global_index)
                
                # 自定义一个简单的渲染器给对照模式
                item_htmls = []
                script = """<script>
                function play(idx) {
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set('play_idx', idx);
                    window.parent.location.href = url.toString();
                }
                </script>"""
                item_htmls.append(script)
                
                for i, b in enumerate(curr_blocks):
                    real_idx = global_start_idx + i
                    bg_style = "background: rgba(255,200,0,0.3);" if real_idx == current_idx else ""
                    item_htmls.append(
                        f'<div onclick="play({real_idx})" style="cursor:pointer; padding:5px; margin-bottom:10px; {bg_style}" title="点击朗读">{b["html"]}</div>'
                    )
                
                full_left = f"""<div style="font-family:sans-serif; color:{'#ddd' if theme=='Dark' else '#222'}">{''.join(item_htmls)}</div>"""
                components.html(full_left, height=1000, scrolling=True)

            with colR:
                # 右侧：翻译
                if st.button("翻译当前页", key=f"trans_{page_idx}", use_container_width=True):
                    with st.spinner("正在翻译..."):
                        if trans_provider == "Gemini":
                            zh = translate_en_to_zh_gemini(src_text_all, model=gemini_model, style=trans_style)
                        else:
                            zh = translate_en_to_zh_openai(src_text_all, model="gpt-4o-mini") # 示例模型
                    st.text_area("译文", zh, height=1000)
                else:
                    st.text_area("译文", "点击上方按钮开始翻译...", height=1000)

if __name__ == "__main__":
    main()
