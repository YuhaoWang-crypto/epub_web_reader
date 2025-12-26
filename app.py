import os
import base64
import hashlib
import io
import re
import wave
import zipfile
import xml.etree.ElementTree as ET
import posixpath

import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup

# ============================================================
# 依赖检查
# ============================================================
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ============================================================
# 配置
# ============================================================
st.set_page_config(page_title="EPUB AI Reader", layout="wide")

TTS_MODEL_ID = "gemini-2.5-flash-preview-tts"
TEXT_MODEL_ID = "gemini-2.0-flash"

# ============================================================
# 辅助函数
# ============================================================
def normalize_zip_path(path: str) -> str:
    path = (path or "").replace("\\", "/")
    path = re.sub(r"^\./", "", path)
    return posixpath.normpath(path)

def resolve_href(base_dir: str, href: str):
    href = (href or "").strip()
    if not href: return None, ""
    if re.match(r"^[a-zA-Z]+://", href): return None, ""
    href_no_frag = href.split("#", 1)[0].split("?", 1)[0]
    fragment = href.split("#", 1)[1] if "#" in href else ""
    target = normalize_zip_path(posixpath.join(base_dir, href_no_frag))
    return target, fragment

def decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "utf-16"):
        try: return b.decode(enc)
        except UnicodeDecodeError: continue
    return b.decode("latin-1", errors="ignore")

def soup_html(html: str) -> BeautifulSoup:
    try: return BeautifulSoup(html, "lxml")
    except: return BeautifulSoup(html, "html.parser")

def first_child_text(parent, tag_suffix: str):
    if parent is None: return None
    for el in parent.iter():
        if isinstance(el.tag, str) and el.tag.endswith(tag_suffix):
            if el.text and el.text.strip(): return el.text.strip()
    return None

def get_secret(*names: str) -> str:
    for n in names:
        if hasattr(st, "secrets") and n in st.secrets and str(st.secrets.get(n, "")).strip():
            return str(st.secrets.get(n)).strip()
        if str(os.environ.get(n, "")).strip():
            return str(os.environ.get(n)).strip()
    return ""

# ============================================================
# EPUB 解析
# ============================================================
def parse_nav_toc(z: zipfile.ZipFile, nav_path: str):
    try:
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
                if path: entries.append({"title": title, "path": path})
        return entries
    except:
        return []

def parse_ncx_toc(z: zipfile.ZipFile, ncx_path: str):
    try:
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
                        if path: entries.append({"title": title or "Untitled", "path": path})
                    walk(child)
        navmap = next((e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("navMap")), root)
        walk(navmap)
        return entries
    except:
        return []

@st.cache_data(show_spinner=False)
def parse_epub(epub_bytes: bytes):
    z = zipfile.ZipFile(io.BytesIO(epub_bytes))
    file_list = set(z.namelist())

    if "META-INF/container.xml" not in file_list:
        raise ValueError("无效文件：缺少 container.xml")

    container = ET.fromstring(z.read("META-INF/container.xml"))
    opf_path = next((el.attrib.get("full-path") for el in container.iter() if el.tag.endswith("rootfile")), None)
    if not opf_path: raise ValueError("无法找到 OPF")
    
    opf_path = normalize_zip_path(opf_path)
    opf_dir = posixpath.dirname(opf_path)
    opf_root = ET.fromstring(z.read(opf_path))
    
    # Metadata
    metadata = next((e for e in opf_root.iter() if e.tag.endswith("metadata")), None)
    title = first_child_text(metadata, "title") or "Untitled"
    
    # Manifest
    manifest = {}
    manifest_el = next((e for e in opf_root.iter() if e.tag.endswith("manifest")), None)
    if manifest_el is not None:
        for item in list(manifest_el):
            if item.tag.endswith("item"):
                iid, href = item.attrib.get("id"), item.attrib.get("href")
                media_type = item.attrib.get("media-type", "")
                props = item.attrib.get("properties", "")
                if iid and href:
                    path = normalize_zip_path(posixpath.join(opf_dir, href))
                    manifest[iid] = {"href": href, "path": path, "media_type": media_type, "properties": props}

    # Spine
    spine_el = next((e for e in opf_root.iter() if e.tag.endswith("spine")), None)
    spine_paths = []
    ncx_id = spine_el.attrib.get("toc") if spine_el is not None else None
    
    if spine_el is not None:
        for itemref in list(spine_el):
            if itemref.tag.endswith("itemref"):
                idref = itemref.attrib.get("idref")
                if idref in manifest:
                    m = manifest[idref]
                    if "html" in (m.get("media_type") or "").lower() and m["path"] in file_list:
                        spine_paths.append(m["path"])

    # Try Parse TOC
    nav_path = next((m["path"] for m in manifest.values() if "nav" in (m.get("properties") or "").split()), None)
    ncx_path = manifest[ncx_id]["path"] if ncx_id and ncx_id in manifest else None
    
    toc_entries = []
    if nav_path and nav_path in file_list: toc_entries = parse_nav_toc(z, nav_path)
    elif ncx_path and ncx_path in file_list: toc_entries = parse_ncx_toc(z, ncx_path)
    
    # Map TOC to chapters
    chapter_titles = [f"第 {i+1} 章" for i in range(len(spine_paths))]
    path_to_idx = {p: i for i, p in enumerate(spine_paths)}
    
    for e in toc_entries:
        if e["path"] in path_to_idx:
            idx = path_to_idx[e["path"]]
            chapter_titles[idx] = e["title"]

    return {
        "title": title,
        "spine_paths": spine_paths,
        "chapter_titles": chapter_titles,
        "mime_by_path": {m["path"]: m.get("media_type") for m in manifest.values()},
        "file_list": file_list
    }

def extract_chapter_blocks(epub_bytes: bytes, book: dict, chapter_idx: int, embed_images: bool):
    z = zipfile.ZipFile(io.BytesIO(epub_bytes))
    path = book["spine_paths"][chapter_idx]
    html = decode_bytes(z.read(path))
    soup = soup_html(html)
    body = soup.body or soup
    for s in body.find_all("script"): s.decompose()
    
    if embed_images:
        base_dir = posixpath.dirname(path)
        for img in body.find_all("img"):
            src = img.get("src")
            target, _ = resolve_href(base_dir, src)
            if target in book["file_list"]:
                raw = z.read(target)
                mime = book["mime_by_path"].get(target, "image/jpeg")
                b64 = base64.b64encode(raw).decode("ascii")
                img["src"] = f"data:{mime};base64,{b64}"

    blocks = []
    for el in body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div"]):
        text = el.get_text(" ", strip=True)
        # 只取有内容的块，且不取仅仅包含图片的块(除非有alt)
        if len(text) > 1:
            if el.find(["p", "div", "li", "h1", "h2"]): continue # 简单防嵌套
            blocks.append({
                "text": text,
                "html": str(el), # 保留 HTML 标签
                "tag": el.name
            })
            
    if not blocks:
        txt = body.get_text("\n", strip=True)
        parts = [p for p in txt.split('\n') if p.strip()]
        for p in parts:
            blocks.append({"text": p, "html": f"<p>{p}</p>", "tag": "p"})
            
    return blocks

# ============================================================
# AI 核心
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
def gemini_translate(text: str, style: str) -> str:
    if not GEMINI_AVAILABLE: return "Error: 库未安装"
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = f"Translate the following text to Simplified Chinese.\nStyle: {style}\n\n{text}"
    try:
        resp = client.models.generate_content(model=TEXT_MODEL_ID, contents=prompt)
        return resp.text.strip()
    except Exception as e:
        return f"翻译出错: {str(e)}"

@st.cache_data(show_spinner=False)
def gemini_tts(text: str, voice: str) -> bytes:
    if not GEMINI_AVAILABLE: return b""
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=TTS_MODEL_ID,
            contents=text[:4000],
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

@st.cache_data(show_spinner=False)
def translate_en_to_zh_openai(text: str) -> str:
    if not OPENAI_AVAILABLE: return "Error: openai not installed."
    client = OpenAI()
    prompt = f"Translate to Simplified Chinese. Keep paragraphs.\n\n{text}"
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI Error: {e}"

# ============================================================
# UI 渲染 (修复缩进问题)
# ============================================================
def render_clickable_blocks(blocks, current_play_idx, theme):
    text_color = "#e6e6e6" if theme == "Dark" else "#111111"
    hover_bg = "rgba(255,255,255,0.1)" if theme == "Dark" else "rgba(0,0,0,0.05)"
    active_bg = "rgba(255, 200, 100, 0.3)" if theme == "Dark" else "rgba(255, 230, 0, 0.4)"
    
    # CSS: 定义链接样式，使其看起来像块级元素
    css = f"""
    <style>
    .block-link {{
        display: block !important;
        text-decoration: none !important;
        color: {text_color} !important;
        padding: 6px 10px;
        margin-bottom: 8px;
        border-radius: 4px;
        border-left: 3px solid transparent;
        transition: background 0.1s;
    }}
    .block-link:hover {{
        background-color: {hover_bg};
        border-left: 3px solid #888;
    }}
    .block-active {{
        background-color: {active_bg} !important;
        border-left: 3px solid #f60 !important;
    }}
    /* 让原有的 p 标签样式不干扰链接 */
    .block-link p, .block-link div, .block-link h1, .block-link h2 {{
        margin: 0; padding: 0; color: inherit;
    }}
    </style>
    """
    
    html_list = [css]
    
    for i, block in enumerate(blocks):
        active_class = "block-active" if i == current_play_idx else ""
        content = block["html"].strip()
        # 核心修复：单行拼接，无缩进，避免 Markdown 识别为代码块
        link = f'<a href="?play_idx={i}" target="_self" class="block-link {active_class}" id="blk-{i}">{content}</a>'
        html_list.append(link)
        
    return "".join(html_list)

# ============================================================
# Main
# ============================================================
def main():
    query = st.query_params
    play_idx_str = query.get("play_idx", None)
    current_play_idx = int(play_idx_str) if play_idx_str is not None else None

    with st.sidebar:
        st.header("📖 EPUB AI Reader")
        uploaded = st.file_uploader("上传文件", type=["epub"])
        
        st.divider()
        st.subheader("🔊 朗读 (Gemini)")
        voice = st.selectbox("声音", ["Kore", "Zephyr", "Puck", "Charon", "Fenrir"], index=0)
        speed = st.slider("语速", 0.5, 2.0, 1.25, 0.1)
        auto_next = st.checkbox("自动连播", value=True)
        
        st.divider()
        st.subheader("🛠️ 设置")
        view_mode = st.radio("模式", ["点击朗读", "对照翻译"], index=0)
        if view_mode == "对照翻译":
            trans_engine = st.radio("翻译引擎", ["Gemini", "OpenAI"], horizontal=True)
        
        theme = st.radio("主题", ["Light", "Dark"], index=1, horizontal=True)

    if not uploaded:
        st.info("👈 请在左侧上传 EPUB 文件。")
        st.stop()
        
    # 解析
    epub_bytes = uploaded.getvalue()
    book_hash = hashlib.sha256(epub_bytes).hexdigest()
    
    if "book_hash" not in st.session_state or st.session_state.book_hash != book_hash:
        try:
            st.session_state.book = parse_epub(epub_bytes)
            st.session_state.book_hash = book_hash
            st.session_state.chapter_idx = 0
            st.query_params.clear()
        except Exception as e:
            st.error(f"解析失败: {e}")
            st.stop()
            
    book = st.session_state.book
    
    # 导航
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("⬅️ 上一章", use_container_width=True):
            st.session_state.chapter_idx = max(0, st.session_state.chapter_idx - 1)
            st.query_params.clear()
            st.rerun()
    with col2:
        chap_list = book["chapter_titles"]
        curr_idx = st.session_state.chapter_idx
        # 确保索引不越界（防御性编程）
        if curr_idx >= len(chap_list): curr_idx = 0
        
        new_chap = st.selectbox("章节", range(len(chap_list)), 
                                index=curr_idx, 
                                format_func=lambda i: chap_list[i], 
                                label_visibility="collapsed")
        if new_chap != curr_idx:
            st.session_state.chapter_idx = new_chap
            st.query_params.clear()
            st.rerun()
    with col3:
        if st.button("下一章 ➡️", use_container_width=True):
            st.session_state.chapter_idx = min(len(chap_list)-1, st.session_state.chapter_idx + 1)
            st.query_params.clear()
            st.rerun()

    # 内容
    blocks = extract_chapter_blocks(epub_bytes, book, st.session_state.chapter_idx, embed_images=True)
    
    # 音频播放
    if current_play_idx is not None and 0 <= current_play_idx < len(blocks):
        target_text = blocks[current_play_idx]["text"]
        if target_text.strip():
            wav = gemini_tts(target_text, voice)
            if wav:
                b64 = base64.b64encode(wav).decode()
                next_js = ""
                if auto_next and current_play_idx + 1 < len(blocks):
                    next_url = f"?play_idx={current_play_idx + 1}"
                    next_js = f"""
                    aud.onended = function() {{
                        window.parent.location.search = "{next_url}";
                    }};
                    """
                
                # 播放器组件
                player_html = f"""
                <div style="position:fixed; bottom:20px; left:50%; transform:translateX(-50%); 
                            background:#222; padding:8px 15px; border-radius:30px; 
                            box-shadow:0 5px 20px rgba(0,0,0,0.5); z-index:99999; display:flex; align-items:center; gap:10px;">
                    <span style="color:#fff; font-size:14px; white-space:nowrap;">▶ ({current_play_idx + 1}/{len(blocks)})</span>
                    <audio id="player" controls autoplay style="height:30px;">
                        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    </audio>
                </div>
                <script>
                    var aud = document.getElementById("player");
                    aud.playbackRate = {speed};
                    {next_js}
                </script>
                """
                components.html(player_html, height=0)

    # 视图渲染
    if view_mode == "点击朗读":
        st.caption("提示：点击段落即可朗读。")
        html_content = render_clickable_blocks(blocks, current_play_idx, theme)
        bg = "#0e1117" if theme == "Dark" else "#ffffff"
        
        # 修复：移除 f-string 内部的缩进，防止被当做代码块
        st.markdown(f'<div style="background-color:{bg}; padding:20px; border-radius:8px; max-width:800px; margin:0 auto;">{html_content}</div>', unsafe_allow_html=True)
        
        # 滚动 JS
        if current_play_idx is not None:
             components.html(f"<script>setTimeout(function(){{var el=window.parent.document.getElementById('blk-{current_play_idx}');if(el)el.scrollIntoView({{behavior:'smooth',block:'center'}});}},500);</script>", height=0)

    else: # 翻译模式
        st.caption("左侧点击段落可朗读，右侧点击按钮翻译。")
        per_page = 10
        total_pages = max(1, (len(blocks) + per_page - 1) // per_page)
        page = st.number_input("页码", 1, total_pages, 1) - 1
        
        chunk = blocks[page*per_page : (page+1)*per_page]
        
        colL, colR = st.columns(2)
        with colL:
            # 这里的左侧我们也用 render_clickable_blocks 渲染，支持点击朗读
            # 但是要调整 index 偏移
            # 我们需要造一个临时的 blocks 列表，但是 ID 必须对应全局 index
            # render_clickable_blocks 内部是 enumerate(blocks)，所以我们不能直接传 chunk
            # 而是要手动生成 HTML
            
            # 手动生成左侧 HTML
            left_html_parts = []
            # 引入样式
            left_html_parts.append(render_clickable_blocks([], None, theme)) #只拿css
            
            for i, block in enumerate(chunk):
                real_idx = page * per_page + i
                active_class = "block-active" if real_idx == current_play_idx else ""
                content = block["html"].strip()
                link = f'<a href="?play_idx={real_idx}" target="_self" class="block-link {active_class}" id="blk-{real_idx}">{content}</a>'
                left_html_parts.append(link)
            
            full_left = "".join(left_html_parts)
            bg = "#0e1117" if theme == "Dark" else "#ffffff"
            st.markdown(f'<div style="background-color:{bg}; padding:10px;">{full_left}</div>', unsafe_allow_html=True)

        with colR:
            src_text = "\n\n".join([b["text"] for b in chunk])
            if st.button("翻译当前页", key=f"btn_trans_{page}", use_container_width=True):
                with st.spinner("翻译中..."):
                    if trans_engine == "Gemini":
                        res = gemini_translate(src_text, "文学流畅")
                    else:
                        res = translate_en_to_zh_openai(src_text)
                st.session_state[f"trans_{page}"] = res
            
            if f"trans_{page}" in st.session_state:
                st.info(st.session_state[f"trans_{page}"])
            else:
                st.text_area("原文预览", src_text, height=600, disabled=True)

if __name__ == "__main__":
    main()
