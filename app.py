import os
import asyncio
import base64
import hashlib
import io
import re
import threading
import wave
import zipfile
import xml.etree.ElementTree as ET
import urllib.parse

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

# 固定模型配置，防止混用
TTS_MODEL_ID = "gemini-2.5-flash-preview-tts"  # 专用语音模型
TEXT_MODEL_ID = "gemini-2.0-flash"             # 专用文本/翻译模型

# ============================================================
# 辅助函数
# ============================================================
def normalize_zip_path(path: str) -> str:
    path = (path or "").replace("\\", "/")
    path = re.sub(r"^\./", "", path)
    return posixpath.normpath(path)

import posixpath

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

def guess_mime(path: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "gif": "image/gif", "svg": "image/svg+xml", "webp": "image/webp"
    }.get(ext, "application/octet-stream")

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
# EPUB 解析核心
# ============================================================
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
                if iid and href:
                    path = normalize_zip_path(posixpath.join(opf_dir, href))
                    manifest[iid] = {"href": href, "path": path, "media_type": item.attrib.get("media-type", "")}

    # Spine
    spine_el = next((e for e in opf_root.iter() if e.tag.endswith("spine")), None)
    spine_paths = []
    if spine_el is not None:
        for itemref in list(spine_el):
            if itemref.tag.endswith("itemref"):
                idref = itemref.attrib.get("idref")
                if idref in manifest:
                    m = manifest[idref]
                    if "html" in (m.get("media_type") or "").lower() and m["path"] in file_list:
                        spine_paths.append(m["path"])

    # Simple TOC
    chapter_titles = [f"Chapter {i+1}" for i in range(len(spine_paths))] # 简化处理，避免解析复杂toc导致报错

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
    
    # 图片嵌入
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
    # 提取顶层块级元素
    for el in body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div"]):
        # 简单防重：如果父级已经是我们提取过的块，跳过子级
        # 但这里为了简单，只提取有实际文本的叶子或近叶子节点
        text = el.get_text(" ", strip=True)
        if len(text) > 1:
            # 检查这个元素是否包含其他块级元素，如果包含则跳过（避免重复）
            if el.find(["p", "div", "li"]):
                continue
            
            blocks.append({
                "text": text,
                "html": str(el), # 原始 HTML
                "tag": el.name
            })
            
    # 如果没提取到，兜底
    if not blocks:
        txt = body.get_text("\n", strip=True)
        parts = [p for p in txt.split('\n') if p.strip()]
        for p in parts:
            blocks.append({"text": p, "html": f"<p>{p}</p>", "tag": "p"})
            
    return blocks

# ============================================================
# Gemini API 调用
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
    """必须使用 gemini-2.0-flash (文本模型)"""
    if not GEMINI_AVAILABLE: return "Error: 库未安装"
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    prompt = f"Translate the following text to Simplified Chinese.\nStyle: {style}\n\n{text}"
    try:
        # 强制使用文本模型
        resp = client.models.generate_content(model=TEXT_MODEL_ID, contents=prompt)
        return resp.text.strip()
    except Exception as e:
        return f"翻译出错: {str(e)}"

@st.cache_data(show_spinner=False)
def gemini_tts(text: str, voice: str) -> bytes:
    """必须使用 gemini-2.5-flash-preview-tts (语音模型)"""
    if not GEMINI_AVAILABLE: return b""
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    safe_text = text[:4000] # 长度保护
    
    try:
        resp = client.models.generate_content(
            model=TTS_MODEL_ID,
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
# UI 渲染 (关键修改：使用原生 Link 触发)
# ============================================================
def render_clickable_blocks(blocks, current_play_idx, theme):
    """
    使用 st.markdown 渲染 HTML。
    核心技巧：将文本包裹在 <a href="?play_idx=X" target="_self"> 中。
    点击链接 = 刷新页面 = 触发 Python 逻辑。
    """
    
    # 颜色配置
    text_color = "#e6e6e6" if theme == "Dark" else "#111111"
    link_color = text_color # 让链接看起来像普通文字
    hover_bg = "rgba(255,255,255,0.1)" if theme == "Dark" else "rgba(0,0,0,0.05)"
    active_bg = "rgba(255, 200, 100, 0.3)" if theme == "Dark" else "rgba(255, 230, 0, 0.4)"
    
    html_parts = []
    
    # CSS 样式
    html_parts.append(f"""
    <style>
    .block-link {{
        display: block;
        text-decoration: none;
        color: {text_color} !important;
        padding: 6px 10px;
        margin-bottom: 8px;
        border-radius: 4px;
        transition: background 0.15s;
        border-left: 3px solid transparent;
    }}
    .block-link:hover {{
        background-color: {hover_bg};
        border-left: 3px solid #888;
        text-decoration: none !important;
    }}
    .block-active {{
        background-color: {active_bg} !important;
        border-left: 3px solid #f60;
    }}
    .reader-img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; }}
    </style>
    """)
    
    for i, block in enumerate(blocks):
        is_active = (i == current_play_idx)
        active_class = "block-active" if is_active else ""
        
        # 提取 HTML 内容 (去除原有的 p 标签，因为我们要用 a 标签包裹)
        content = block["html"]
        # 简单的清理，防止嵌套非法 (a 里面不能套 div/p 在某些 DOCTYPE 下，但在 HTML5 流式内容中通常浏览器能容忍)
        # 为了安全，我们只取内容文本或者 innerHTML
        # 这里直接用 block['text'] 最安全，如果需要保留加粗等格式，需要更精细的处理。
        # 为了保留原书格式（粗体/斜体），我们直接包裹 block["html"]。
        # 浏览器通常允许 <a style="display:block">...</a>
        
        # 构造链接。注意 target="_self" 是关键，强制在当前页刷新
        link = f"""
        <a href="?play_idx={i}" target="_self" class="block-link {active_class}" id="blk-{i}">
            {content}
        </a>
        """
        html_parts.append(link)
        
    return "\n".join(html_parts)

# ============================================================
# 主程序
# ============================================================
def main():
    # 1. 获取 URL 参数 (Streamlit 1.30+ 用 st.query_params)
    query = st.query_params
    play_idx_str = query.get("play_idx", None)
    current_play_idx = int(play_idx_str) if play_idx_str is not None else None

    with st.sidebar:
        st.header("📖 1. 文件")
        uploaded = st.file_uploader("EPUB 上传", type=["epub"])
        
        st.divider()
        st.header("🔊 2. Gemini 朗读")
        if not GEMINI_AVAILABLE: st.error("请安装 google-genai")
        
        voice = st.selectbox("声音", ["Kore", "Zephyr", "Puck", "Charon", "Fenrir"], index=0)
        speed = st.slider("语速", 0.5, 2.0, 1.25, 0.1)
        auto_next = st.checkbox("自动连播", value=True)
        
        st.divider()
        st.header("👁️ 3. 显示")
        view_mode = st.radio("模式", ["点击朗读", "对照翻译"], index=0)
        theme = st.radio("主题", ["Light", "Dark"], index=1, horizontal=True)

    if not uploaded:
        st.info("请先上传 EPUB。")
        st.stop()
        
    # 解析文件
    epub_bytes = uploaded.getvalue()
    book_hash = hashlib.sha256(epub_bytes).hexdigest()
    
    if "book_hash" not in st.session_state or st.session_state.book_hash != book_hash:
        try:
            st.session_state.book = parse_epub(epub_bytes)
            st.session_state.book_hash = book_hash
            st.session_state.chapter_idx = 0
            st.query_params.clear() # 重置参数
        except Exception as e:
            st.error(f"解析失败: {e}")
            st.stop()
            
    book = st.session_state.book
    
    # 章节导航
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("⬅️", use_container_width=True):
            st.session_state.chapter_idx = max(0, st.session_state.chapter_idx - 1)
            st.query_params.clear()
            st.rerun()
    with col2:
        chap_list = book["chapter_titles"]
        new_chap = st.selectbox("当前章节", range(len(chap_list)), 
                                index=st.session_state.chapter_idx, 
                                format_func=lambda i: chap_list[i], 
                                label_visibility="collapsed")
        if new_chap != st.session_state.chapter_idx:
            st.session_state.chapter_idx = new_chap
            st.query_params.clear()
            st.rerun()
    with col3:
        if st.button("➡️", use_container_width=True):
            st.session_state.chapter_idx = min(len(chap_list)-1, st.session_state.chapter_idx + 1)
            st.query_params.clear()
            st.rerun()

    # 提取当前章节内容
    blocks = extract_chapter_blocks(epub_bytes, book, st.session_state.chapter_idx, embed_images=True)
    
    # --------------------------------------------------------
    # 音频处理逻辑 (如果在播放状态)
    # --------------------------------------------------------
    if current_play_idx is not None and 0 <= current_play_idx < len(blocks):
        # 自动滚动到当前播放位置 (通过 HTML anchor)
        target_block = blocks[current_play_idx]
        text_to_speak = target_block["text"]
        
        # 生成音频 (使用专门的语音模型)
        if text_to_speak.strip():
            # 显示一个固定的播放栏在顶部
            wav_bytes = gemini_tts(text_to_speak, voice)
            
            if wav_bytes:
                b64_audio = base64.b64encode(wav_bytes).decode()
                
                # 下一段的 URL
                next_url = ""
                if auto_next and current_play_idx + 1 < len(blocks):
                    # 构造下一段的 query string
                    # 注意：这里需要全路径或者相对路径，Streamlit 重载通常在根路径
                    next_url = f"?play_idx={current_play_idx + 1}"
                
                # 播放器 HTML (原生 Audio + JS 监听结束跳转)
                # 放在 st.markdown 中，位置固定在底部
                player_html = f"""
                <div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); 
                            background: #222; padding: 10px 20px; border-radius: 30px; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.5); z-index: 99999; display: flex; align-items: center; gap: 10px;">
                    <span style="color: #fff; font-size: 14px; font-weight: bold;">
                        ▶ 正在朗读 ({current_play_idx + 1}/{len(blocks)})
                    </span>
                    <audio id="global-player" controls autoplay style="height: 30px;">
                        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
                    </audio>
                </div>
                <script>
                    var aud = document.getElementById("global-player");
                    if (aud) {{
                        aud.playbackRate = {speed};
                        aud.onended = function() {{
                            if ("{next_url}" !== "") {{
                                window.parent.location.search = "{next_url}";
                            }}
                        }};
                    }}
                </script>
                """
                components.html(player_html, height=0) # 0高度不可见iframe，但其中的fixed元素可见

    # --------------------------------------------------------
    # 主视图渲染
    # --------------------------------------------------------
    
    if view_mode == "点击朗读":
        # 使用 Markdown + HTML 渲染
        # 这是修复点击卡顿的关键：直接把文字变成链接
        st.caption("提示：点击任意段落，Gemini 将从该处开始朗读。")
        
        html_content = render_clickable_blocks(blocks, current_play_idx, theme)
        
        # 容器背景色
        bg_color = "#0e1117" if theme == "Dark" else "#ffffff"
        
        st.markdown(f"""
        <div style="background-color:{bg_color}; padding: 20px; border-radius: 10px; max-width: 800px; margin: 0 auto;">
            {html_content}
        </div>
        """, unsafe_allow_html=True)
        
        # 尝试 JS 滚动 (如果刚加载页面)
        if current_play_idx is not None:
             # 这段 JS 尝试把视图滚动到 id="blk-{idx}" 的元素
             components.html(f"""
             <script>
                setTimeout(function(){{
                    var el = window.parent.document.getElementById('blk-{current_play_idx}');
                    if(el) el.scrollIntoView({{behavior: "smooth", block: "center"}});
                }}, 500);
             </script>
             """, height=0)

    else: # 对照翻译模式
        st.caption("👈 点击左侧 **翻译当前页** 按钮查看译文。")
        
        # 分页逻辑
        per_page = 10
        total_pages = (len(blocks) + per_page - 1) // per_page
        if total_pages == 0: total_pages = 1
        
        page = st.number_input("页码", 1, total_pages, 1) - 1
        
        start = page * per_page
        end = start + per_page
        page_blocks = blocks[start:end]
        
        colL, colR = st.columns(2)
        
        src_text = "\n\n".join([b["text"] for b in page_blocks])
        src_html = "\n".join([b["html"] for b in page_blocks])
        
        with colL:
            st.markdown("### 原文")
            st.markdown(f"<div style='opacity:0.9'>{src_html}</div>", unsafe_allow_html=True)
            
        with colR:
            st.markdown("### 译文")
            if st.button("翻译当前页 (Gemini)", use_container_width=True, key=f"trans_{page}"):
                with st.spinner("Gemini 正在翻译..."):
                    # 此时肯定调用的是 TEXT_MODEL_ID
                    res = gemini_translate(src_text, style="流畅、文学")
                    if "Error" in res: st.error(res)
                    else: st.success("翻译完成")
                    st.session_state[f"trans_res_{page}"] = res
            
            if f"trans_res_{page}" in st.session_state:
                st.markdown(st.session_state[f"trans_res_{page}"])

if __name__ == "__main__":
    main()
