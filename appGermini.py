import os
import base64
import hashlib
import io
import re
import wave
import zipfile
import time
import uuid
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
def get_secret(*names: str) -> str:
    for n in names:
        if hasattr(st, "secrets") and n in st.secrets and str(st.secrets.get(n, "")).strip():
            return str(st.secrets.get(n)).strip()
        if str(os.environ.get(n, "")).strip():
            return str(os.environ.get(n)).strip()
    return ""

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

# ============================================================
# EPUB 解析
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
    
    metadata = next((e for e in opf_root.iter() if e.tag.endswith("metadata")), None)
    title = first_child_text(metadata, "title") or "Untitled"
    
    manifest = {}
    manifest_el = next((e for e in opf_root.iter() if e.tag.endswith("manifest")), None)
    if manifest_el is not None:
        for item in list(manifest_el):
            if item.tag.endswith("item"):
                iid, href = item.attrib.get("id"), item.attrib.get("href")
                media_type = item.attrib.get("media-type", "")
                if iid and href:
                    path = normalize_zip_path(posixpath.join(opf_dir, href))
                    manifest[iid] = {"href": href, "path": path, "media_type": media_type}

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

    chapter_titles = [f"Section {i+1}" for i in range(len(spine_paths))]
    
    return {
        "title": title,
        "spine_paths": spine_paths,
        "chapter_titles": chapter_titles,
        "mime_by_path": {m["path"]: m.get("media_type") for m in manifest.values()},
        "file_list": file_list
    }

def extract_chapter_content(epub_bytes: bytes, book: dict, chapter_idx: int):
    z = zipfile.ZipFile(io.BytesIO(epub_bytes))
    path = book["spine_paths"][chapter_idx]
    html = decode_bytes(z.read(path))
    soup = soup_html(html)
    body = soup.body or soup
    for s in body.find_all("script"): s.decompose()
    
    raw_blocks = []
    for el in body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div"]):
        text = el.get_text(" ", strip=True)
        if len(text) > 1:
            if el.find(["p", "div", "li", "h1"]): continue 
            raw_blocks.append({
                "text": text,
                "html": str(el)
            })
            
    if not raw_blocks:
        txt = body.get_text("\n", strip=True)
        parts = [p for p in txt.split('\n') if p.strip()]
        for p in parts:
            raw_blocks.append({"text": p, "html": f"<p>{p}</p>"})
            
    return raw_blocks

def merge_blocks_into_chunks(raw_blocks, max_chars=800):
    chunks = []
    if not raw_blocks: return chunks
    
    current_chunk = {"text": "", "html": ""}
    
    for block in raw_blocks:
        if len(current_chunk["text"]) + len(block["text"]) < max_chars:
            current_chunk["text"] += "\n" + block["text"]
            current_chunk["html"] += block["html"]
        else:
            if current_chunk["text"]:
                chunks.append(current_chunk)
            current_chunk = {"text": block["text"], "html": block["html"]}
            
    if current_chunk["text"]:
        chunks.append(current_chunk)
        
    return chunks

# ============================================================
# AI 逻辑 
# ============================================================
def pcm16_to_wav_bytes(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()

def gemini_tts(text: str, voice: str):
    if not GEMINI_AVAILABLE: return None, "未安装 google-genai 库"
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key: return None, "未设置 GEMINI_API_KEY"
    
    client = genai.Client(api_key=api_key)
    safe_text = text[:4500] 
    
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
        return pcm16_to_wav_bytes(pcm), None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def gemini_translate(text: str) -> str:
    if not GEMINI_AVAILABLE: return "Error: 库未安装"
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = f"Translate the following text to Simplified Chinese.\nKeep the tone natural.\n\n{text}"
    try:
        resp = client.models.generate_content(model=TEXT_MODEL_ID, contents=prompt)
        return resp.text.strip()
    except Exception as e:
        return f"翻译出错: {str(e)}"

@st.cache_data(show_spinner=False)
def openai_translate(text: str) -> str:
    if not OPENAI_AVAILABLE: return "Error: openai not installed"
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Translate to Chinese:\n\n{text}"}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI Error: {e}"

# ============================================================
# Main UI
# ============================================================
def main():
    # Session State
    if "playing_idx" not in st.session_state: st.session_state.playing_idx = None
    if "audio_data" not in st.session_state: st.session_state.audio_data = None
    if "auto_next_trigger" not in st.session_state: st.session_state.auto_next_trigger = False
    # 增加一个 unique_id 用于强制刷新 audio 标签
    if "audio_uid" not in st.session_state: st.session_state.audio_uid = str(uuid.uuid4())

    with st.sidebar:
        st.header("📖 EPUB AI Reader")
        uploaded = st.file_uploader("上传 EPUB", type=["epub"])
        
        st.divider()
        st.subheader("🔊 朗读设置")
        if not GEMINI_AVAILABLE: st.error("⚠️ 需要安装 google-genai")
        voice = st.selectbox("声音", ["Kore", "Zephyr", "Puck", "Charon", "Fenrir"], index=0)
        speed = st.slider("语速", 0.5, 2.0, 1.25, 0.1)
        chunk_size = st.slider("分段长度", 300, 2000, 800, 100)
        auto_play = st.checkbox("自动连播", value=True)

        st.divider()
        st.subheader("🛠️ 视图设置")
        view_mode = st.radio("模式", ["阅读模式", "对照翻译"], index=0)
        if view_mode == "对照翻译":
            trans_engine = st.selectbox("翻译引擎", ["Gemini", "OpenAI"])
        theme = st.radio("主题", ["Light", "Dark"], index=1, horizontal=True)
        
        # 隐藏的触发按钮（放在侧边栏最底部）
        def trigger_next():
            st.session_state.auto_next_trigger = True
            # 关键：清除旧音频，防止 Rerun 时显示旧的播放器
            st.session_state.audio_data = None
        
        st.markdown("---")
        # 这个按钮是 JS 点击的目标
        st.button("NEXT_TRIGGER", key="hidden_next_btn", on_click=trigger_next, type="secondary")
        # 视觉隐藏
        st.markdown("<style>div.stButton > button:contains('NEXT_TRIGGER') { display: none; }</style>", unsafe_allow_html=True)

    if not uploaded:
        st.info("👈 请在左侧上传 EPUB 文件。")
        st.stop()
        
    # 解析文件
    epub_bytes = uploaded.getvalue()
    book_hash = hashlib.sha256(epub_bytes).hexdigest()
    
    if "book_hash" not in st.session_state or st.session_state.book_hash != book_hash:
        try:
            st.session_state.book = parse_epub(epub_bytes)
            st.session_state.book_hash = book_hash
            st.session_state.chapter_idx = 0
            st.session_state.playing_idx = None
            st.session_state.audio_data = None
        except Exception as e:
            st.error(f"解析失败: {e}")
            st.stop()
            
    book = st.session_state.book
    
    # 章节导航
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if st.button("⬅️ 上一章", use_container_width=True):
            st.session_state.chapter_idx = max(0, st.session_state.chapter_idx - 1)
            st.session_state.playing_idx = None
            st.rerun()
    with c3:
        if st.button("下一章 ➡️", use_container_width=True):
            st.session_state.chapter_idx = min(len(book["chapter_titles"])-1, st.session_state.chapter_idx + 1)
            st.session_state.playing_idx = None
            st.rerun()
            
    # 内容处理
    raw_blocks = extract_chapter_content(epub_bytes, book, st.session_state.chapter_idx)
    chunks = merge_blocks_into_chunks(raw_blocks, max_chars=chunk_size)
    
    if not chunks:
        st.warning("本章内容为空。")
        st.stop()

    # ============================================================
    # 逻辑核心：处理自动连播触发 & 音频生成
    # ============================================================
    
    # 1. 如果是由 JS 触发了自动下一段
    if st.session_state.auto_next_trigger:
        next_idx = st.session_state.playing_idx + 1 if st.session_state.playing_idx is not None else 0
        if next_idx < len(chunks):
            st.session_state.playing_idx = next_idx
            st.session_state.audio_data = None # 确保为空，强制进入生成逻辑
            st.session_state.auto_next_trigger = False # 重置触发器
        else:
            st.toast("本章播放结束")
            st.session_state.auto_next_trigger = False

    # 2. 生成音频 (如果需要)
    # 只有当 playing_idx 有值，但 audio_data 为空时，才生成
    if st.session_state.playing_idx is not None and st.session_state.audio_data is None:
        idx = st.session_state.playing_idx
        text = chunks[idx]["text"]
        
        # 在顶部显示明显的加载状态
        with st.spinner(f"正在生成第 {idx+1}/{len(chunks)} 段音频 (Gemini)..."):
            wav, err = gemini_tts(text, voice)
            
        if err:
            st.error(f"生成失败: {err}")
            st.session_state.playing_idx = None
        else:
            st.session_state.audio_data = wav
            # 关键修复：生成一个新的随机 ID，强迫浏览器认为这是个新音频，解决循环播放旧音频的问题
            st.session_state.audio_uid = str(uuid.uuid4())
            st.rerun() # 刷新页面显示播放器

    # ============================================================
    # 播放器组件 (置顶 Fixed)
    # ============================================================
    if st.session_state.audio_data and st.session_state.playing_idx is not None:
        b64 = base64.b64encode(st.session_state.audio_data).decode()
        idx = st.session_state.playing_idx
        # 自动连播 JS
        on_end_js = ""
        if auto_play and idx + 1 < len(chunks):
            # 寻找 innerText 为 NEXT_TRIGGER 的按钮并点击
            on_end_js = """
            aud.onended = function() {
                const btns = window.parent.document.querySelectorAll('button');
                for (let btn of btns) {
                    if (btn.innerText === "NEXT_TRIGGER") {
                        btn.click();
                        break;
                    }
                }
            };
            """
        
        # 顶部固定播放栏 HTML
        # 使用 audio_uid 作为 ID 的一部分，解决缓存问题
        player_html = f"""
        <div style="
            position: fixed; top: 0; left: 0; right: 0; 
            background: #1e1e1e; border-bottom: 1px solid #444; 
            padding: 10px 20px; z-index: 999999; 
            display: flex; align-items: center; justify-content: center; gap: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        ">
            <span style="color: #fff; font-weight: bold; white-space: nowrap;">
                🎧 {idx+1} / {len(chunks)}
            </span>
            <audio id="audio_{st.session_state.audio_uid}" controls autoplay style="width: 100%; max-width: 600px; height: 35px;">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            <div style="color: #888; font-size: 12px; white-space: nowrap;">
                {'自动下一段' if auto_play else '单段播放'}
            </div>
        </div>
        <script>
            var aud = document.getElementById("audio_{st.session_state.audio_uid}");
            if(aud) {{
                aud.playbackRate = {speed};
                {on_end_js}
            }}
        </script>
        """
        # 这里的 height=60 是给 iframe 预留的高度，确保它能显示出来
        # 因为 CSS 设置了 fixed top，所以它会脱离文档流浮在上面
        components.html(player_html, height=60)
        # 为了不让内容被遮挡，加一个垫片
        st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)


    # ============================================================
    # 内容渲染 (阅读模式 vs 翻译模式)
    # ============================================================
    
    # 样式优化
    st.markdown("""
    <style>
    .chunk-box { padding: 12px; border-radius: 8px; margin-bottom: 12px; line-height: 1.6; }
    .trans-box { background: rgba(0,0,0,0.05); padding: 10px; border-radius: 8px; border-left: 3px solid #00aa00; margin-top: 5px; }
    div.stButton > button { width: 100%; height: auto; min-height: 50px; white-space: normal; }
    </style>
    """, unsafe_allow_html=True)

    # 渲染循环
    for i, chunk in enumerate(chunks):
        is_playing = (i == st.session_state.playing_idx)
        
        # 1. 阅读模式 (左按钮 右文本)
        if view_mode == "阅读模式":
            col_btn, col_txt = st.columns([1, 10])
            with col_btn:
                btn_type = "primary" if is_playing else "secondary"
                label = "🔊" if is_playing else f"{i+1}"
                if st.button(label, key=f"play_{i}", type=btn_type, help="点击朗读"):
                    st.session_state.playing_idx = i
                    st.session_state.audio_data = None
                    st.rerun()
            with col_txt:
                bg = "rgba(255, 200, 0, 0.15)" if is_playing else ("rgba(255,255,255,0.05)" if theme=="Dark" else "#f0f2f6")
                border = "2px solid #ffbd45" if is_playing else "1px solid transparent"
                st.markdown(f'<div class="chunk-box" style="background:{bg}; border:{border}">{chunk["html"]}</div>', unsafe_allow_html=True)

        # 2. 对照翻译模式 (左原文+播放，右译文)
        else:
            col_left, col_right = st.columns(2)
            
            # 左栏：原文 + 播放按钮
            with col_left:
                c_btn, c_txt = st.columns([1.5, 8.5])
                with c_btn:
                    btn_type = "primary" if is_playing else "secondary"
                    if st.button(f"Play {i+1}", key=f"play_tr_{i}", type=btn_type):
                        st.session_state.playing_idx = i
                        st.session_state.audio_data = None
                        st.rerun()
                with c_txt:
                    bg = "rgba(255, 200, 0, 0.15)" if is_playing else "transparent"
                    border = "2px solid #ffbd45" if is_playing else "1px solid #444"
                    st.markdown(f'<div class="chunk-box" style="background:{bg}; border:{border}">{chunk["html"]}</div>', unsafe_allow_html=True)
            
            # 右栏：翻译
            with col_right:
                # 检查是否有缓存的翻译
                cache_key = f"trans_{st.session_state.book_hash}_{st.session_state.chapter_idx}_{i}"
                if cache_key in st.session_state:
                    st.markdown(f'<div class="trans-box">{st.session_state[cache_key]}</div>', unsafe_allow_html=True)
                else:
                    if st.button("翻译", key=f"trans_btn_{i}"):
                        with st.spinner("翻译中..."):
                            if trans_engine == "Gemini":
                                res = gemini_translate(chunk["text"])
                            else:
                                res = openai_translate(chunk["text"])
                        st.session_state[cache_key] = res
                        st.rerun()

if __name__ == "__main__":
    main()
