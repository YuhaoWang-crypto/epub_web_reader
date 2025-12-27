import os
import base64
import hashlib
import io
import re
import wave
import zipfile
import xml.etree.ElementTree as ET
import posixpath
import time

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
    
    # 提取所有块级元素
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
    """
    智能合并算法：将散碎的段落合并成较大的 Section，
    直到字符数超过 max_chars。
    """
    chunks = []
    if not raw_blocks: return chunks
    
    current_chunk = {"text": "", "html": ""}
    
    for block in raw_blocks:
        # 如果当前块加上新块还没超标，就合并
        if len(current_chunk["text"]) + len(block["text"]) < max_chars:
            current_chunk["text"] += "\n" + block["text"]
            current_chunk["html"] += block["html"]
        else:
            # 超标了，先保存当前的（如果有内容）
            if current_chunk["text"]:
                chunks.append(current_chunk)
            # 开启新块
            current_chunk = {"text": block["text"], "html": block["html"]}
            
    # 别忘了最后一个
    if current_chunk["text"]:
        chunks.append(current_chunk)
        
    return chunks

# ============================================================
# AI 逻辑 (带详细错误捕捉)
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
    """返回 (wav_bytes, error_msg)"""
    if not GEMINI_AVAILABLE: return None, "未安装 google-genai 库"
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key: return None, "未设置 GEMINI_API_KEY"
    
    client = genai.Client(api_key=api_key)
    
    # 保护性截断
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

# ============================================================
# Main UI
# ============================================================
def main():
    # Session State 初始化
    if "playing_idx" not in st.session_state: st.session_state.playing_idx = None
    if "audio_data" not in st.session_state: st.session_state.audio_data = None
    if "auto_next_trigger" not in st.session_state: st.session_state.auto_next_trigger = False

    with st.sidebar:
        st.header("📖 EPUB AI Reader")
        uploaded = st.file_uploader("上传 EPUB", type=["epub"])
        
        st.divider()
        st.subheader("🔊 朗读设置")
        if not GEMINI_AVAILABLE: st.error("⚠️ 需要安装 google-genai")
        voice = st.selectbox("声音", ["Kore", "Zephyr", "Puck", "Charon", "Fenrir"], index=0)
        speed = st.slider("语速", 0.5, 2.0, 1.25, 0.1)
        
        # 调整合并粒度
        chunk_size = st.slider("分段长度 (字符)", 300, 2000, 800, 100, help="把多个短段落合并成一个大段朗读，减少点击次数。")
        auto_play = st.checkbox("自动连播 (实验性)", value=True, help="一段播完自动尝试播放下一段")

        st.divider()
        st.subheader("🛠️ 外观")
        theme = st.radio("主题", ["Light", "Dark"], index=1, horizontal=True)

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
            
    # 提取内容并合并
    raw_blocks = extract_chapter_content(epub_bytes, book, st.session_state.chapter_idx)
    chunks = merge_blocks_into_chunks(raw_blocks, max_chars=chunk_size)
    
    if not chunks:
        st.warning("本章内容为空。")
        st.stop()

    # ------------------------------------------------------------
    # 核心交互逻辑 (渲染列表 + 处理点击)
    # ------------------------------------------------------------
    
    st.caption(f"当前章节共 {len(chunks)} 个朗读分段 (基于 {chunk_size} 字符合并)。")
    
    # 自定义样式：让按钮和文本对齐更好
    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        height: 100%;
        min-height: 60px; /* 让按钮高一点，容易点 */
        white-space: normal;
        word-wrap: break-word;
    }
    .chunk-box {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 检查是否触发了自动连播
    if st.session_state.auto_next_trigger:
        next_idx = st.session_state.playing_idx + 1
        if next_idx < len(chunks):
            # 自动触发生成逻辑
            st.session_state.playing_idx = next_idx
            st.session_state.auto_next_trigger = False # 重置触发器
            # 不需要 rerun，直接流转到下面的生成逻辑
        else:
            st.toast("本章朗读结束")
            st.session_state.auto_next_trigger = False

    # 遍历显示所有分段
    for i, chunk in enumerate(chunks):
        is_playing = (i == st.session_state.playing_idx)
        
        col_btn, col_txt = st.columns([1, 8])
        
        with col_btn:
            # 按钮状态：正在播放显示“播放中”，否则显示“▶”
            label = "🔊 播放中" if is_playing else f"▶ 第 {i+1} 段"
            btn_type = "primary" if is_playing else "secondary"
            
            # 点击按钮逻辑
            if st.button(label, key=f"chunk_{i}", type=btn_type, use_container_width=True):
                st.session_state.playing_idx = i
                st.session_state.audio_data = None # 清除旧音频，准备生成新音频
                st.rerun() # 立即刷新，触发下面的生成逻辑

        with col_txt:
            bg_color = "rgba(255, 200, 0, 0.15)" if is_playing else ("rgba(255,255,255,0.05)" if theme=="Dark" else "#f0f2f6")
            border = "2px solid #ffbd45" if is_playing else "1px solid transparent"
            
            # 显示文本内容
            st.markdown(
                f'<div class="chunk-box" style="background:{bg_color}; border:{border}">'
                f'{chunk["html"]}'
                f'</div>',
                unsafe_allow_html=True
            )

    # ------------------------------------------------------------
    # 音频生成与播放器 (固定底部)
    # ------------------------------------------------------------
    
    # 如果处于播放状态，且音频数据还没生成，则开始生成
    if st.session_state.playing_idx is not None:
        idx = st.session_state.playing_idx
        
        # 只有当 audio_data 为空时才调用 API (防止重复调用)
        if st.session_state.audio_data is None:
            target_text = chunks[idx]["text"]
            
            # 弹出一个明显的 Toast 提示
            st.toast(f"正在生成第 {idx+1} 段音频，请稍候...", icon="⏳")
            
            # 在底部显示转圈圈
            with st.spinner(f"Gemini 正在合成第 {idx+1} 段 ({len(target_text)} 字符)..."):
                wav, err = gemini_tts(target_text, voice)
                
            if err:
                st.error(f"生成失败: {err}")
                st.session_state.playing_idx = None # 重置状态
            else:
                st.session_state.audio_data = wav
                st.rerun() # 生成完毕，刷新显示播放器

        # 如果有音频数据，显示播放器
        if st.session_state.audio_data:
            b64 = base64.b64encode(st.session_state.audio_data).decode()
            
            # 自动连播逻辑：
            # 我们创建一个隐藏的 button，当 audio onended 时，JS 点击这个 button
            # 这个 button 的 callback 会设置 auto_next_trigger = True
            
            # 这里的 JS 有点技巧：它寻找页面上特定的 hidden button 并 click 它
            on_end_js = ""
            if auto_play and idx + 1 < len(chunks):
                on_end_js = """
                aud.onended = function() {
                    // 寻找 id 为 next-trigger-btn 的按钮并点击
                    const btns = window.parent.document.querySelectorAll('button');
                    for (let btn of btns) {
                        if (btn.innerText === "NEXT_TRIGGER") {
                            btn.click();
                            break;
                        }
                    }
                };
                """
            
            # 播放器组件
            player_html = f"""
            <div style="position:fixed; bottom:0; left:0; right:0; background:#262730; border-top:1px solid #444; padding:15px; z-index:9999; display:flex; align-items:center; justify-content:center; gap:20px; box-shadow: 0 -5px 20px rgba(0,0,0,0.5);">
                <span style="color:#fff; font-weight:bold; font-size:16px;">
                    🎧 正在朗读第 {idx+1} / {len(chunks)} 段
                </span>
                <audio id="main-player" controls autoplay style="width: 400px; height:40px;">
                    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                </audio>
                <div style="color:#aaa; font-size:12px;">(播放结束自动跳下一段)</div>
            </div>
            <script>
                var aud = document.getElementById("main-player");
                if(aud) {{
                    aud.playbackRate = {speed};
                    {on_end_js}
                }}
            </script>
            """
            components.html(player_html, height=80)
            
            # 这是一个“隐形”的按钮，用于接收 JS 的点击事件
            # 当它被点击时，触发 Python 逻辑跳转下一段
            def trigger_next():
                st.session_state.auto_next_trigger = True
                
            # 我们把这个按钮藏在视觉死角，或者用 CSS 隐藏，但 Streamlit button 很难完全隐藏
            # 我们可以把它放在 sidebar 最下面，或者用 empty 容器
            with st.sidebar:
                # 这里的 label 必须和 JS 里的 innerText 匹配
                st.button("NEXT_TRIGGER", key="auto_next_hidden_btn", on_click=trigger_next, 
                          type="secondary")
                # 用 CSS 隐藏这个按钮
                st.markdown("""
                <style>
                button[kind="secondary"] { 
                    /* 这是一个全局 hack，可能会误伤，但在 sidebar 底部通常没事 */
                }
                /* 专门针对特定文本的按钮隐藏比较难，
                   我们把它做的很小或者透明 */
                div.stButton > button:contains("NEXT_TRIGGER") {
                   display: none;
                }
                /* 这种 CSS 选择器 Streamlit 不一定支持，
                   所以上面的 JS 循环查找 innerText 是最稳的。
                   为了美观，我们在 Python 端不让用户容易看到它即可。
                */
                </style>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
