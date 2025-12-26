import asyncio
import base64
import hashlib
import io
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET

import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup

# -----------------------------
# Optional TTS (edge-tts)
# -----------------------------
try:
    import edge_tts  # pip install edge-tts
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="EPUB Web Reader", layout="wide")


# -----------------------------
# Helpers: path / decode
# -----------------------------
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
    if re.match(r"^[a-zA-Z]+://", href):
        return None, ""  # external link, ignore
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


def first_child_text(parent, tag_suffix: str):
    if parent is None:
        return None
    for el in parent.iter():
        if isinstance(el.tag, str) and el.tag.endswith(tag_suffix):
            if el.text and el.text.strip():
                return el.text.strip()
    return None


def soup_html(html: str) -> BeautifulSoup:
    # Prefer lxml; fallback to html.parser
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


# -----------------------------
# TOC parsing
# -----------------------------
def parse_nav_toc(z: zipfile.ZipFile, nav_path: str):
    html = decode_bytes(z.read(nav_path))
    soup = soup_html(html)

    nav = soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav", attrs={"role": "doc-toc"})
    if not nav:
        # fallback: any nav that seems like toc
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


# -----------------------------
# EPUB parsing (DRM-free)
# -----------------------------
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

    # metadata
    metadata_el = next((e for e in opf_root.iter() if isinstance(e.tag, str) and e.tag.endswith("metadata")), None)
    title = first_child_text(metadata_el, "title") or "Untitled"
    creator = first_child_text(metadata_el, "creator") or ""
    language = first_child_text(metadata_el, "language") or ""

    # manifest
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

    # mime map
    mime_by_path = {m["path"]: (m.get("media_type") or "") for m in manifest.values()}

    # spine
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
    spine_ids = []
    for idref in spine_idrefs:
        m = manifest.get(idref)
        if m and is_doc(m.get("media_type", "")):
            p = m["path"]
            if p in file_list:  # only include existing
                spine_paths.append(p)
                spine_ids.append(idref)

    if not spine_paths:
        raise ValueError("未能从 spine 中解析到可阅读的章节文档（可能是非常规 EPUB 或受保护文件）。")

    # find nav doc (epub3)
    nav_path = None
    for m in manifest.values():
        if "nav" in (m.get("properties") or "").split():
            nav_path = m["path"]
            break

    # find ncx (epub2)
    ncx_path = None
    if ncx_id and ncx_id in manifest:
        ncx_path = manifest[ncx_id]["path"]
    if not ncx_path:
        for m in manifest.values():
            mt = (m.get("media_type") or "").lower()
            if "dtbncx" in mt or mt.endswith(".ncx"):
                ncx_path = m["path"]
                break

    # cover image (best-effort)
    cover_path = None
    for m in manifest.values():
        props = (m.get("properties") or "").split()
        if "cover-image" in props and (m.get("media_type") or "").startswith("image/"):
            cover_path = m["path"]
            break
    if not cover_path:
        for m in manifest.values():
            if (m.get("media_type") or "").startswith("image/") and re.search(r"cover", m.get("id", ""), re.I):
                cover_path = m["path"]
                break
    if cover_path and cover_path not in file_list:
        cover_path = None

    # toc
    toc_entries = []
    if nav_path and nav_path in file_list:
        toc_entries = parse_nav_toc(z, nav_path)
    elif ncx_path and ncx_path in file_list:
        toc_entries = parse_ncx_toc(z, ncx_path)

    # map toc -> spine index
    path_to_index = {p: i for i, p in enumerate(spine_paths)}
    for e in toc_entries:
        p = e.get("path")
        if p in path_to_index:
            e["chapter_index"] = path_to_index[p]

    # chapter titles (use toc title if possible)
    chapter_titles = [None] * len(spine_paths)
    for e in toc_entries:
        idx = e.get("chapter_index")
        if isinstance(idx, int) and 0 <= idx < len(chapter_titles) and not chapter_titles[idx]:
            chapter_titles[idx] = e.get("title") or None

    for i in range(len(chapter_titles)):
        if not chapter_titles[i]:
            chapter_titles[i] = f"第 {i+1} 章"

    return {
        "title": title,
        "creator": creator,
        "language": language,
        "opf_path": opf_path,
        "spine_paths": spine_paths,
        "chapter_titles": chapter_titles,
        "toc_entries": toc_entries,
        "mime_by_path": mime_by_path,
        "cover_path": cover_path,
        "file_list": file_list,
    }


# -----------------------------
# Chapter rendering
# -----------------------------
def embed_images_in_body(body, chapter_path: str, z: zipfile.ZipFile, mime_by_path: dict, file_list: set, max_images: int = 200):
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

    # remove scripts
    for s in body.find_all("script"):
        s.decompose()

    # text
    text = body.get_text("\n", strip=True)

    # html for display
    if embed_images:
        embed_images_in_body(body, chapter_path, z, book["mime_by_path"], book["file_list"])
    body_html = "".join(str(x) for x in body.contents)

    return text, body_html


def wrap_reader_html(body_html: str, font_size: int, line_height: float, max_width: int, theme: str):
    if theme == "Dark":
        bg = "#0f1115"
        fg = "#e6e6e6"
        subtle = "#b6b6b6"
    else:
        bg = "#ffffff"
        fg = "#111111"
        subtle = "#555555"

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
    padding: 24px 18px 64px 18px;
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
</style>
</head>
<body>
  <div class="reader">
    {body_html}
  </div>
</body>
</html>"""


# -----------------------------
# Optional TTS
# -----------------------------
def chunk_text(text: str, max_chars: int = 3000):
    # Chunk by paragraphs first; fallback to hard split.
    paras = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
    chunks = []
    buf = ""
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

    # If a single paragraph is still too long, hard split
    out = []
    for c in chunks:
        if len(c) <= max_chars:
            out.append(c)
        else:
            for i in range(0, len(c), max_chars):
                out.append(c[i : i + max_chars])
    return out


def strip_id3_if_present(mp3_bytes: bytes) -> bytes:
    # Very small best-effort: remove ID3 header to reduce repeated tags when concatenating.
    if len(mp3_bytes) < 10:
        return mp3_bytes
    if mp3_bytes[:3] != b"ID3":
        return mp3_bytes

    # ID3 header: bytes 6-9 are synchsafe size
    size_bytes = mp3_bytes[6:10]
    size = ((size_bytes[0] & 0x7F) << 21) | ((size_bytes[1] & 0x7F) << 14) | ((size_bytes[2] & 0x7F) << 7) | (size_bytes[3] & 0x7F)
    start = 10 + size
    if start < len(mp3_bytes):
        return mp3_bytes[start:]
    return mp3_bytes


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


# -----------------------------
# UI
# -----------------------------
st.title("EPUB 网页阅读器（上传即读）")

with st.sidebar:
    st.header("打开 EPUB")
    uploaded = st.file_uploader("上传 EPUB（建议 DRM-free）", type=["epub"])

    st.divider()
    st.subheader("阅读设置")
    theme = st.radio("主题", ["Light", "Dark"], horizontal=True)
    font_size = st.slider("字号", 14, 28, 18, 1)
    line_height = st.slider("行距", 1.2, 2.2, 1.7, 0.05)
    max_width = st.slider("版心宽度（px）", 520, 1100, 780, 10)

    view_mode = st.radio("显示模式", ["排版（HTML）", "纯文本"], index=0)
    embed_images = st.checkbox("嵌入图片（部分书有插图时更完整）", value=True)

    st.divider()
    st.caption("提示：此工具用于你有合法权限阅读的 EPUB；DRM 书籍可能无法解析。")

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

# initialize / reset chapter index when book changes
if "book_hash" not in st.session_state or st.session_state.book_hash != book_hash:
    st.session_state.book_hash = book_hash
    st.session_state.chapter_idx = 0

# top info
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

# sidebar TOC selector
chapter_count = len(book["spine_paths"])
labels = book["chapter_titles"]

selected = st.sidebar.selectbox(
    "目录（按章节顺序）",
    options=list(range(chapter_count)),
    index=st.session_state.chapter_idx,
    format_func=lambda i: labels[i] if 0 <= i < len(labels) else f"第 {i+1} 章",
)

if selected != st.session_state.chapter_idx:
    st.session_state.chapter_idx = int(selected)

# navigation buttons
nav_cols = st.columns([1, 1, 6])
with nav_cols[0]:
    if st.button("上一章", disabled=(st.session_state.chapter_idx <= 0), use_container_width=True):
        st.session_state.chapter_idx -= 1
        st.rerun()
with nav_cols[1]:
    if st.button("下一章", disabled=(st.session_state.chapter_idx >= chapter_count - 1), use_container_width=True):
        st.session_state.chapter_idx += 1
        st.rerun()
with nav_cols[2]:
    st.markdown(f"### {labels[st.session_state.chapter_idx]}")

# render current chapter
chapter_text, chapter_body_html = extract_chapter_text_and_html(
    epub_bytes=epub_bytes,
    book=book,
    chapter_idx=st.session_state.chapter_idx,
    embed_images=embed_images if view_mode == "排版（HTML）" else False,
)

if view_mode == "纯文本":
    # Use markdown to preserve basic spacing
    safe_text = chapter_text.replace("\n", "  \n")
    st.markdown(safe_text)
else:
    full_html = wrap_reader_html(
        body_html=chapter_body_html,
        font_size=font_size,
        line_height=line_height,
        max_width=max_width,
        theme=theme,
    )
    components.html(full_html, height=900, scrolling=True)

# -----------------------------
# Optional TTS section
# -----------------------------
st.divider()
st.subheader("可选：生成“本章 MP3”并在线播放/下载（TTS）")

if not EDGE_TTS_AVAILABLE:
    st.caption("未安装 edge-tts。若需要本功能：在项目目录执行 `pip install edge-tts` 后重启 Streamlit。")
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

    tts_note = (
        "提示：本功能用于个人朗读/辅助阅读。若章节特别长，系统会自动分段合成并拼接为一个 MP3。"
    )
    st.caption(tts_note)

    if st.button("生成本章 MP3", use_container_width=True):
        # Basic safeguard against empty text
        if not chapter_text.strip():
            st.warning("本章内容为空或无法提取文本。")
        else:
            with st.spinner("正在生成音频…"):
                audio_bytes = asyncio.run(
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
