import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageChops, ImageOps
import cv2
import io
import base64
import copy
from scipy.ndimage import gaussian_filter
from skimage import exposure

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StreamPhoto — Pro Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg: #0f0f11;
  --panel: #18181c;
  --panel2: #202026;
  --accent: #7c6af7;
  --accent2: #f76a8c;
  --text: #e8e8f0;
  --muted: #6b6b7e;
  --border: #2a2a34;
  --success: #4ade80;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1,h2,h3 { font-family: 'Space Mono', monospace !important; }

.block-container { padding: 1rem 1.5rem !important; }

/* Buttons */
.stButton > button {
  background: var(--panel2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }

/* Selectbox */
[data-testid="stSelectbox"] select,
.stSelectbox > div > div {
  background: var(--panel2) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}

/* Expander */
[data-testid="stExpander"] {
  background: var(--panel2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* Tabs */
[data-testid="stTabs"] button {
  color: var(--muted) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.75rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* Canvas area */
.canvas-wrap {
  background: #111114;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
}

/* Toolbar */
.toolbar {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  padding: 8px;
  background: var(--panel);
  border-radius: 8px;
  border: 1px solid var(--border);
  margin-bottom: 10px;
}

/* Badge */
.badge {
  background: var(--accent);
  color: #fff;
  font-size: 0.65rem;
  padding: 2px 7px;
  border-radius: 20px;
  font-family: 'Space Mono', monospace;
}

/* Header bar */
.header-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0 14px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 14px;
}

.header-bar h1 {
  font-size: 1.35rem !important;
  margin: 0 !important;
  background: linear-gradient(135deg, #7c6af7, #f76a8c);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* Info box */
.info-box {
  background: var(--panel2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 0.8rem;
  color: var(--muted);
  margin-bottom: 10px;
}

.stImage > img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "original_image": None,
        "current_image": None,
        "history": [],          # list of PIL images for undo
        "redo_stack": [],
        "layers": [],           # list of dicts {name, image, visible, opacity, blend}
        "active_layer": 0,
        "tool": "Select",
        "brush_size": 20,
        "brush_color": "#ff0000",
        "brush_opacity": 100,
        "eraser_size": 20,
        "clone_src": None,      # (x, y) clone stamp source
        "clone_src_set": False,
        "text_content": "Your Text",
        "text_size": 36,
        "text_color": "#ffffff",
        "zoom": 100,
        "canvas_width": 800,
        "canvas_height": 600,
        "selection": None,      # (x1,y1,x2,y2)
        "crop_applied": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def pil_to_b64(img: Image.Image) -> str:
    return base64.b64encode(pil_to_bytes(img)).decode()

def push_history(img: Image.Image):
    st.session_state.history.append(img.copy())
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)
    st.session_state.redo_stack.clear()

def undo():
    if st.session_state.history:
        st.session_state.redo_stack.append(st.session_state.current_image.copy())
        st.session_state.current_image = st.session_state.history.pop()
        sync_layers_from_current()

def redo():
    if st.session_state.redo_stack:
        push_history(st.session_state.current_image)
        st.session_state.current_image = st.session_state.redo_stack.pop()
        sync_layers_from_current()

def sync_layers_from_current():
    if st.session_state.layers:
        st.session_state.layers[0]["image"] = st.session_state.current_image.copy()

def ensure_rgba(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGBA")
    return img

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def composite_layers() -> Image.Image:
    """Flatten all visible layers into one RGBA image."""
    if not st.session_state.layers:
        return st.session_state.current_image
    base = None
    for layer in reversed(st.session_state.layers):
        if not layer["visible"]:
            continue
        limg = ensure_rgba(layer["image"].copy())
        # apply opacity
        if layer["opacity"] < 100:
            r, g, b, a = limg.split()
            a = a.point(lambda x: int(x * layer["opacity"] / 100))
            limg = Image.merge("RGBA", (r, g, b, a))
        if base is None:
            base = limg
        else:
            base = Image.alpha_composite(base, limg)
    return base if base else st.session_state.current_image

def img_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(ensure_rgb(img)), cv2.COLOR_RGB2BGR)

def cv_to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

# ─────────────────────────────────────────────
# ADJUSTMENTS
# ─────────────────────────────────────────────
def apply_brightness(img, val):
    return ImageEnhance.Brightness(img).enhance(val / 100)

def apply_contrast(img, val):
    return ImageEnhance.Contrast(img).enhance(val / 100)

def apply_saturation(img, val):
    return ImageEnhance.Color(img).enhance(val / 100)

def apply_sharpness(img, val):
    return ImageEnhance.Sharpness(img).enhance(val / 100)

def apply_hue_rotation(img, degrees):
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + degrees / 2) % 180
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray((rgb * 255).astype(np.uint8))

def apply_gamma(img, gamma):
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    return Image.fromarray((arr * 255).astype(np.uint8))

def apply_exposure(img, stops):
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    arr = np.clip(arr * (2 ** stops), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def apply_curves(img, r_pts, g_pts, b_pts):
    """Apply per-channel tone curves via LUT."""
    arr = np.array(img.convert("RGB"))
    def make_lut(pts):
        lut = np.zeros(256, dtype=np.uint8)
        pts = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        for i in range(256):
            lut[i] = int(np.interp(i, xs, ys))
        return lut
    lr, lg, lb = make_lut(r_pts), make_lut(g_pts), make_lut(b_pts)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    arr[:,:,0] = lr[r]; arr[:,:,1] = lg[g]; arr[:,:,2] = lb[b]
    return Image.fromarray(arr)

def apply_levels(img, in_min, in_max, out_min, out_max):
    arr = np.array(img.convert("RGB")).astype(np.float32)
    arr = np.clip((arr - in_min) / max(in_max - in_min, 1), 0, 1)
    arr = arr * (out_max - out_min) + out_min
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_shadows_highlights(img, shadows, highlights):
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    # Shadows: lift dark areas
    if shadows != 0:
        mask = 1.0 - arr
        arr = arr + (shadows / 200.0) * mask * (1.0 - arr)
    # Highlights: compress bright areas
    if highlights != 0:
        arr = arr + (highlights / 200.0) * arr * (1.0 - arr)
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

def apply_vibrance(img, vibrance):
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    # Boost less-saturated pixels more
    sat = hsv[:,:,1]
    boost = vibrance / 200.0 * (1.0 - sat)
    hsv[:,:,1] = np.clip(sat + boost, 0, 1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray((rgb * 255).astype(np.uint8))

# ─────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────
def apply_filter(img: Image.Image, filter_name: str) -> Image.Image:
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if filter_name == "Blur":
        result = cv2.GaussianBlur(cv_img, (21, 21), 0)
        return cv_to_img(result)
    elif filter_name == "Sharpen":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        result = cv2.filter2D(cv_img, -1, kernel)
        return cv_to_img(result)
    elif filter_name == "Edge Detect":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    elif filter_name == "Emboss":
        kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
        result = cv2.filter2D(cv_img, -1, kernel) + 128
        return cv_to_img(result)
    elif filter_name == "Grayscale":
        return rgb.convert("L").convert("RGB")
    elif filter_name == "Sepia":
        sepia_filter = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
        result = cv2.transform(arr.astype(np.float32), sepia_filter)
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    elif filter_name == "Vintage":
        result = arr.astype(np.float32)
        result[:,:,0] = np.clip(result[:,:,0] * 1.1, 0, 255)
        result[:,:,2] = np.clip(result[:,:,2] * 0.85, 0, 255)
        img_v = Image.fromarray(result.astype(np.uint8))
        return ImageEnhance.Contrast(img_v).enhance(0.85)
    elif filter_name == "Vignette":
        rows, cols = arr.shape[:2]
        X = cv2.getGaussianKernel(cols, cols * 0.5)
        Y = cv2.getGaussianKernel(rows, rows * 0.5)
        mask = Y * X.T
        mask = mask / mask.max()
        result = arr.astype(np.float32)
        for i in range(3):
            result[:,:,i] *= mask
        return Image.fromarray(result.astype(np.uint8))
    elif filter_name == "Cartoon":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(cv_img, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cv_to_img(cartoon)
    elif filter_name == "Sketch":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256.0)
        return Image.fromarray(cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB))
    elif filter_name == "Oil Paint":
        result = cv2.xphoto.oilPainting(cv_img, 7, 1) if hasattr(cv2, 'xphoto') else cv2.bilateralFilter(cv_img, 9, 75, 75)
        return cv_to_img(result)
    elif filter_name == "HDR":
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv_to_img(result)
    elif filter_name == "Noise Reduction":
        result = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
        return cv_to_img(result)
    elif filter_name == "Chromatic Aberration":
        result = arr.copy()
        shift = 4
        result[:, shift:, 0] = arr[:, :-shift, 0]
        result[:, :-shift, 2] = arr[:, shift:, 2]
        return Image.fromarray(result)
    elif filter_name == "Glitch":
        result = arr.copy()
        for _ in range(8):
            y = np.random.randint(0, arr.shape[0])
            shift = np.random.randint(-20, 20)
            result[y] = np.roll(arr[y], shift, axis=0)
        return Image.fromarray(result)
    elif filter_name == "Posterize":
        return ImageOps.posterize(rgb, 3)
    elif filter_name == "Solarize":
        return ImageOps.solarize(rgb, 128)
    return rgb

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
def rotate_image(img, degrees, expand=True):
    return img.rotate(-degrees, expand=expand, resample=Image.BICUBIC)

def flip_horizontal(img):
    return ImageOps.mirror(img)

def flip_vertical(img):
    return ImageOps.flip(img)

def resize_image(img, width, height, resample="Lanczos"):
    methods = {"Nearest": Image.NEAREST, "Bilinear": Image.BILINEAR,
               "Bicubic": Image.BICUBIC, "Lanczos": Image.LANCZOS}
    return img.resize((width, height), methods.get(resample, Image.LANCZOS))

def crop_image(img, x1, y1, x2, y2):
    w, h = img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        return img.crop((x1, y1, x2, y2))
    return img

def perspective_transform(img, src_points, dst_points):
    arr = img_to_cv(img)
    h, w = arr.shape[:2]
    src = np.float32(src_points)
    dst = np.float32(dst_points)
    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(arr, M, (w, h))
    return cv_to_img(result)

# ─────────────────────────────────────────────
# DRAWING / PAINTING TOOLS
# ─────────────────────────────────────────────
def draw_brush_stroke(img: Image.Image, x, y, size, color_hex, opacity):
    rgba = ensure_rgba(img.copy())
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)
    a = int(255 * opacity / 100)
    half = size // 2
    draw.ellipse([x - half, y - half, x + half, y + half], fill=(r, g, b, a))
    return Image.alpha_composite(rgba, overlay)

def draw_eraser(img: Image.Image, x, y, size):
    rgba = ensure_rgba(img.copy())
    draw = ImageDraw.Draw(rgba)
    half = size // 2
    draw.ellipse([x - half, y - half, x + half, y + half], fill=(0, 0, 0, 0))
    return rgba

def apply_clone_stamp(img: Image.Image, src_x, src_y, dst_x, dst_y, size):
    """Clone a circular region from (src_x,src_y) to (dst_x,dst_y)."""
    rgba = ensure_rgba(img.copy())
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    half = size // 2
    # Extract source patch
    sx1, sy1 = max(0, src_x - half), max(0, src_y - half)
    sx2, sy2 = min(w, src_x + half), min(h, src_y + half)
    patch = arr[sy1:sy2, sx1:sx2].copy()
    ph, pw = patch.shape[:2]
    # Paste to destination
    dx1, dy1 = max(0, dst_x - half), max(0, dst_y - half)
    dx2, dy2 = min(w, dx1 + pw), min(h, dy1 + ph)
    arr[dy1:dy2, dx1:dx2] = patch[:dy2 - dy1, :dx2 - dx1]
    return Image.fromarray(arr)

def apply_healing_brush(img: Image.Image, x, y, size):
    """Simple healing: replace with median of surrounding area."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    half = size // 2
    # Sample a ring around the brush
    x1, y1 = max(0, x - half * 2), max(0, y - half * 2)
    x2, y2 = min(w, x + half * 2), min(h, y + half * 2)
    region = arr[y1:y2, x1:x2]
    median = np.median(region.reshape(-1, 3), axis=0).astype(np.uint8)
    # Fill brush circle with median
    px1, py1 = max(0, x - half), max(0, y - half)
    px2, py2 = min(w, x + half), min(h, y + half)
    for iy in range(py1, py2):
        for ix in range(px1, px2):
            if (ix - x) ** 2 + (iy - y) ** 2 <= half ** 2:
                arr[iy, ix] = median
    return Image.fromarray(arr)

def apply_blur_brush(img: Image.Image, x, y, size, strength=5):
    """Apply localized Gaussian blur."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    half = size // 2
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    region = arr[y1:y2, x1:x2].astype(np.float32)
    blurred = gaussian_filter(region, sigma=strength)
    arr[y1:y2, x1:x2] = blurred.astype(np.uint8)
    return Image.fromarray(arr)

def apply_smudge_brush(img: Image.Image, x, y, dx, dy, size, strength=0.5):
    """Smudge: shift pixels along drag direction."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    half = size // 2
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    region = arr[y1:y2, x1:x2].astype(np.float32)
    # Shift region
    shifted = np.roll(region, (int(dy * strength), int(dx * strength)), axis=(0, 1))
    arr[y1:y2, x1:x2] = (region * (1 - strength) + shifted * strength).astype(np.uint8)
    return Image.fromarray(arr)

def draw_shape(img: Image.Image, shape, x1, y1, x2, y2, color_hex, fill_hex, thickness):
    rgba = ensure_rgba(img.copy())
    draw = ImageDraw.Draw(rgba)

    def hex_to_rgba(h, alpha=255):
        if h and len(h) >= 7:
            return (int(h[1:3],16), int(h[3:5],16), int(h[5:7],16), alpha)
        return None

    stroke = hex_to_rgba(color_hex)
    fill = hex_to_rgba(fill_hex) if fill_hex else None

    if shape == "Rectangle":
        draw.rectangle([x1, y1, x2, y2], outline=stroke, fill=fill, width=thickness)
    elif shape == "Ellipse":
        draw.ellipse([x1, y1, x2, y2], outline=stroke, fill=fill, width=thickness)
    elif shape == "Line":
        draw.line([x1, y1, x2, y2], fill=stroke, width=thickness)
    elif shape == "Arrow":
        draw.line([x1, y1, x2, y2], fill=stroke, width=thickness)
        # Arrowhead
        angle = np.arctan2(y2 - y1, x2 - x1)
        for a in [angle + 2.5, angle - 2.5]:
            ax = int(x2 - 20 * np.cos(a))
            ay = int(y2 - 20 * np.sin(a))
            draw.line([x2, y2, ax, ay], fill=stroke, width=thickness)
    elif shape == "Triangle":
        cx = (x1 + x2) // 2
        pts = [(cx, y1), (x2, y2), (x1, y2)]
        draw.polygon(pts, outline=stroke, fill=fill)
    return rgba

def add_text_layer(img: Image.Image, text, x, y, font_size, color_hex):
    rgba = ensure_rgba(img.copy())
    draw = ImageDraw.Draw(rgba)
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=(r, g, b, 255), font=font)
    return rgba

# ─────────────────────────────────────────────
# SELECTION TOOLS
# ─────────────────────────────────────────────
def apply_selection_fill(img: Image.Image, x1, y1, x2, y2, color_hex):
    rgba = ensure_rgba(img.copy())
    draw = ImageDraw.Draw(rgba)
    r = int(color_hex[1:3], 16)
    g = int(color_hex[3:5], 16)
    b = int(color_hex[5:7], 16)
    draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 255))
    return rgba

def magic_wand_select(img: Image.Image, x, y, tolerance=30):
    """Flood-fill based selection mask."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    seed_color = arr[min(y, h-1), min(x, w-1)]
    mask = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        diff = np.abs(arr[cy, cx].astype(int) - seed_color.astype(int)).sum()
        if diff <= tolerance * 3:
            mask[cy, cx] = True
            for nx, ny in [(cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)]:
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    stack.append((nx, ny))
    return mask

# ─────────────────────────────────────────────
# LAYER MANAGEMENT
# ─────────────────────────────────────────────
def new_layer(name, img=None, w=None, h=None):
    if img is None:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return {"name": name, "image": ensure_rgba(img), "visible": True, "opacity": 100, "blend": "Normal"}

def add_layer():
    if st.session_state.current_image:
        w, h = st.session_state.current_image.size
        layer = new_layer(f"Layer {len(st.session_state.layers) + 1}", w=w, h=h)
        st.session_state.layers.insert(0, layer)
        st.session_state.active_layer = 0

def delete_layer(idx):
    if len(st.session_state.layers) > 1:
        st.session_state.layers.pop(idx)
        st.session_state.active_layer = max(0, idx - 1)

def merge_down(idx):
    if idx < len(st.session_state.layers) - 1:
        top = ensure_rgba(st.session_state.layers[idx]["image"])
        bot = ensure_rgba(st.session_state.layers[idx + 1]["image"])
        merged = Image.alpha_composite(bot, top)
        st.session_state.layers[idx + 1]["image"] = merged
        st.session_state.layers.pop(idx)
        st.session_state.active_layer = max(0, idx - 1)

def flatten_image():
    flat = composite_layers().convert("RGB")
    st.session_state.current_image = flat
    st.session_state.layers = [new_layer("Background", flat)]
    st.session_state.active_layer = 0

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
def export_image(img: Image.Image, fmt: str, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=0)
    elif fmt == "PNG":
        img.save(buf, format="PNG", optimize=True)
    elif fmt == "WEBP":
        img.save(buf, format="WEBP", quality=quality, method=6)
    elif fmt == "BMP":
        img.convert("RGB").save(buf, format="BMP")
    elif fmt == "TIFF":
        img.save(buf, format="TIFF", compression="tiff_lzw")
    return buf.getvalue()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>StreamPhoto</h1>
  <span class="badge">PRO</span>
  <span style="color:#6b6b7e;font-size:0.8rem;margin-left:auto;">Professional Image Editor</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 File")

    uploaded = st.file_uploader("Import Image", type=["png","jpg","jpeg","webp","bmp","tiff","gif"],
                                 label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        st.session_state.original_image = img.copy()
        st.session_state.current_image = img.copy()
        w, h = img.size
        st.session_state.canvas_width = min(w, 900)
        st.session_state.canvas_height = int(h * st.session_state.canvas_width / w)
        st.session_state.layers = [new_layer("Background", img)]
        st.session_state.active_layer = 0
        st.session_state.history.clear()
        st.session_state.redo_stack.clear()
        st.success(f"✓ Loaded {w}×{h}px", icon="🖼️")

    # New blank canvas
    with st.expander("🆕 New Canvas"):
        nc_w = st.number_input("Width", 100, 8000, 1920, 10)
        nc_h = st.number_input("Height", 100, 8000, 1080, 10)
        nc_color = st.color_picker("Background", "#ffffff")
        if st.button("Create Canvas"):
            r = int(nc_color[1:3],16); g = int(nc_color[3:5],16); b = int(nc_color[5:7],16)
            blank = Image.new("RGB", (nc_w, nc_h), (r,g,b))
            st.session_state.original_image = blank.copy()
            st.session_state.current_image = blank.copy()
            st.session_state.canvas_width = min(nc_w, 900)
            st.session_state.canvas_height = int(nc_h * st.session_state.canvas_width / nc_w)
            st.session_state.layers = [new_layer("Background", blank)]
            st.session_state.active_layer = 0
            st.session_state.history.clear()
            st.session_state.redo_stack.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### 🛠 Tools")

    tools = ["Select","Crop","Move",
             "Brush","Eraser","Clone Stamp","Healing Brush",
             "Blur Brush","Smudge",
             "Shape","Text","Fill",
             "Magic Wand","Eyedropper"]
    icons  = ["⬚","✂️","✋",
              "🖌️","◻️","🪝","💊",
              "🌫️","👆",
              "◼️","T","🪣",
              "🪄","🔍"]

    cols = st.columns(3)
    for i, (tool, icon) in enumerate(zip(tools, icons)):
        with cols[i % 3]:
            active = "✓ " if st.session_state.tool == tool else ""
            if st.button(f"{icon}", key=f"tool_{tool}", help=tool, use_container_width=True):
                st.session_state.tool = tool

    st.caption(f"Active: **{st.session_state.tool}**")

    st.markdown("---")
    st.markdown("### ⚙️ Tool Options")

    tool = st.session_state.tool

    if tool in ("Brush","Clone Stamp","Healing Brush","Blur Brush","Smudge","Eraser"):
        size_key = "eraser_size" if tool == "Eraser" else "brush_size"
        st.session_state[size_key] = st.slider("Size", 1, 300,
                                                st.session_state[size_key], key=f"sz_{tool}")

    if tool == "Brush":
        st.session_state.brush_color = st.color_picker("Color", st.session_state.brush_color)
        st.session_state.brush_opacity = st.slider("Opacity %", 1, 100, st.session_state.brush_opacity)

    if tool == "Shape":
        st.session_state["shape_type"] = st.selectbox("Shape", ["Rectangle","Ellipse","Line","Arrow","Triangle"])
        st.session_state["shape_color"] = st.color_picker("Stroke Color", "#ff0000", key="shpcol")
        st.session_state["shape_fill"] = st.color_picker("Fill Color", "#ff000033", key="shpfil") if st.checkbox("Fill") else ""
        st.session_state["shape_thickness"] = st.slider("Thickness", 1, 20, 2)

    if tool == "Text":
        st.session_state.text_content = st.text_input("Text", st.session_state.text_content)
        st.session_state.text_size = st.slider("Font Size", 8, 300, st.session_state.text_size)
        st.session_state.text_color = st.color_picker("Color", st.session_state.text_color)

    if tool == "Clone Stamp":
        st.info("Alt+Click = set source. Click = stamp.")
        if st.session_state.clone_src:
            st.caption(f"Source: {st.session_state.clone_src}")
        if st.button("Clear Source"):
            st.session_state.clone_src = None
            st.session_state.clone_src_set = False

    if tool == "Magic Wand":
        st.session_state["mw_tolerance"] = st.slider("Tolerance", 1, 100, 30)

    if tool == "Fill":
        st.session_state["fill_color"] = st.color_picker("Fill Color", "#ff0000")

    if tool == "Crop":
        with st.expander("Crop Settings"):
            cx1 = st.number_input("X1", 0, 9999, 0, key="crop_x1")
            cy1 = st.number_input("Y1", 0, 9999, 0, key="crop_y1")
            cx2 = st.number_input("X2", 0, 9999, 800, key="crop_x2")
            cy2 = st.number_input("Y2", 0, 9999, 600, key="crop_y2")
            if st.button("Apply Crop") and st.session_state.current_image:
                push_history(st.session_state.current_image)
                st.session_state.current_image = crop_image(
                    st.session_state.current_image, cx1, cy1, cx2, cy2)
                sync_layers_from_current()
                st.rerun()

    st.markdown("---")
    # Undo/Redo
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩ Undo", use_container_width=True):
            undo(); st.rerun()
    with c2:
        if st.button("↪ Redo", use_container_width=True):
            redo(); st.rerun()

    if st.button("↺ Reset to Original", use_container_width=True):
        if st.session_state.original_image:
            push_history(st.session_state.current_image)
            st.session_state.current_image = st.session_state.original_image.copy()
            sync_layers_from_current()
            st.rerun()

    st.markdown("---")
    st.markdown("### 📤 Export")
    exp_fmt = st.selectbox("Format", ["PNG","JPEG","WEBP","BMP","TIFF"])
    exp_quality = st.slider("Quality", 1, 100, 95) if exp_fmt in ("JPEG","WEBP") else 95
    exp_flatten = st.checkbox("Flatten Layers", value=True)

    if st.button("⬇ Export & Download", use_container_width=True):
        if st.session_state.current_image:
            out_img = composite_layers() if exp_flatten else st.session_state.current_image
            data = export_image(out_img, exp_fmt, exp_quality)
            ext = exp_fmt.lower().replace("jpeg","jpg")
            st.download_button(
                label=f"💾 Save {exp_fmt}",
                data=data,
                file_name=f"streampho_export.{ext}",
                mime=f"image/{ext}",
                use_container_width=True
            )
            w, h = out_img.size
            st.caption(f"Size: {w}×{h}px · {len(data)//1024}KB")

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
if st.session_state.current_image is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
    min-height:60vh;gap:16px;color:#6b6b7e;">
      <div style="font-size:4rem;">🎨</div>
      <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#7c6af7;">StreamPhoto</div>
      <div style="font-size:0.9rem;">Import an image or create a new canvas to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_canvas, tab_adjust, tab_filters, tab_transform, tab_layers, tab_histogram = st.tabs([
    "🖼 Canvas", "🎛 Adjust", "✨ Filters", "↔ Transform", "🗂 Layers", "📊 Histogram"
])

# ══════════════════════════════════════════════
# TAB: CANVAS
# ══════════════════════════════════════════════
with tab_canvas:
    img = st.session_state.current_image
    w, h = img.size

    col_canvas, col_props = st.columns([4, 1])

    with col_props:
        st.markdown("**Canvas Info**")
        st.markdown(f"""
        <div class="info-box">
          <b>Size:</b> {w} × {h}px<br>
          <b>Mode:</b> {img.mode}<br>
          <b>Layers:</b> {len(st.session_state.layers)}<br>
          <b>History:</b> {len(st.session_state.history)}<br>
          <b>Tool:</b> {st.session_state.tool}
        </div>
        """, unsafe_allow_html=True)

        st.session_state.zoom = st.slider("Zoom %", 10, 400, st.session_state.zoom, 10)

        st.markdown("**Quick Actions**")
        if st.button("Flatten All Layers", use_container_width=True):
            flatten_image()
            st.rerun()

        st.markdown("**Drawing Controls**")
        if st.session_state.tool == "Brush":
            st.caption(f"Color: {st.session_state.brush_color} | Size: {st.session_state.brush_size}")

        # Interactive drawing area (numeric input based for reliability)
        if st.session_state.tool in ("Brush","Eraser","Clone Stamp","Healing Brush",
                                      "Blur Brush","Smudge","Shape","Text","Fill","Magic Wand"):
            st.markdown("---")
            st.markdown("**Apply at Position**")
            px = st.number_input("X", 0, w-1, w//2, key="draw_x")
            py = st.number_input("Y", 0, h-1, h//2, key="draw_y")

            if st.session_state.tool == "Shape":
                px2 = st.number_input("X2", 0, w-1, min(w-1, w//2+100), key="draw_x2")
                py2 = st.number_input("Y2", 0, h-1, min(h-1, h//2+100), key="draw_y2")

            btn_label = {"Brush":"🖌 Paint","Eraser":"◻ Erase",
                          "Clone Stamp":"🪝 Stamp","Healing Brush":"💊 Heal",
                          "Blur Brush":"🌫 Blur","Smudge":"👆 Smudge",
                          "Shape":"◼ Draw Shape","Text":"T Add Text",
                          "Fill":"🪣 Fill","Magic Wand":"🪄 Select"}.get(st.session_state.tool,"Apply")

            if st.button(btn_label, use_container_width=True):
                push_history(st.session_state.current_image)
                tool = st.session_state.tool
                ci = st.session_state.current_image

                if tool == "Brush":
                    ci = draw_brush_stroke(ci, px, py,
                                           st.session_state.brush_size,
                                           st.session_state.brush_color,
                                           st.session_state.brush_opacity)
                elif tool == "Eraser":
                    ci = draw_eraser(ci, px, py, st.session_state.eraser_size)
                elif tool == "Clone Stamp":
                    if st.session_state.clone_src:
                        sx, sy = st.session_state.clone_src
                        ci = apply_clone_stamp(ci, sx, sy, px, py, st.session_state.brush_size)
                    else:
                        st.session_state.clone_src = (px, py)
                        st.info("Source set! Now choose destination and stamp again.")
                elif tool == "Healing Brush":
                    ci = apply_healing_brush(ci, px, py, st.session_state.brush_size)
                elif tool == "Blur Brush":
                    ci = apply_blur_brush(ci, px, py, st.session_state.brush_size)
                elif tool == "Smudge":
                    ci = apply_smudge_brush(ci, px, py, 5, 5, st.session_state.brush_size)
                elif tool == "Shape":
                    ci = draw_shape(ci,
                                    st.session_state.get("shape_type","Rectangle"),
                                    px, py, px2, py2,
                                    st.session_state.get("shape_color","#ff0000"),
                                    st.session_state.get("shape_fill",""),
                                    st.session_state.get("shape_thickness",2))
                elif tool == "Text":
                    ci = add_text_layer(ci,
                                        st.session_state.text_content,
                                        px, py,
                                        st.session_state.text_size,
                                        st.session_state.text_color)
                elif tool == "Fill":
                    ci = apply_selection_fill(ci, px-20, py-20, px+20, py+20,
                                              st.session_state.get("fill_color","#ff0000"))
                elif tool == "Magic Wand":
                    mask = magic_wand_select(ci, px, py,
                                             st.session_state.get("mw_tolerance",30))
                    overlay = ensure_rgba(ci.copy())
                    arr = np.array(overlay)
                    arr[mask, 3] = 128
                    ci = Image.fromarray(arr)

                st.session_state.current_image = ci
                if st.session_state.layers:
                    st.session_state.layers[st.session_state.active_layer]["image"] = ensure_rgba(ci)
                st.rerun()

            if st.session_state.tool == "Clone Stamp":
                st.markdown("---")
                st.markdown("**Set Clone Source**")
                csx = st.number_input("Src X", 0, w-1, w//4, key="clone_sx")
                csy = st.number_input("Src Y", 0, h-1, h//4, key="clone_sy")
                if st.button("📍 Set Source"):
                    st.session_state.clone_src = (csx, csy)
                    st.success(f"Source: ({csx},{csy})")

    with col_canvas:
        # Compute display size
        zoom = st.session_state.zoom / 100
        dw = int(w * zoom)
        dh = int(h * zoom)
        dw = max(100, min(dw, 1400))
        dh = max(100, min(dh, 900))

        display_img = composite_layers()
        display_img_resized = display_img.resize((dw, dh), Image.LANCZOS)

        # Show canvas
        st.image(display_img_resized, use_container_width=False,
                 caption=f"{w}×{h}px | {display_img.mode} | Zoom {st.session_state.zoom}%",
                 clamp=True)

        st.caption("💡 Use tool controls in the right panel to apply drawing operations at coordinates")

# ══════════════════════════════════════════════
# TAB: ADJUSTMENTS
# ══════════════════════════════════════════════
with tab_adjust:
    img = st.session_state.current_image
    col_adj1, col_adj2 = st.columns(2)

    with col_adj1:
        with st.expander("☀️ Basic Adjustments", expanded=True):
            brightness = st.slider("Brightness", 1, 300, 100, key="adj_bright")
            contrast   = st.slider("Contrast",   1, 300, 100, key="adj_cont")
            saturation = st.slider("Saturation", 0, 300, 100, key="adj_sat")
            sharpness  = st.slider("Sharpness",  0, 300, 100, key="adj_sharp")
            vibrance   = st.slider("Vibrance", -100, 100, 0, key="adj_vib")

            if st.button("✅ Apply Basic Adjustments"):
                push_history(img)
                ci = apply_brightness(img, brightness)
                ci = apply_contrast(ci, contrast)
                ci = apply_saturation(ci, saturation)
                ci = apply_sharpness(ci, sharpness)
                if vibrance != 0:
                    ci = apply_vibrance(ci, vibrance)
                st.session_state.current_image = ci
                sync_layers_from_current()
                st.rerun()

        with st.expander("🌡 Tone"):
            exposure_v = st.slider("Exposure (stops)", -3.0, 3.0, 0.0, 0.1, key="adj_exp")
            gamma_v    = st.slider("Gamma", 0.2, 3.0, 1.0, 0.05, key="adj_gam")
            hue_v      = st.slider("Hue Rotation", -180, 180, 0, key="adj_hue")
            shadow_v   = st.slider("Shadows", -100, 100, 0, key="adj_shad")
            highlight_v= st.slider("Highlights", -100, 100, 0, key="adj_high")

            if st.button("✅ Apply Tone"):
                push_history(img)
                ci = img
                if exposure_v != 0:
                    ci = apply_exposure(ci, exposure_v)
                if gamma_v != 1.0:
                    ci = apply_gamma(ci, gamma_v)
                if hue_v != 0:
                    ci = apply_hue_rotation(ci, hue_v)
                if shadow_v != 0 or highlight_v != 0:
                    ci = apply_shadows_highlights(ci, shadow_v, highlight_v)
                st.session_state.current_image = ci
                sync_layers_from_current()
                st.rerun()

    with col_adj2:
        with st.expander("📊 Levels"):
            in_min  = st.slider("Input Min",  0, 254, 0,   key="lv_inmin")
            in_max  = st.slider("Input Max",  1, 255, 255, key="lv_inmax")
            out_min = st.slider("Output Min", 0, 254, 0,   key="lv_outmin")
            out_max = st.slider("Output Max", 1, 255, 255, key="lv_outmax")
            if st.button("✅ Apply Levels"):
                push_history(img)
                st.session_state.current_image = apply_levels(img, in_min, in_max, out_min, out_max)
                sync_layers_from_current()
                st.rerun()

        with st.expander("📈 Curves (per channel)"):
            st.caption("Define control points as (input, output) pairs")
            ch = st.selectbox("Channel", ["Red","Green","Blue"], key="curve_ch")
            pts_input = st.text_area("Control Points (x,y pairs)",
                                      "0,0\n128,128\n255,255", key="curve_pts")
            if st.button("✅ Apply Curves"):
                try:
                    pts = [tuple(map(int, p.strip().split(",")))
                           for p in pts_input.strip().split("\n") if p.strip()]
                    push_history(img)
                    r_pts = pts if ch == "Red"   else [(0,0),(255,255)]
                    g_pts = pts if ch == "Green" else [(0,0),(255,255)]
                    b_pts = pts if ch == "Blue"  else [(0,0),(255,255)]
                    st.session_state.current_image = apply_curves(img, r_pts, g_pts, b_pts)
                    sync_layers_from_current()
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid points: {e}")

        with st.expander("🎨 Color Balance"):
            cb_r = st.slider("Red Channel",  -100, 100, 0, key="cb_r")
            cb_g = st.slider("Green Channel",-100, 100, 0, key="cb_g")
            cb_b = st.slider("Blue Channel", -100, 100, 0, key="cb_b")
            if st.button("✅ Apply Color Balance"):
                push_history(img)
                arr = np.array(img.convert("RGB")).astype(np.float32)
                arr[:,:,0] = np.clip(arr[:,:,0] + cb_r, 0, 255)
                arr[:,:,1] = np.clip(arr[:,:,1] + cb_g, 0, 255)
                arr[:,:,2] = np.clip(arr[:,:,2] + cb_b, 0, 255)
                st.session_state.current_image = Image.fromarray(arr.astype(np.uint8))
                sync_layers_from_current()
                st.rerun()

# ══════════════════════════════════════════════
# TAB: FILTERS
# ══════════════════════════════════════════════
with tab_filters:
    filter_list = ["Blur","Sharpen","Edge Detect","Emboss","Grayscale","Sepia",
                   "Vintage","Vignette","Cartoon","Sketch","HDR","Noise Reduction",
                   "Chromatic Aberration","Glitch","Posterize","Solarize"]

    st.markdown("#### 🖼 Preview & Apply Filters")
    img = st.session_state.current_image

    # Show filter grid previews
    thumb = img.copy()
    thumb.thumbnail((160, 120), Image.LANCZOS)

    cols_per_row = 4
    rows = [filter_list[i:i+cols_per_row] for i in range(0, len(filter_list), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, fname in zip(cols, row):
            with col:
                try:
                    prev = apply_filter(thumb.copy(), fname)
                    st.image(prev, caption=fname, use_container_width=True)
                    if st.button(f"Apply", key=f"flt_{fname}", use_container_width=True):
                        push_history(img)
                        st.session_state.current_image = apply_filter(img, fname)
                        sync_layers_from_current()
                        st.rerun()
                except Exception as e:
                    st.caption(f"{fname}: error")

    st.markdown("---")
    st.markdown("#### 🔧 Advanced Filters")
    col_adv1, col_adv2 = st.columns(2)

    with col_adv1:
        with st.expander("🌊 Custom Blur"):
            blur_type = st.selectbox("Type", ["Gaussian","Median","Box","Motion"])
            blur_radius = st.slider("Radius", 1, 50, 5, key="cblur_r")
            if st.button("Apply Custom Blur"):
                push_history(img)
                arr = img_to_cv(img)
                if blur_type == "Gaussian":
                    r = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                    result = cv2.GaussianBlur(arr, (r, r), 0)
                elif blur_type == "Median":
                    r = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                    result = cv2.medianBlur(arr, r)
                elif blur_type == "Box":
                    result = cv2.boxFilter(arr, -1, (blur_radius, blur_radius))
                elif blur_type == "Motion":
                    kernel = np.zeros((blur_radius, blur_radius))
                    kernel[blur_radius//2, :] = np.ones(blur_radius) / blur_radius
                    result = cv2.filter2D(arr, -1, kernel)
                st.session_state.current_image = cv_to_img(result)
                sync_layers_from_current()
                st.rerun()

        with st.expander("🎯 Unsharp Mask"):
            um_strength = st.slider("Strength", 0.1, 5.0, 1.5, 0.1)
            um_radius   = st.slider("Radius", 1, 10, 2)
            um_threshold= st.slider("Threshold", 0, 50, 3)
            if st.button("Apply Unsharp Mask"):
                push_history(img)
                result = img.filter(ImageFilter.UnsharpMask(
                    radius=um_radius, percent=int(um_strength*100), threshold=um_threshold))
                st.session_state.current_image = result
                sync_layers_from_current()
                st.rerun()

    with col_adv2:
        with st.expander("🌈 Gradient Map"):
            gm_start = st.color_picker("Shadow Color", "#000066")
            gm_end   = st.color_picker("Highlight Color", "#ffcc00")
            if st.button("Apply Gradient Map"):
                push_history(img)
                gray = np.array(img.convert("L")).astype(np.float32) / 255.0
                sr = int(gm_start[1:3],16); sg = int(gm_start[3:5],16); sb = int(gm_start[5:7],16)
                er = int(gm_end[1:3],16);   eg = int(gm_end[3:5],16);   eb = int(gm_end[5:7],16)
                r_ch = (gray * (er-sr) + sr).astype(np.uint8)
                g_ch = (gray * (eg-sg) + sg).astype(np.uint8)
                b_ch = (gray * (eb-sb) + sb).astype(np.uint8)
                mapped = Image.fromarray(np.stack([r_ch,g_ch,b_ch], axis=-1))
                st.session_state.current_image = mapped
                sync_layers_from_current()
                st.rerun()

        with st.expander("🎭 Blend Modes (with color)"):
            bm_mode = st.selectbox("Mode", ["Multiply","Screen","Overlay","Soft Light","Difference"])
            bm_color = st.color_picker("Blend Color", "#ff8800")
            bm_opacity = st.slider("Opacity", 1, 100, 50, key="bm_op")
            if st.button("Apply Blend"):
                push_history(img)
                r2=int(bm_color[1:3],16); g2=int(bm_color[3:5],16); b2=int(bm_color[5:7],16)
                color_layer = Image.new("RGB", img.size, (r2,g2,b2))
                base = img.convert("RGB")
                base_arr = np.array(base).astype(np.float32)/255
                over_arr = np.array(color_layer).astype(np.float32)/255
                if bm_mode == "Multiply":
                    result = base_arr * over_arr
                elif bm_mode == "Screen":
                    result = 1-(1-base_arr)*(1-over_arr)
                elif bm_mode == "Overlay":
                    mask = base_arr < 0.5
                    result = np.where(mask, 2*base_arr*over_arr, 1-2*(1-base_arr)*(1-over_arr))
                elif bm_mode == "Soft Light":
                    result = (1-2*over_arr)*base_arr**2 + 2*over_arr*base_arr
                elif bm_mode == "Difference":
                    result = np.abs(base_arr - over_arr)
                blend = Image.fromarray((result*255).astype(np.uint8))
                op = bm_opacity/100
                final = Image.blend(base, blend, op)
                st.session_state.current_image = final
                sync_layers_from_current()
                st.rerun()

# ══════════════════════════════════════════════
# TAB: TRANSFORM
# ══════════════════════════════════════════════
with tab_transform:
    img = st.session_state.current_image
    w, h = img.size

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        with st.expander("🔄 Rotate", expanded=True):
            rot_angle = st.slider("Angle (°)", -180, 180, 0, key="rot_ang")
            rot_expand = st.checkbox("Expand canvas", value=True, key="rot_exp")
            if st.button("Apply Rotation"):
                push_history(img)
                st.session_state.current_image = rotate_image(img, rot_angle, rot_expand)
                sync_layers_from_current()
                st.rerun()

        with st.expander("↔ Flip"):
            c1, c2 = st.columns(2)
            with c1:
                if st.button("↔ Horizontal", use_container_width=True):
                    push_history(img); st.session_state.current_image = flip_horizontal(img)
                    sync_layers_from_current(); st.rerun()
            with c2:
                if st.button("↕ Vertical", use_container_width=True):
                    push_history(img); st.session_state.current_image = flip_vertical(img)
                    sync_layers_from_current(); st.rerun()

        with st.expander("📐 Resize"):
            rs_method = st.selectbox("Resample", ["Lanczos","Bicubic","Bilinear","Nearest"])
            keep_ar = st.checkbox("Keep Aspect Ratio", value=True)
            new_w = st.number_input("Width", 1, 16000, w, key="rs_w")
            if keep_ar:
                new_h = int(h * new_w / w)
                st.caption(f"Height → {new_h}px (locked)")
            else:
                new_h = st.number_input("Height", 1, 16000, h, key="rs_h")
            if st.button("Apply Resize"):
                push_history(img)
                st.session_state.current_image = resize_image(img, new_w, new_h, rs_method)
                sync_layers_from_current()
                st.rerun()

    with col_t2:
        with st.expander("✂️ Crop"):
            cr_x1 = st.number_input("X1", 0, w-1, 0, key="tcr_x1")
            cr_y1 = st.number_input("Y1", 0, h-1, 0, key="tcr_y1")
            cr_x2 = st.number_input("X2", 1, w, w, key="tcr_x2")
            cr_y2 = st.number_input("Y2", 1, h, h, key="tcr_y2")
            if st.button("Apply Crop", key="tcrop_btn"):
                push_history(img)
                st.session_state.current_image = crop_image(img, cr_x1, cr_y1, cr_x2, cr_y2)
                sync_layers_from_current()
                st.rerun()

        with st.expander("📏 Canvas Size (Extend/Pad)"):
            pad_top    = st.number_input("Pad Top",    0, 2000, 0)
            pad_bottom = st.number_input("Pad Bottom", 0, 2000, 0)
            pad_left   = st.number_input("Pad Left",   0, 2000, 0)
            pad_right  = st.number_input("Pad Right",  0, 2000, 0)
            pad_color  = st.color_picker("Pad Color", "#000000")
            if st.button("Apply Canvas Extend"):
                push_history(img)
                pr=int(pad_color[1:3],16); pg=int(pad_color[3:5],16); pb=int(pad_color[5:7],16)
                new_w2 = w + pad_left + pad_right
                new_h2 = h + pad_top + pad_bottom
                canvas = Image.new(img.mode, (new_w2, new_h2), (pr, pg, pb))
                canvas.paste(img, (pad_left, pad_top))
                st.session_state.current_image = canvas
                sync_layers_from_current()
                st.rerun()

        with st.expander("🪞 Perspective / Warp"):
            st.caption(f"Canvas: {w}×{h}")
            st.markdown("Source points → Destination points")
            src_pts = [[0,0],[w-1,0],[w-1,h-1],[0,h-1]]
            dst_pts_input = st.text_area("Dst Points (x,y per line)",
                                          f"0,0\n{w-1},0\n{w-1},{h-1}\n0,{h-1}",
                                          key="persp_pts")
            if st.button("Apply Perspective"):
                try:
                    dst = [list(map(int, p.strip().split(",")))
                           for p in dst_pts_input.strip().split("\n") if p.strip()]
                    if len(dst) == 4:
                        push_history(img)
                        st.session_state.current_image = perspective_transform(img, src_pts, dst)
                        sync_layers_from_current()
                        st.rerun()
                    else:
                        st.error("Need exactly 4 destination points")
                except Exception as e:
                    st.error(str(e))

        with st.expander("🔡 Auto-Straighten"):
            if st.button("Auto Straighten (Hough Lines)"):
                push_history(img)
                arr = img_to_cv(img)
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
                angle = 0
                if lines is not None:
                    angles = [l[0][1] for l in lines[:10]]
                    angle = (np.median(angles) - np.pi/2) * 180 / np.pi
                st.session_state.current_image = rotate_image(img, angle, expand=True)
                sync_layers_from_current()
                st.caption(f"Detected angle: {angle:.1f}°")
                st.rerun()

# ══════════════════════════════════════════════
# TAB: LAYERS
# ══════════════════════════════════════════════
with tab_layers:
    st.markdown("#### 🗂 Layer Manager")

    col_lb, col_lc = st.columns([3, 1])

    with col_lc:
        if st.button("➕ New Layer", use_container_width=True):
            add_layer(); st.rerun()
        if st.button("🔃 Flatten All", use_container_width=True):
            flatten_image(); st.rerun()
        if st.button("📋 Duplicate Active", use_container_width=True):
            idx = st.session_state.active_layer
            if st.session_state.layers:
                orig = st.session_state.layers[idx]
                dup = new_layer(f"{orig['name']} copy", orig["image"].copy())
                st.session_state.layers.insert(idx, dup)
                st.rerun()

    with col_lb:
        for i, layer in enumerate(st.session_state.layers):
            is_active = i == st.session_state.active_layer
            border = "2px solid #7c6af7" if is_active else "1px solid #2a2a34"
            st.markdown(f"""
            <div style="border:{border};border-radius:8px;padding:8px;
                        background:{'#202026' if is_active else '#18181c'};margin-bottom:6px;">
              <b>{'▶ ' if is_active else ''}{layer['name']}</b>
              {'👁' if layer['visible'] else '🚫'}
              <span style="color:#7c6af7;font-size:0.75rem;">
                {layer['blend']} | {layer['opacity']}%
              </span>
            </div>
            """, unsafe_allow_html=True)

            lcols = st.columns([1,1,1,1,1])
            with lcols[0]:
                if st.button("✓", key=f"sel_l{i}", help="Select", use_container_width=True):
                    st.session_state.active_layer = i; st.rerun()
            with lcols[1]:
                if st.button("👁", key=f"vis_l{i}", help="Toggle", use_container_width=True):
                    st.session_state.layers[i]["visible"] = not layer["visible"]; st.rerun()
            with lcols[2]:
                if st.button("🗑", key=f"del_l{i}", help="Delete", use_container_width=True):
                    delete_layer(i); st.rerun()
            with lcols[3]:
                if st.button("⬇", key=f"mgd_l{i}", help="Merge Down", use_container_width=True):
                    merge_down(i); st.rerun()
            with lcols[4]:
                new_op = st.number_input("Op%", 0, 100, layer["opacity"],
                                          key=f"op_l{i}", label_visibility="collapsed")
                if new_op != layer["opacity"]:
                    st.session_state.layers[i]["opacity"] = new_op; st.rerun()

    st.markdown("---")
    st.markdown("#### 🎭 Active Layer Blend Mode")
    if st.session_state.layers:
        idx = st.session_state.active_layer
        blend_modes = ["Normal","Multiply","Screen","Overlay","Soft Light","Hard Light",
                       "Difference","Exclusion","Color Dodge","Color Burn"]
        chosen = st.selectbox("Blend Mode", blend_modes,
                               index=blend_modes.index(st.session_state.layers[idx]["blend"])
                                     if st.session_state.layers[idx]["blend"] in blend_modes else 0,
                               key="layer_blend")
        if chosen != st.session_state.layers[idx]["blend"]:
            st.session_state.layers[idx]["blend"] = chosen; st.rerun()

# ══════════════════════════════════════════════
# TAB: HISTOGRAM
# ══════════════════════════════════════════════
with tab_histogram:
    st.markdown("#### 📊 Histogram & Image Statistics")
    img = st.session_state.current_image
    arr = np.array(img.convert("RGB"))

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        import streamlit as _st
        # Compute histograms
        channels = {"Red": arr[:,:,0], "Green": arr[:,:,1], "Blue": arr[:,:,2]}
        hist_data = {}
        for ch_name, ch_arr in channels.items():
            hist, _ = np.histogram(ch_arr.ravel(), bins=256, range=(0,255))
            hist_data[ch_name] = hist

        # Display as Streamlit chart
        import pandas as pd
        df = pd.DataFrame({
            "Red":   hist_data["Red"].astype(float),
            "Green": hist_data["Green"].astype(float),
            "Blue":  hist_data["Blue"].astype(float),
        }, index=range(256))
        df.index.name = "Value"
        st.line_chart(df, color=["#ff4444","#44ff44","#4444ff"])

        # Luminance histogram
        lum = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.uint8)
        lum_hist, _ = np.histogram(lum.ravel(), bins=256, range=(0,255))
        lum_df = pd.DataFrame({"Luminance": lum_hist.astype(float)}, index=range(256))
        st.area_chart(lum_df, color="#cccccc")

    with col_h2:
        st.markdown("**Image Statistics**")
        h, w = arr.shape[:2]

        stats = {
            "Dimensions": f"{w} × {h}px",
            "Total Pixels": f"{w*h:,}",
            "Megapixels": f"{w*h/1_000_000:.2f}MP",
            "Channels": arr.shape[2] if len(arr.shape) == 3 else 1,
        }
        for ch_name, ch_arr in channels.items():
            stats[f"{ch_name} Mean"]   = f"{ch_arr.mean():.1f}"
            stats[f"{ch_name} Std"]    = f"{ch_arr.std():.1f}"
            stats[f"{ch_name} Min/Max"]= f"{ch_arr.min()} / {ch_arr.max()}"

        for k, v in stats.items():
            c1, c2 = st.columns([2,3])
            with c1: st.caption(k)
            with c2: st.caption(str(v))

        st.markdown("---")
        st.markdown("**Clipping Analysis**")
        shadow_clip = float(np.mean(arr < 5) * 100)
        high_clip   = float(np.mean(arr > 250) * 100)
        st.caption(f"Shadow clipping: {shadow_clip:.1f}%")
        st.progress(min(shadow_clip/100, 1.0))
        st.caption(f"Highlight clipping: {high_clip:.1f}%")
        st.progress(min(high_clip/100, 1.0))

        # Color distribution pie-ish
        r_mean = float(arr[:,:,0].mean())
        g_mean = float(arr[:,:,1].mean())
        b_mean = float(arr[:,:,2].mean())
        total  = r_mean + g_mean + b_mean
        st.markdown("**Color Distribution**")
        st.caption(f"R: {r_mean/total*100:.1f}% | G: {g_mean/total*100:.1f}% | B: {b_mean/total*100:.1f}%")
        df_pie = pd.DataFrame({"Channel": ["Red","Green","Blue"], "Mean": [r_mean,g_mean,b_mean]})
        st.bar_chart(df_pie.set_index("Channel"), color=["#ff4444"])
