"""
╔══════════════════════════════════════════════════════════════╗
║   Cervical Cancer Cytology Classifier — Web App              ║
║   Hosted on Hugging Face Spaces (Gradio SDK)                 ║
║   MSB7216: Deep Learning for Health                          ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gradio as gr

# ── 1. Configuration ─────────────────────────────────────────────
CLASS_NAMES  = ['cancer', 'lesion', 'normal', 'others']
NUM_CLASSES  = 4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model files sit next to app.py in the Space repo
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models")

CKPT_PATHS = {
    "EfficientNet_V2_S ✦ Best Model (87.08% Val Acc)": os.path.join(MODEL_DIR, "efficientnet_best.pth"),
    "ResNet18 (83.08% Val Acc)":                        os.path.join(MODEL_DIR, "resnet18_best.pth"),
}

# Clinical risk metadata per class
RISK_META = {
    "cancer":  ("#ef4444", "🔴 HIGH RISK — Malignant Cells Detected",
                 "Invasive carcinoma features detected. Urgent specialist referral recommended."),
    "lesion":  ("#f97316", "🟠 MODERATE RISK — Precancerous Lesion",
                 "Squamous intraepithelial lesion (SIL). Colposcopy follow-up recommended."),
    "normal":  ("#22c55e", "🟢 LOW RISK — Normal Cytology",
                 "No abnormal cellular morphology detected. Routine screening schedule maintained."),
    "others":  ("#eab308", "🟡 INDETERMINATE — Artifact / Inflammation",
                 "Background debris or inflammatory cells. A repeat smear may be required."),
}

# ── 2. Inference Transform ────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── 3. Model Builders ─────────────────────────────────────────────
def build_resnet18():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

def build_efficientnet():
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m

# ── 4. Cached model loader ────────────────────────────────────────
_cache: dict = {}

def get_model(name: str) -> nn.Module:
    if name in _cache:
        return _cache[name]
    path = CKPT_PATHS[name]
    model = build_efficientnet() if "Efficient" in name else build_resnet18()
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    _cache[name] = model
    return model

# Pre-load both models at startup so first inference is instant
for _n in CKPT_PATHS:
    try:
        get_model(_n)
        print(f"✓ Loaded: {_n}")
    except Exception as e:
        print(f"✗ Could not load {_n}: {e}")

# ── 5. Inference Function ─────────────────────────────────────────
def classify(image: Image.Image, model_name: str):
    if image is None:
        return {}, "<p style='color:#64748b;text-align:center;padding:20px'>Upload an image to begin.</p>"

    try:
        model = get_model(model_name)
    except Exception as e:
        err = f"""<div style='background:#fee2e2;border:2px solid #ef4444;border-radius:10px;padding:16px'>
                  <b>⚠️ Model Load Error</b><br><code>{e}</code></div>"""
        return {}, err

    img = image.convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()

    confidences  = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    top_idx      = int(np.argmax(probs))
    top_class    = CLASS_NAMES[top_idx]
    top_conf     = probs[top_idx] * 100
    color, label, note = RISK_META[top_class]

    risk_html = f"""
    <div style="background:linear-gradient(135deg,{color}18,{color}08);
                border:2px solid {color};border-radius:14px;padding:20px 24px;
                font-family:'Inter',sans-serif;">
      <div style="font-size:20px;font-weight:800;color:{color};margin-bottom:8px">{label}</div>
      <div style="font-size:15px;color:#e2e8f0;margin-bottom:6px">
        Predicted class: <b>{top_class.upper()}</b> &nbsp;|&nbsp;
        Confidence: <b>{top_conf:.1f}%</b>
      </div>
      <div style="font-size:13px;color:#94a3b8;margin-bottom:12px">{note}</div>
      <hr style="border:none;border-top:1px solid {color}55;margin:10px 0">
      <div style="font-size:11px;color:#64748b">
        ⚠️ <b>Research use only.</b> This tool does not replace professional cytopathological diagnosis.
        All findings must be reviewed by a qualified clinician before any clinical decision is made.
      </div>
    </div>"""

    return confidences, risk_html

# ── 6. Example Images (shown on the landing page) ─────────────────
EXAMPLE_DIR = os.path.join(BASE_DIR, "examples")
examples = []
if os.path.isdir(EXAMPLE_DIR):
    for f in sorted(os.listdir(EXAMPLE_DIR)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            examples.append([os.path.join(EXAMPLE_DIR, f),
                              list(CKPT_PATHS.keys())[0]])

# ── 7. Gradio Interface ───────────────────────────────────────────
DESCRIPTION = """
<div style="text-align:center;max-width:700px;margin:0 auto 8px">
  <p style="color:#94a3b8;font-size:14px;line-height:1.7">
    Upload a conventional Pap smear cytology tile to classify it into one of four diagnostic
    categories using deep learning models trained on the CPSMI2025 cytology dataset.<br>
    <b style="color:#c4b5fd">ResNet18</b> (85% test accuracy · Macro F1 0.79) &nbsp;|&nbsp;
    <b style="color:#60a5fa">EfficientNet_V2_S</b> (85% test accuracy · Macro F1 0.78)
  </p>
</div>
"""

with gr.Blocks(
    title="Cervical Cytology AI Classifier",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
    css="""
    /* ── Global ─────────────────────────────── */
    body, .gradio-container { background: #0d0d1a !important; }
    .gradio-container { max-width: 1000px !important; margin: auto; }

    /* ── Header ─────────────────────────────── */
    #header {
        background: linear-gradient(135deg, #4c1d9530, #1e3a5f30);
        border: 1px solid #7c3aed44;
        border-radius: 18px;
        padding: 32px;
        text-align: center;
        margin-bottom: 4px;
    }
    #header h1 {
        font-size: 30px;
        font-weight: 900;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px;
    }
    #header p.sub {
        color: #64748b;
        font-size: 13px;
        margin: 0 0 14px;
    }
    .badge {
        display: inline-block;
        background: #7c3aed22;
        border: 1px solid #7c3aed55;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 12px;
        color: #c4b5fd;
        margin: 3px;
    }

    /* ── Panels ──────────────────────────────── */
    .panel {
        background: #12122200;
        border: 1px solid #1e1b4b;
        border-radius: 14px;
        padding: 20px;
    }

    /* ── Class Guide ─────────────────────────── */
    #guide {
        background: #0f0f2a;
        border: 1px solid #1e1b4b;
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 14px;
    }
    #guide h3 { color: #a78bfa; font-size: 14px; margin: 0 0 14px; }
    .class-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .class-card {
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.5;
    }
    .class-card b { display: block; font-size: 13px; margin-bottom: 4px; }
    .cancer-card  { background:#ef444412; border:1px solid #ef444444; }
    .lesion-card  { background:#f9731612; border:1px solid #f9731644; }
    .normal-card  { background:#22c55e12; border:1px solid #22c55e44; }
    .others-card  { background:#eab30812; border:1px solid #eab30844; }
    """,
) as demo:

    # ── Header ──────────────────────────────────────────────────
    gr.HTML("""
    <div id="header">
      <h1>🔬 Cervical Cytology AI Classifier</h1>
      <p class="sub">MSB7216: Deep Learning for Health &nbsp;·&nbsp; Patricia Mystica Apio</p>
      <span class="badge">ResNet18</span>
      <span class="badge">EfficientNet_V2_S</span>
      <span class="badge">PyTorch 2.0</span>
      <span class="badge">CPSMI2025 Dataset</span>
      <span class="badge">4-Class Classification</span>
    </div>
    """)

    gr.HTML(DESCRIPTION)

    # ── Main Content ─────────────────────────────────────────────
    with gr.Row(equal_height=True):

        # Left — Inputs
        with gr.Column(scale=1, elem_classes="panel"):
            image_in = gr.Image(
                type="pil",
                label="Upload Pap Smear Cytology Image",
                height=280,
                sources=["upload", "clipboard"],
            )
            model_dd = gr.Dropdown(
                choices=list(CKPT_PATHS.keys()),
                value=list(CKPT_PATHS.keys())[0],
                label="Select Model Architecture",
                info="EfficientNet_V2_S is recommended (highest validation accuracy)",
            )
            run_btn = gr.Button("🔍 Classify Image", variant="primary", size="lg")

        # Right — Outputs
        with gr.Column(scale=1, elem_classes="panel"):
            risk_out  = gr.HTML(
                value="<p style='color:#475569;text-align:center;padding:30px 0'>"
                      "Upload a cytology image to see the clinical assessment.</p>"
            )
            label_out = gr.Label(num_top_classes=4, label="Class Probability Distribution")

    # ── Class Reference Guide ─────────────────────────────────────
    gr.HTML("""
    <div id="guide">
      <h3>📋 Classification Reference Guide</h3>
      <div class="class-grid">
        <div class="class-card cancer-card">
          <b style="color:#ef4444">🔴 cancer</b>
          Invasive squamous cell carcinoma. Marked nuclear enlargement,
          hyperchromatic nuclei, high nucleus-to-cytoplasm (N:C) ratio.
        </div>
        <div class="class-card lesion-card">
          <b style="color:#f97316">🟠 lesion</b>
          Low/high-grade squamous intraepithelial lesions (LSIL/HSIL).
          Precancerous dysplastic changes — koilocytic vacuoles, perinuclear halos.
        </div>
        <div class="class-card normal-card">
          <b style="color:#22c55e">🟢 normal</b>
          Normal superficial and intermediate squamous cells.
          Small pyknotic nucleus, abundant flat cytoplasm.
        </div>
        <div class="class-card others-card">
          <b style="color:#eab308">🟡 others</b>
          Background artifacts, atrophy, or inflammatory cells
          (neutrophils, bacteria) — not fitting main diagnostic categories.
        </div>
      </div>
    </div>
    """)

    # ── Example Images ────────────────────────────────────────────
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[image_in, model_dd],
            outputs=[label_out, risk_out],
            fn=classify,
            cache_examples=True,
            label="Example Cytology Images",
        )

    # ── Footer ────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px;font-size:11px;color:#334155">
      <b>⚠️ Research Use Only</b> — This system is not a certified medical device.
      Results must not be used as the sole basis for any clinical decision.<br>
      MSB7216 Deep Learning for Health &nbsp;·&nbsp; Makerere University
    </div>
    """)

    # ── Event Bindings ────────────────────────────────────────────
    run_btn.click(fn=classify, inputs=[image_in, model_dd],
                  outputs=[label_out, risk_out])
    image_in.change(fn=classify, inputs=[image_in, model_dd],
                    outputs=[label_out, risk_out])

if __name__ == "__main__":
    demo.launch()
