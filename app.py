#!/usr/bin/env python3
"""Team 07 - TOXICITY Web Application.

A lightweight, zero-dependency local web interface for interactive Turkish toxicity detection.
Uses Python's standard library http.server, HTML5, CSS3, and JavaScript.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure src/ is in sys.path so toxicity_detection and pickled preprocessors are importable
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("toxicity_web_app")

# Model path
MODEL_PATH = BASE_DIR / "models" / "baselines" / "selected_baseline" / "pipeline.joblib"
MODEL_CONFIG_PATH = BASE_DIR / "models" / "baselines" / "selected_baseline" / "config.json"

PIPELINE = None
FEATURE_NAMES = None
COEFFICIENTS = None


def load_model():
    """Load the trained baseline pipeline into memory."""
    global PIPELINE, FEATURE_NAMES, COEFFICIENTS
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    logger.info("Loading model from %s...", MODEL_PATH)
    PIPELINE = joblib.load(MODEL_PATH)
    feature_union = PIPELINE.named_steps["features"]
    classifier = PIPELINE.named_steps["classifier"]
    FEATURE_NAMES = feature_union.get_feature_names_out()
    COEFFICIENTS = classifier.coef_[0]
    logger.info("Model loaded successfully (%d features).", len(FEATURE_NAMES))


def analyze_text(text: str) -> dict:
    """Analyze a single Turkish text and compute prediction, probability, and top cues."""
    if not text or not text.strip():
        return {
            "text": "",
            "label": 0,
            "label_name": "TEMİZ",
            "prob_toxic": 0.0,
            "prob_clean": 1.0,
            "cues": [],
        }

    clean_text = text.strip()
    prob_toxic = float(PIPELINE.predict_proba([clean_text])[0][1])
    prob_clean = 1.0 - prob_toxic
    pred_label = int(PIPELINE.predict([clean_text])[0])

    # Extract top contributing features (Explainability / Cues)
    feature_union = PIPELINE.named_steps["features"]
    feat_vec = feature_union.transform([clean_text])
    nonzero_indices = feat_vec.nonzero()[1]

    cues = []
    for idx in nonzero_indices:
        val = feat_vec[0, idx]
        weight = COEFFICIENTS[idx]
        contrib = float(val * weight)
        raw_feat = FEATURE_NAMES[idx]
        
        # Clean up feature name display (remove word__ / char__ prefix)
        feat_type = "Kelime" if raw_feat.startswith("word__") else "Karakter"
        clean_feat = raw_feat.replace("word__", "").replace("char__", "")
        
        cues.append({
            "feature": clean_feat,
            "type": feat_type,
            "impact": contrib,
            "direction": "toxic" if contrib > 0 else "clean",
        })

    # Sort cues by absolute impact
    cues.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "text": clean_text,
        "label": pred_label,
        "label_name": "TOKSİK" if pred_label == 1 else "TEMİZ",
        "prob_toxic": round(prob_toxic, 4),
        "prob_clean": round(prob_clean, 4),
        "cues": cues[:8],  # Top 8 most influential cues
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Team 07 – Türkçe Toksisite Analiz Arayüzü</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-indigo: #6366f1;
            --toxic-red: #ef4444;
            --clean-green: #10b981;
            --hover-bg: #2d3b55;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px 20px;
        }

        .container {
            width: 100%;
            max-width: 900px;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        .badge-team {
            display: inline-block;
            background: linear-gradient(135deg, #4f46e5, #06b6d4);
            color: white;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
            padding: 5px 14px;
            border-radius: 20px;
            margin-bottom: 12px;
            text-transform: uppercase;
        }

        h1 {
            font-size: 28px;
            font-weight: 800;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #f8fafc 40%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 14px;
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 10px;
        }

        textarea {
            width: 100%;
            height: 110px;
            background: #0f172a;
            border: 1.5px solid var(--card-border);
            border-radius: 12px;
            color: #fff;
            padding: 14px;
            font-size: 15px;
            line-height: 1.5;
            resize: vertical;
            transition: all 0.2s ease;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }

        .presets {
            margin-top: 14px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }

        .presets span {
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: 600;
            margin-right: 4px;
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-secondary);
            font-size: 12px;
            padding: 6px 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .preset-btn:hover {
            background: rgba(56, 189, 248, 0.15);
            border-color: var(--accent-blue);
            color: #fff;
        }

        .action-row {
            display: flex;
            justify-content: flex-end;
            margin-top: 18px;
            gap: 12px;
        }

        .btn-clear {
            background: transparent;
            border: 1px solid var(--card-border);
            color: var(--text-secondary);
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: 0.15s;
        }

        .btn-clear:hover {
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
        }

        .btn-analyze {
            background: linear-gradient(135deg, #0284c7, #2563eb);
            border: none;
            color: #fff;
            padding: 12px 28px;
            font-size: 15px;
            font-weight: 700;
            border-radius: 10px;
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
            transition: all 0.2s ease;
        }

        .btn-analyze:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .result-card {
            display: none;
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--card-border);
            margin-bottom: 20px;
        }

        .badge-status {
            font-size: 16px;
            font-weight: 800;
            padding: 8px 18px;
            border-radius: 30px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .status-clean {
            background: rgba(16, 185, 129, 0.15);
            color: var(--clean-green);
            border: 1.5px solid var(--clean-green);
        }

        .status-toxic {
            background: rgba(239, 68, 68, 0.15);
            color: var(--toxic-red);
            border: 1.5px solid var(--toxic-red);
        }

        .prob-section {
            margin-bottom: 24px;
        }

        .prob-label {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .progress-bar-bg {
            height: 14px;
            background: #0f172a;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            border: 1px solid var(--card-border);
        }

        .progress-bar-fill {
            height: 100%;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .cues-title {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .cue-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }

        .cue-item {
            background: #0f172a;
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
        }

        .cue-impact-toxic {
            color: var(--toxic-red);
            font-weight: 700;
        }

        .cue-impact-clean {
            color: var(--clean-green);
            font-weight: 700;
        }

        .info-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px dashed var(--card-border);
            border-radius: 12px;
            padding: 16px 20px;
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .info-card strong {
            color: #fff;
        }

        footer {
            margin-top: 20px;
            font-size: 12px;
            color: var(--text-secondary);
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge-team">GTU CSE 484 / 654 NLP · Team 07</div>
            <h1>Türkçe Toksisite ve Örtük Saldırı Tespit Modeli</h1>
            <p class="subtitle">Kelime + Karakter TF-IDF & Lojistik Regresyon (147,749 Boyutlu Doğrulanmış Model)</p>
        </header>

        <div class="card">
            <label for="text-input">Analiz Edilecek Türkçe Metin:</label>
            <textarea id="text-input" placeholder="Buraya bir yorum, tweet veya cümle yazın ya da aşağıdaki örneklerden birini seçin..."></textarea>

            <div class="presets">
                <span>Hızlı Örnekler:</span>
                <button class="preset-btn" onclick="setPreset('Bugün hava çok güzel, projede herkese kolaylıklar ve başarılar dilerim.')">🟢 Temiz Günlük</button>
                <button class="preset-btn" onclick="setPreset('Sen ne iğrenç ve karaktersiz bir insansın, senden nefret ediyorum.')">🔴 Açık Hakaret</button>
                <button class="preset-btn" onclick="setPreset('Değil unutmak acısı hafifleyenin kanı kurusun.')">⚠️ Örtük Beddua (kanı kurusun)</button>
                <button class="preset-btn" onclick="setPreset('Adı üzerinde lağım medyası bunlardan başka ne beklenir ki.')">🔴 Metaforik (lağım medyası)</button>
                <button class="preset-btn" onclick="setPreset('Hiç birimiz aptal veya cahil değiliz, sadece sabrediyoruz.')">🟢 Yadsınan Hakaret (aptal değiliz)</button>
                <button class="preset-btn" onclick="setPreset('İnsanoğlu naziktir, ağır sözü kaldırmaz gönül incinir.')">🟢 Atasözü (ağır söz)</button>
            </div>

            <div class="action-row">
                <button class="btn-clear" onclick="clearInput()">Temizle</button>
                <button class="btn-analyze" onclick="runAnalysis()">Analiz Et</button>
            </div>
        </div>

        <div id="result-card" class="card result-card">
            <div class="result-header">
                <div>
                    <h2 style="font-size: 18px; margin-bottom: 4px;">Analiz Sonucu</h2>
                    <span style="font-size: 13px; color: var(--text-secondary);" id="res-timestamp">Model: tfidf_combined_c10.0</span>
                </div>
                <div id="badge-status" class="badge-status status-clean">
                    <span id="badge-icon">🟢</span>
                    <span id="badge-text">TEMİZ</span>
                </div>
            </div>

            <div class="prob-section">
                <div class="prob-label">
                    <span>Toksisite Olasılığı: <strong id="prob-toxic-val">0.0%</strong></span>
                    <span>Temiz Olasılığı: <strong id="prob-clean-val">100.0%</strong></span>
                </div>
                <div class="progress-bar-bg">
                    <div id="progress-bar-fill" class="progress-bar-fill" style="width: 0%; background: var(--clean-green);"></div>
                </div>
            </div>

            <div class="cues-title">
                🔍 Kararı En Çok Etkileyen İpuçları (Top Cues / Karakter & Kelime Ağırlıkları):
            </div>
            <div id="cue-grid" class="cue-grid">
                <!-- Cues inserted dynamically -->
            </div>
        </div>

        <div class="info-card">
            <strong>💡 Bilimsel Metodoloji Notu:</strong> Bu arayüz, $N = 76,381$ veri ile eğitilmiş ve $N = 8,485$ kilitli test setinde %93.93 Makro-F1 skoru elde etmiş olan nihai <strong>Kelime (1-2) + Karakter (3-5) Kombine TF-IDF</strong> modelini çalıştırmaktadır. <em>"Değil unutmak acısı hafifleyenin kanı kurusun"</em> gibi örtük beddualar küfür içermediği için n-gram modelleri tarafından düşük olasılık alırken, derin BERTurk modeli bunları %100 doğrulukla yakalamaktadır.
        </div>

        <footer>
            Gebze Teknik Üniversitesi · Doğal Dil İşleme Dersi Projesi · Kasım 2026
        </footer>
    </div>

    <script>
        function setPreset(text) {
            document.getElementById('text-input').value = text;
            runAnalysis();
        }

        function clearInput() {
            document.getElementById('text-input').value = '';
            document.getElementById('result-card').style.display = 'none';
        }

        async function runAnalysis() {
            const text = document.getElementById('text-input').value.trim();
            if (!text) {
                alert('Lütfen analiz edilecek bir metin girin.');
                return;
            }

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) {
                    throw new Error('Analiz hatası: ' + response.statusText);
                }

                const data = await response.json();
                renderResult(data);
            } catch (err) {
                alert('Hata oluştu: ' + err.message);
            }
        }

        function renderResult(data) {
            const card = document.getElementById('result-card');
            const badgeStatus = document.getElementById('badge-status');
            const badgeIcon = document.getElementById('badge-icon');
            const badgeText = document.getElementById('badge-text');
            const probToxicVal = document.getElementById('prob-toxic-val');
            const probCleanVal = document.getElementById('prob-clean-val');
            const progressBarFill = document.getElementById('progress-bar-fill');
            const cueGrid = document.getElementById('cue-grid');

            const isToxic = data.label === 1;
            const toxicPct = (data.prob_toxic * 100).toFixed(1);
            const cleanPct = (data.prob_clean * 100).toFixed(1);

            badgeText.textContent = isToxic ? 'TOKSİK' : 'TEMİZ';
            badgeIcon.textContent = isToxic ? '🔴' : '🟢';

            if (isToxic) {
                badgeStatus.className = 'badge-status status-toxic';
                progressBarFill.style.background = 'var(--toxic-red)';
            } else {
                badgeStatus.className = 'badge-status status-clean';
                progressBarFill.style.background = 'var(--clean-green)';
            }

            probToxicVal.textContent = `%${toxicPct}`;
            probCleanVal.textContent = `%${cleanPct}`;
            progressBarFill.style.width = `${toxicPct}%`;

            cueGrid.innerHTML = '';
            if (data.cues && data.cues.length > 0) {
                data.cues.forEach(cue => {
                    const item = document.createElement('div');
                    item.className = 'cue-item';
                    const isPushToxic = cue.impact > 0;
                    const sign = isPushToxic ? '+' : '';
                    const impactClass = isPushToxic ? 'cue-impact-toxic' : 'cue-impact-clean';
                    
                    item.innerHTML = `
                        <div>
                            <strong>${escapeHtml(cue.feature)}</strong>
                            <div style="font-size: 11px; color: var(--text-secondary);">${cue.type}</div>
                        </div>
                        <span class="${impactClass}">${sign}${cue.impact.toFixed(3)}</span>
                    `;
                    cueGrid.appendChild(item);
                });
            } else {
                cueGrid.innerHTML = '<div style="color: var(--text-secondary); font-size: 13px;">Belirgin bir n-gram eşleşmesi bulunamadı.</div>';
            }

            card.style.display = 'block';
            card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Allow Ctrl+Enter or Cmd+Enter to submit
        document.getElementById('text-input').addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                runAnalysis();
            }
        });
    </script>
</body>
</html>
"""


class ToxicityRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler providing REST API and single-page web app."""

    def log_message(self, format, *args):
        # Override to keep logs clean
        logger.info("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), format % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "model_loaded": PIPELINE is not None}).encode("utf-8"))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)

            try:
                payload = json.loads(post_body.decode("utf-8"))
                text = payload.get("text", "")
                result = analyze_text(text)

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            except Exception as exc:
                logger.error("Prediction error: %s", exc)
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint Not Found")


def run_server(port: int = 5000, open_browser: bool = True):
    """Start the HTTP server on the given port."""
    load_model()
    server_address = ("", port)

    try:
        httpd = ThreadingHTTPServer(server_address, ToxicityRequestHandler)
    except OSError:
        # Port already in use, fallback to next available
        port = port + 1
        server_address = ("", port)
        httpd = ThreadingHTTPServer(server_address, ToxicityRequestHandler)

    url = f"http://localhost:{port}"
    logger.info("=================================================================")
    logger.info(" Türkçe Toksisite Analiz Web Sunucusu Başlatıldı!")
    logger.info(" Arayüze tarayıcınızdan erişmek için adrese gidin:")
    logger.info("   -> %s", url)
    logger.info(" Durdurmak için terminalde Ctrl+C tuşlarına basın.")
    logger.info("=================================================================")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nSunucu kapatılıyor...")
        httpd.server_close()
        logger.info("Sunucu güvenli bir şekilde durduruldu.")


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_server(port=port_arg, open_browser=True)

