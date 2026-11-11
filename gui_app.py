#!/usr/bin/env python3
"""Team 07 - TOXICITY Desktop Application (Tkinter GUI).

Native macOS / Cross-platform Desktop GUI for Turkish toxicity detection.
Zero extra dependencies, runs directly on Python 3.11 with Tkinter.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

# Ensure src/ is in sys.path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib

MODEL_PATH = BASE_DIR / "models" / "baselines" / "selected_baseline" / "pipeline.joblib"


class ToxicityApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Team 07 – Türkçe Toksisite Analiz Arayüzü")
        self.root.geometry("780x620")
        self.root.minsize(650, 500)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Load Model
        self.load_model()
        self.create_widgets()

    def load_model(self):
        try:
            self.pipeline = joblib.load(MODEL_PATH)
            feature_union = self.pipeline.named_steps["features"]
            classifier = self.pipeline.named_steps["classifier"]
            self.feature_names = feature_union.get_feature_names_out()
            self.coefficients = classifier.coef_[0]
        except Exception as e:
            messagebox.showerror("Model Yükleme Hatası", f"Model yüklenirken hata oluştu:\n{e}")
            sys.exit(1)

    def create_widgets(self):
        # Header Frame
        header = ttk.Frame(self.root, padding="15 15 15 10")
        header.pack(fill=tk.X)

        title = ttk.Label(
            header,
            text="GTU CSE 484 / 654 NLP · Team 07",
            font=("Helvetica", 11, "bold"),
            foreground="#4f46e5",
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            header,
            text="Türkçe Örtük ve Açık Toksisite Tespit Modeli (TF-IDF + Lojistik Regresyon)",
            font=("Helvetica", 14, "bold"),
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # Main Input Frame
        input_frame = ttk.LabelFrame(self.root, text=" Metin Girişi ", padding="12")
        input_frame.pack(fill=tk.BOTH, expand=False, padx=15, pady=5)

        self.text_input = tk.Text(input_frame, height=5, font=("Helvetica", 13), wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Presets Frame
        preset_frame = ttk.Frame(input_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(preset_frame, text="Hızlı Örnekler: ", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)

        presets = [
            ("Temiz", "Bugün hava çok güzel, projede herkese kolaylıklar ve başarılar dilerim."),
            ("Açık Hakaret", "Sen ne iğrenç ve karaktersiz bir insansın, senden nefret ediyorum."),
            ("Örtük Beddua", "Değil unutmak acısı hafifleyenin kanı kurusun."),
            ("Metaforik", "Adı üzerinde lağım medyası bunlardan başka ne beklenir ki."),
            ("Yadsınan Hakaret", "Hiç birimiz aptal veya cahil değiliz, sadece sabrediyoruz."),
        ]

        for label, txt in presets:
            btn = ttk.Button(preset_frame, text=label, command=lambda t=txt: self.set_text(t))
            btn.pack(side=tk.LEFT, padx=3)

        # Buttons Row
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        clear_btn = ttk.Button(btn_frame, text="Temizle", command=self.clear_text)
        clear_btn.pack(side=tk.RIGHT, padx=(5, 0))

        analyze_btn = ttk.Button(btn_frame, text="Analiz Et", command=self.analyze)
        analyze_btn.pack(side=tk.RIGHT)

        # Result Frame
        self.result_frame = ttk.LabelFrame(self.root, text=" Değerlendirme ve Analiz Sonucu ", padding="15")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Status Label (Clean / Toxic)
        self.status_label = tk.Label(
            self.result_frame,
            text="Metin yazıp 'Analiz Et' butonuna tıklayın.",
            font=("Helvetica", 15, "bold"),
            fg="#64748b",
        )
        self.status_label.pack(anchor="w", pady=(0, 5))

        # Probability Progress
        self.prob_label = ttk.Label(self.result_frame, text="Toksisite Olasılığı: -", font=("Helvetica", 11))
        self.prob_label.pack(anchor="w", pady=(0, 5))

        self.progress = ttk.Progressbar(self.result_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(0, 10))

        # Cues Table
        ttk.Label(self.result_frame, text="En Etkili Öznitelik İpuçları (Top Cues):", font=("Helvetica", 11, "bold")).pack(anchor="w")

        self.cue_list = ttk.Treeview(self.result_frame, columns=("feature", "type", "impact"), show="headings", height=5)
        self.cue_list.heading("feature", text="Öznitelik / Kelime")
        self.cue_list.heading("type", text="Türü")
        self.cue_list.heading("impact", text="Ağırlık Etkisi")
        self.cue_list.column("feature", width=300)
        self.cue_list.column("type", width=120)
        self.cue_list.column("impact", width=150)
        self.cue_list.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    def set_text(self, text: str):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert(tk.END, text)
        self.analyze()

    def clear_text(self):
        self.text_input.delete("1.0", tk.END)
        self.status_label.config(text="Metin yazıp 'Analiz Et' butonuna tıklayın.", fg="#64748b")
        self.prob_label.config(text="Toksisite Olasılığı: -")
        self.progress["value"] = 0
        for item in self.cue_list.get_children():
            self.cue_list.delete(item)

    def analyze(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Uyarı", "Lütfen analiz edilecek bir metin girin.")
            return

        prob_toxic = float(self.pipeline.predict_proba([text])[0][1])
        pred_label = int(self.pipeline.predict([text])[0])

        pct = prob_toxic * 100
        self.progress["value"] = pct

        if pred_label == 1:
            self.status_label.config(text=f"🔴 TOKSİK (Etiket: 1)", fg="#dc2626")
            self.prob_label.config(text=f"Toksisite Olasılığı: %{pct:.2f} (Temiz Olasılığı: %{100 - pct:.2f})")
        else:
            self.status_label.config(text=f"🟢 TEMİZ (Etiket: 0)", fg="#16a34a")
            self.prob_label.config(text=f"Toksisite Olasılığı: %{pct:.2f} (Temiz Olasılığı: %{100 - pct:.2f})")

        # Cues calculation
        feature_union = self.pipeline.named_steps["features"]
        feat_vec = feature_union.transform([text])
        nonzero_indices = feat_vec.nonzero()[1]

        cues = []
        for idx in nonzero_indices:
            val = feat_vec[0, idx]
            weight = self.coefficients[idx]
            contrib = float(val * weight)
            raw_feat = self.feature_names[idx]
            feat_type = "Kelime (Word)" if raw_feat.startswith("word__") else "Karakter (Char)"
            clean_feat = raw_feat.replace("word__", "").replace("char__", "")
            cues.append((clean_feat, feat_type, contrib))

        cues.sort(key=lambda x: abs(x[2]), reverse=True)

        for item in self.cue_list.get_children():
            self.cue_list.delete(item)

        for feat, ftype, imp in cues[:8]:
            sign = "+" if imp > 0 else ""
            self.cue_list.insert("", tk.END, values=(feat, ftype, f"{sign}{imp:.4f}"))


def main():
    root = tk.Tk()
    app = ToxicityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

