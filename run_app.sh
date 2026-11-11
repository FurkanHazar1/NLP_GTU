#!/usr/bin/env bash
# ==============================================================================
# Team 07 – TOXICITY Web Arayüzü Başlatıcı
# ==============================================================================

# Script dizinine geç
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Sanal ortam python belirleme
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "Hata: Python bulunamadı!"
    exit 1
fi

echo "=========================================================="
echo " Team 07 – Türkçe Toksisite Analiz Arayüzü Başlatılıyor..."
echo " Python Yolu: $PYTHON_BIN"
echo "=========================================================="

export PYTHONPATH="src:$PYTHONPATH"
export MPLBACKEND="Agg"
export MPLCONFIGDIR="/tmp/matplotlib"

"$PYTHON_BIN" app.py "$@"

