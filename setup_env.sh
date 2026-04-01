#!/usr/bin/env bash
# setup_env.sh — Erstellt und aktiviert die Conda-Umgebung für ellipsoid-fit
set -e

ENV_NAME="ellipsoid-fit"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  ellipsoid-fit  —  Environment Setup"
echo "========================================"

# 1. Prüfen ob conda verfügbar ist
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda nicht gefunden. Bitte Miniconda/Anaconda installieren:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 2. Bestehende Umgebung entfernen (optional)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Umgebung '${ENV_NAME}' existiert bereits."
    read -rp "  Neu erstellen? (j/N): " RECREATE
    if [[ "$RECREATE" =~ ^[jJ]$ ]]; then
        echo "[INFO] Entferne alte Umgebung..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "[INFO] Bestehende Umgebung wird verwendet."
        echo ""
        echo "Aktivieren mit:  conda activate ${ENV_NAME}"
        echo "Starten mit:     python main.py"
        exit 0
    fi
fi

# 3. Umgebung aus environment.yml erstellen
echo "[INFO] Erstelle Umgebung aus environment.yml ..."
conda env create -f "${SCRIPT_DIR}/environment.yml"

echo ""
echo "========================================"
echo "  Setup abgeschlossen!"
echo "========================================"
echo ""
echo "Umgebung aktivieren:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "App starten:"
echo "  python main.py"
echo ""
echo "Benchmark ausführen:"
echo "  python benchmark_sdf.py"
echo ""
