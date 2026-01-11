#!/bin/bash
# Install OCR dependencies

echo "Installing EasyOCR (recommended)..."
pip install easyocr

# Alternative: Install doctr
# echo "Installing doctr..."
# pip install python-doctr[torch]

# Alternative: Install PaddleOCR
# echo "Installing PaddleOCR..."
# pip install paddlepaddle-gpu paddleocr

echo "Done! OCR package installed."
echo ""
echo "Test OCR with:"
echo "  python -c \"import easyocr; print('EasyOCR OK')\""
