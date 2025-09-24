#!/bin/bash
# Fashion MNIST Deep Learning Project Setup
# Скрипт для налаштування проекту

echo "Fashion MNIST Deep Learning Project - Setup"
echo "==========================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Project files:"
echo "- fashion_mnist_deep_learning.py    # Головний Python скрипт"
echo "- Fashion_MNIST_Deep_Learning.ipynb # Jupyter Notebook"
echo "- fashion_mnist_demo.py             # Демо-версія (без інтернету)"
echo "- requirements.txt                  # Python залежності"
echo "- README.md                         # Документація"
echo "- setup.sh                          # Цей файл"

echo ""
echo "Команди для запуску:"
echo "1. Повна версія (потребує інтернет):"
echo "   python fashion_mnist_deep_learning.py"
echo ""
echo "2. Jupyter Notebook:"
echo "   jupyter notebook Fashion_MNIST_Deep_Learning.ipynb"
echo ""
echo "3. Демо-версія (без інтернету):"
echo "   python fashion_mnist_demo.py"
echo ""
echo "4. Google Colab:"
echo "   Завантажте Fashion_MNIST_Deep_Learning.ipynb до Colab"

echo ""
echo "Налаштування завершено!"
echo "Детальну інформацію дивіться у README.md"