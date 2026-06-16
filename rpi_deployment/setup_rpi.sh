#!/bin/bash
# setup_rpi.sh — Configura la Raspberry Pi Zero 2W para el receptor UART

echo "=== Configurando RPi Zero 2W ==="

# 1. Actualizar sistema
sudo apt update && sudo apt upgrade -y

# 2. Instalar dependencias del sistema
sudo apt install -y python3-pip python3-opencv tesseract-ocr

# 3. Instalar paquetes Python
pip3 install -r requirements.txt

# 4. Habilitar UART
echo "Habilitando UART..."
sudo raspi-config nonint do_serial 2

# 5. Deshabilitar Bluetooth (libera UART0)
echo "dtoverlay=disable-bt" | sudo tee -a /boot/config.txt

echo ""
echo "=== Configuracion completa ==="
echo ""
echo "Reinicia la RPi: sudo reboot"
echo ""
echo "Para iniciar el receptor:"
echo "  python3 rpi_receiver.py --port /dev/serial0 --model rf_classifier.pkl --templates ./templates/"
