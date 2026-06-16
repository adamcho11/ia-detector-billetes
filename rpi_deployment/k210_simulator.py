#!/usr/bin/env python3
"""
K210 UART Simulator — Emula la transmision de features detectadas por la Maix Bit.
Lee imagenes reales, aplica degradaciones aleatorias (simulando fallos de la KPU),
comprime en JPG y transmite via UART con el protocolo especificado.

Protocolo:
    START:<num_features>\n
    <TIPO_FEATURE>:<size_bytes>\n<binary_jpg_buffer>
    ...

Uso:
    python k210_simulator.py --folder ./features --port /dev/ttyUSB0 --baud 115200
    python k210_simulator.py --folder ./features --virtual   # modo loopback local
"""

import os
import io
import time
import struct
import random
import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import serial
except ImportError:
    print("[WARN] pyserial no instalado. Solo modo --virtual disponible.")
    serial = None


# ===================== CONFIGURACION =====================
JPG_QUALITY = 85
FEATURE_TYPES = [
    "valor_10bs", "valor_20bs", "valor_50bs", "valor_100bs", "valor_200bs",
    "ir_10bs", "ir_20bs", "ir_50bs", "ir_100bs", "ir_200bs",
    "animal_10bs", "animal_20bs", "animal_50bs", "animal_100bs", "animal_200bs",
    "personaje_10bs", "personaje_20bs", "personaje_50bs", "personaje_100bs", "personaje_200bs",
    "serie_a",
]


# ===================== DEGRADACIONES =====================
def degrade_image(img):
    """
    Simula los fallos de deteccion de la KPU:
    - Desplazamiento aleatorio de pixeles (shift)
    - Recorte del 0-15% en bordes (crop)
    - Rotacion leve de hasta 10 grados
    - Ruido Gaussiano leve
    """
    h, w = img.shape[:2]

    # 1. Desplazamiento
    shift_x = random.randint(-8, 8)
    shift_y = random.randint(-8, 8)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Recorte parcial (simula bounding box incompleto)
    crop_pct = random.uniform(0, 0.15)
    crop_px = int(min(w, h) * crop_pct)
    if crop_px > 1:
        side = random.choice(['left', 'right', 'top', 'bottom', 'none'])
        if side == 'left':
            img = img[:, crop_px:]
        elif side == 'right':
            img = img[:, :-crop_px]
        elif side == 'top':
            img = img[crop_px:, :]
        elif side == 'bottom':
            img = img[:-crop_px, :]

    if img.size == 0:
        return None

    # 3. Rotacion leve
    angle = random.uniform(-10, 10)
    h2, w2 = img.shape[:2]
    center = (w2 // 2, h2 // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (w2, h2), borderMode=cv2.BORDER_REPLICATE)

    # 4. Ruido Gaussiano
    noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    return img


# ===================== EMPAQUETADO UART =====================
def encode_frame(img, feature_type):
    """Codifica la imagen en JPG y la empaqueta segun el protocolo UART."""
    success, jpg_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
    if not success:
        return None
    data = jpg_buf.tobytes()
    header = f"{feature_type}:{len(data)}\n".encode('ascii')
    return header + data


def encode_start(num_features):
    return f"START:{num_features}\n".encode('ascii')


# ===================== TRANSMISION =====================
def transmit_virtual(packets):
    """Modo virtual: imprime resumen de lo que se transmitiria."""
    total_bytes = 0
    for pkt in packets:
        total_bytes += len(pkt)
        text = pkt[:50]
        try:
            text = pkt.decode('ascii', errors='replace').rstrip()
        except Exception:
            text = f"<binary {len(pkt)} bytes>"
        print(f"  TX: {text}")
    print(f"\n[VIRTUAL] Transmitidos {len(packets)} paquetes, {total_bytes} bytes totales.")


def transmit_serial(ser, packets):
    """Modo real: envia por puerto serie."""
    for pkt in packets:
        ser.write(pkt)
        ser.flush()
        time.sleep(0.005)  # pausa minima entre paquetes
    print(f"[SERIAL] Transmitidos {len(packets)} paquetes.")


# ===================== MAIN =====================
def _detect_feature_type(path):
    """Extrae tipo de feature del path. Ej: .../valor_10bs/img_xxx.png -> valor_10bs"""
    known = ["valor_", "animal_", "ir_", "personaje_", "serie_a"]
    fname = path.stem
    for parent in path.parts:
        for prefix in known:
            if parent.startswith(prefix):
                return parent
    for prefix in known:
        if prefix in fname:
            idx = fname.find(prefix)
            end = fname.find('_', idx + len(prefix))
            if end == -1:
                end = len(fname)
            return fname[idx:end]
    return None


def main():
    parser = argparse.ArgumentParser(description="K210 UART Simulator")
    parser.add_argument("--folder", required=True, help="Carpeta con imagenes de features")
    parser.add_argument("--port", default=None, help="Puerto serie (ej: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--virtual", action="store_true", help="Modo loopback local sin hardware")
    parser.add_argument("--num-features", type=int, default=None,
                        help="Fuerza N features por billete. Si no, aleatorio entre 1 y 5.")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay entre billetes (seg)")
    args = parser.parse_args()

    # Cargar imagenes
    folder = Path(args.folder)
    image_paths = sorted(
        list(folder.rglob("*.png")) +
        list(folder.rglob("*.jpg")) +
        list(folder.rglob("*.bmp"))
    )
    if not image_paths:
        print(f"[ERROR] No se encontraron imagenes en {folder}")
        return

    print(f"[SIMULATOR] {len(image_paths)} imagenes cargadas de {folder}")

    # Configurar serial
    ser = None
    if args.port and not args.virtual:
        if serial is None:
            print("[ERROR] pyserial no instalado. Usa --virtual.")
            return
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"[SERIAL] Conectado a {args.port} @ {args.baud}")

    try:
        while True:
            # Seleccionar features aleatorias
            num = args.num_features if args.num_features else random.randint(1, 5)
            selected = random.sample(image_paths, min(num, len(image_paths)))

            packets = []

            for img_path in selected:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                degraded = degrade_image(img)
                if degraded is None:
                    continue

                # Detectar tipo de feature del path
                feature_type = _detect_feature_type(img_path)
                if feature_type is None:
                    feature_type = random.choice(FEATURE_TYPES)

                pkt = encode_frame(degraded, feature_type)
                if pkt is None:
                    continue

                packets.append(pkt)

            # START va al inicio
            start_pkt = encode_start(len(packets))
            packets.insert(0, start_pkt)

            if args.virtual:
                transmit_virtual(packets)
            else:
                transmit_serial(ser, packets)

            time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n[Interrumpido]")
    finally:
        if ser:
            ser.close()


if __name__ == "__main__":
    main()
