#!/usr/bin/env python3
"""
RPi Zero 2W — Receptor UART con OpenCV Random Forest + OCR Tesseract.
Sin dependencia de scikit-learn.

Pipeline:
- serie_a → Tesseract OCR (9 digitos + 1 letra)
- resto    → alineacion ORB+FLANN+Homografia → cv2.ml.RTrees
- Votacion estricta: TODAS las features deben ser AUTENTICO.

Uso:
    python3 rpi_receiver.py --port /dev/serial0 --templates ./templates/ --model rf_classifier.xml
"""

import io
import re
import time
import argparse
from collections import Counter

import cv2
import numpy as np

try:
    import serial
except ImportError:
    exit("[ERROR] pyserial no instalado.")
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("[WARN] pytesseract no instalado.")


TEMPLATE_SIZE = (100, 100)
HIST_BINS = 32
LAPLACIAN_BINS = 32

ORB_FEATURES = 2000
FLANN_CHECKS = 100
RANSAC_THRESHOLD = 5.0
MATCH_RATIO = 0.75

SERIAL_PATTERN = re.compile(r'^\d{9}[A-Z]$')


class MultiTemplateAligner:
    def __init__(self, templates_dir):
        self.templates = {}
        self.template_kp = {}
        self.template_des = {}
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=1, edgeThreshold=3)
        index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        search_params = dict(checks=FLANN_CHECKS)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        import os
        for f in os.listdir(templates_dir):
            if not f.endswith('.png'):
                continue
            ft = f.replace('.png', '')
            img = cv2.imread(os.path.join(templates_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, TEMPLATE_SIZE)
            self.templates[ft] = img
            kp, des = self.orb.detectAndCompute(img, None)
            self.template_kp[ft] = kp if kp else []
            self.template_des[ft] = des
        print(f"[ALIGNER] {len(self.templates)} templates")

    def align(self, img, feature_type):
        if feature_type not in self.templates or img is None or img.size == 0:
            return None
        img_kp, img_des = self.orb.detectAndCompute(img, None)
        if img_des is not None and len(img_des) >= 4:
            try:
                td = self.template_des.get(feature_type)
                if td is not None and len(td) >= 4:
                    matches = self.flann.knnMatch(td, img_des, k=2)
                    good = [m for m, n in matches if m.distance < MATCH_RATIO * n.distance]
                    if len(good) >= 4:
                        src_pts = np.float32([self.template_kp[feature_type][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([img_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, RANSAC_THRESHOLD)
                        if H is not None:
                            return cv2.warpPerspective(img, H, TEMPLATE_SIZE, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            except Exception:
                pass
        return cv2.resize(img, TEMPLATE_SIZE)


def extract_features(img):
    hist = cv2.calcHist([img], [0], None, [HIST_BINS], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.abs(lap).astype(np.uint8)
    lap_hist = cv2.calcHist([lap], [0], None, [LAPLACIAN_BINS], [0, 256])
    lap_hist = cv2.normalize(lap_hist, lap_hist).flatten()
    return np.concatenate([hist, lap_hist]).astype(np.float32)


def ocr_validate(jpg_bytes):
    if not HAS_TESSERACT:
        return True, "N/A"
    try:
        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, "decode"
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(
            img, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip().replace(' ', '').upper()
        return bool(SERIAL_PATTERN.match(text)), text
    except Exception as e:
        return False, str(e)[:30]


def classify_feature_from_crop(crop, feature_type, aligner, rf):
    """Clasifica un crop numpy (grayscale) con RF."""
    if crop is None or crop.size == 0:
        return False, 0
    aligned = aligner.align(crop, feature_type)
    if aligned is None:
        return False, 0
    vec = extract_features(aligned).reshape(1, -1).astype(np.float32)
    _, result = rf.predict(vec)
    return int(result[0]) == 1, int(result[0])


def classify_feature(jpg_bytes, feature_type, aligner, rf):
    """Clasifica desde JPG bytes (para compatibilidad con protocolo anterior)."""
    try:
        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, 0.0
    except Exception:
        return False, 0.0
    return classify_feature_from_crop(img, feature_type, aligner, rf)


class UARTFrameParser:
    """Parsea el protocolo: START:n\\n IMG:size\\n<jpg> BOX:label:x1:y1:x2:y2\\n ..."""
    def __init__(self):
        self.buffer = b""
        self.state = "WAIT_START"
        self.jpg_data = None

    def feed(self, data):
        self.buffer += data

    def try_parse(self):
        while True:
            if self.state == "WAIT_START":
                idx = self.buffer.find(b"START:")
                if idx == -1: return None
                self.buffer = self.buffer[idx:]
                nl = self.buffer.find(b"\n")
                if nl == -1: return None
                try:
                    self.num_features = int(self.buffer[len("START:"):nl].decode())
                except ValueError:
                    self.buffer = self.buffer[nl+1:]; continue
                self.buffer = self.buffer[nl+1:]
                self.state = "READ_IMG"
                continue

            elif self.state == "READ_IMG":
                # Buscar "IMG:" header
                idx = self.buffer.find(b"IMG:")
                if idx != 0:
                    # puede haber basura, buscar IMG: desde el inicio
                    idx2 = self.buffer.find(b"IMG:")
                    if idx2 == -1: return None
                    self.buffer = self.buffer[idx2:]
                nl = self.buffer.find(b"\n")
                if nl == -1: return None
                try:
                    img_size = int(self.buffer[4:nl].decode())
                except ValueError:
                    self.buffer = self.buffer[nl+1:]; continue
                data_start = nl + 1
                data_end = data_start + img_size
                if len(self.buffer) < data_end: return None
                self.jpg_data = self.buffer[data_start:data_end]
                self.buffer = self.buffer[data_end:]
                self.boxes = []
                self.state = "READ_BOXES"
                continue

            elif self.state == "READ_BOXES":
                while len(self.boxes) < self.num_features:
                    nl = self.buffer.find(b"\n")
                    if nl == -1: return None
                    line = self.buffer[:nl].decode('ascii', errors='ignore').strip()
                    self.buffer = self.buffer[nl+1:]
                    if line.startswith("BOX:"):
                        parts = line.split(":")
                        if len(parts) == 6:
                            tipo, x1, y1, x2, y2 = parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                            self.boxes.append((tipo, x1, y1, x2, y2))
                    else:
                        continue
                
                result = (self.jpg_data, list(self.boxes))
                self.jpg_data = None
                self.boxes = []
                self.state = "WAIT_START"
                return result

        return None


class BillValidator:
    def __init__(self, aligner, rf, ser):
        self.aligner = aligner
        self.rf = rf
        self.ser = ser
        self.parser = UARTFrameParser()

    def run(self):
        print("[RECEPTOR] Escuchando UART...")
        while True:
            try:
                if self.ser.in_waiting > 0:
                    self.parser.feed(self.ser.read(self.ser.in_waiting))
                features = self.parser.try_parse()
                if features is not None:
                    self._process(features)
            except serial.SerialException as e:
                print(f"[SERIAL ERR] {e}"); time.sleep(0.5)
            except Exception as e:
                print(f"[ERR] {e}"); time.sleep(0.1)

    def _process_stdin(self):
        """Procesa todo el buffer de stdin y sale."""
        self.parser.feed(self.ser.buf)
        while True:
            features = self.parser.try_parse()
            if features is None:
                break
            self._process(features)

    def _process(self, frame):
        """frame = (jpg_data, [(tipo, x1, y1, x2, y2), ...])"""
        jpg_data, boxes = frame
        num = len(boxes)
        print(f"\n[BILLETE] Recibidas {num} features")

        # Decodificar imagen IR completa
        img_full = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img_full is None:
            print("  [ERROR] No se pudo decodificar JPG")
            self.ser.write(b"RESULT:FALSE\n"); self.ser.flush()
            return

        results = []
        for tipo, x1, y1, x2, y2 in boxes:
            # Recortar crop desde la imagen IR
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_full.shape[1], x2), min(img_full.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                results.append((tipo, False, "bbox_invalid"))
                print(f"  {tipo}: FAIL (bbox invalido)")
                continue

            crop = img_full[y1:y2, x1:x2]

            if tipo == "serie_a":
                # OCR sobre el crop
                _, buf = cv2.imencode('.jpg', crop)
                ok, info = ocr_validate(buf.tobytes())
                results.append((tipo, ok, str(info)))
                print(f"  {tipo}: {'OK' if ok else 'FAIL'} serial={info}")
            else:
                ok, _ = classify_feature_from_crop(crop, tipo, self.aligner, self.rf)
                results.append((tipo, ok, ''))
                print(f"  {tipo}: {'OK' if ok else 'FAIL'}")

        all_ok = all(r[1] for r in results) and len(results) > 0
        v = "TRUE" if all_ok else "FALSE"
        self.ser.write(f"RESULT:{v}\n".encode()); self.ser.flush()
        print(f"  => RESULT:{v}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/serial0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--model", required=True)
    p.add_argument("--templates", required=True)
    p.add_argument("--stdin", action="store_true", help="Leer de stdin en vez de UART")
    a = p.parse_args()

    rf = cv2.ml.RTrees_load(a.model)
    print(f"[MODELO] {a.model}")

    aligner = MultiTemplateAligner(a.templates)

    if a.stdin:
        import sys
        data = sys.stdin.buffer.read()
        class StdinSerial:
            def __init__(self, buf):
                self.buf = buf
                self.pos = 0
                self.in_waiting = 1
            def read(self, n):
                r = self.buf[self.pos:self.pos + n]
                self.pos += len(r)
                self.in_waiting = max(0, len(self.buf) - self.pos)
                return r
            def write(self, data):
                pass
            def flush(self):
                pass
            def reset_input_buffer(self):
                pass
            def close(self):
                pass
        ser = StdinSerial(data)
        print("[STDIN] Modo lectura desde stdin")
        validator = BillValidator(aligner, rf, ser)
        validator._process_stdin()
    else:
        ser = serial.Serial(a.port, a.baud, timeout=0.05)
        ser.reset_input_buffer()
        print(f"[SERIAL] {a.port} @ {a.baud}")
        try:
            BillValidator(aligner, rf, ser).run()
        except KeyboardInterrupt:
            print("\n[STOP]")
        finally:
            ser.close()


if __name__ == "__main__":
    main()
