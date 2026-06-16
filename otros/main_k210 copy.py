"""
=============================================================================
IR-BillVerifier - Maix Bit (K210) Firmware Principal
=============================================================================
Flujo:
  1. Sensor TCRT5000 detecta billete → activa rodillos
  2. Captura 3 frames IR via OV2640
  3. Clasifica denominacion + orientacion (promedio de 3 frames)
  4. Libera memoria del clasificador
  5. Carga detector YOLO especifico por denominacion
  6. Detecta features IR → bounding boxes + etiquetas
  7. Determina lado del billete (anverso/reverso, lado A/B)
  8. Recorta crops detectados
  9. Transmite vía UART a Raspberry Pi:
     {denom, orientacion, lado, n_crops, [crop_data...]}
 10. Recibe resultado de Pi → activa servo seleccion + OLED + buzzer
=============================================================================
"""

import sensor
import image
import lcd
import time
from machine import UART, Timer, PWM
from fpioa_manager import fm
import KPU as kpu
import gc

# =============================================================================
# CONFIGURACION DE HARDWARE
# =============================================================================

# Pines K210 (ajustar segun conexion real)
PIN_LED_B           = 17

# UART para comunicacion con Raspberry Pi
UART_TX = 6
UART_RX = 7

# Configurar UART
fm.register(UART_TX, fm.fpioa.UART1_TX, force=True)
fm.register(UART_RX, fm.fpioa.UART1_RX, force=True)
uart = UART(UART1, baudrate=115200, timeout=1000, timeout_char=1000)

# =============================================================================
# MODELOS (rutas en SD)
# =============================================================================

MODEL_CLASSIFIER = "/sd/models/classifier_denom.kmodel"  # 10 clases

MODEL_DETECTORS = {
    "10_Bs":  "/sd/models/yolo_10bs.kmodel",
    "20_Bs":  "/sd/models/yolo_20bs.kmodel",
    "50_Bs":  "/sd/models/yolo_50bs.kmodel",
    "100_Bs": "/sd/models/yolo_100bs.kmodel",
    "200_Bs": "/sd/models/yolo_200bs.kmodel",
}

# Etiquetas del clasificador (deben coincidir con orden de entrenamiento)
CLASSIFIER_LABELS = [
    "10_Bs_normal", "10_Bs_rotado",
    "20_Bs_normal", "20_Bs_rotado",
    "50_Bs_normal", "50_Bs_rotado",
    "100_Bs_normal", "100_Bs_rotado",
    "200_Bs_normal", "200_Bs_rotado",
]

# Etiquetas del detector YOLO (21 clases, mismo orden que data.yaml)
YOLO_LABELS = [
    "animal_100bs", "animal_10bs", "animal_200bs", "animal_20bs", "animal_50bs",
    "personaje_100bs", "personaje_10bs", "personaje_200bs", "personaje_20bs", "personaje_50bs",
    "serie_a",
    "valor_100bs", "valor_10bs", "valor_200bs", "valor_20bs", "valor_50bs",
    "valor_ir_100bs", "valor_ir_10bs", "valor_ir_200bs", "valor_ir_20bs", "valor_ir_50bs",
]

# Features que determinan el lado del billete
FEATURES_ANVERSO_A = {"serie_a", "personaje_100bs", "personaje_10bs", "personaje_200bs", "personaje_20bs", "personaje_50bs", "valor_100bs", "valor_10bs", "valor_200bs", "valor_20bs", "valor_50bs"}
FEATURES_ANVERSO_B = {"valor_ir_100bs", "valor_ir_10bs", "valor_ir_200bs", "valor_ir_20bs", "valor_ir_50bs"}
FEATURES_REVERSO   = {"animal_100bs", "animal_10bs", "animal_200bs", "animal_20bs", "animal_50bs"}

# =============================================================================
# FUNCIONES DE CAMARA Y CAPTURA
# =============================================================================

def init_camera():
    """Inicializa la OV2640 en modo IR (escala de grises)"""
    sensor.reset()
    sensor.set_pixformat(sensor.GRAYSCALE)
    sensor.set_framesize(sensor.QVGA)  # 320x240
    sensor.set_vflip(False)
    sensor.set_hmirror(False)
    sensor.skip_frames(30)
    sensor.run(1)

# =============================================================================
# CLASIFICACION (3 frames → promedio)
# =============================================================================

def classify_bill(model_kpu, n_frames=3):
    """
    Captura n_frames, clasifica cada uno y promedia probabilidades.
    Retorna: (denominacion_str, es_rotado_bool, confianza_promedio)
    """
    accum = [0.0] * len(CLASSIFIER_LABELS)
    
    for _ in range(n_frames):
        img = sensor.snapshot()
        # Preprocesar para el modelo (el modelo espera 224x224)
        img_resized = img.resize(224, 224)
        img_rgb = img_resized.to_rgb565()  # si el modelo espera RGB565
        
        # Inferencia
        features = kpu.forward(model_kpu, img_rgb)
        # features es una lista de probabilidades softmax
        for i, prob in enumerate(features):
            accum[i] += prob
    
    # Promediar
    probs = [v / n_frames for v in accum]
    best_idx = probs.index(max(probs))
    label = CLASSIFIER_LABELS[best_idx]
    
    # Parsear: "10_Bs_normal" → denominacion="10_Bs", rotado=False
    denom, orient = label.rsplit("_", 1)
    es_rotado = (orient == "rotado")
    confianza = probs[best_idx]
    
    return denom, es_rotado, confianza

# =============================================================================
# DETECCION YOLO
# =============================================================================

def detect_features(denom):
    """
    Carga el modelo YOLO para la denominacion, ejecuta deteccion,
    retorna lista de (label, x1, y1, x2, y2, confianza).
    """
    model_path = MODEL_DETECTORS.get(denom)
    if model_path is None:
        raise ValueError(f"Sin modelo detector para {denom}")
    
    # Cargar modelo YOLO
    task = kpu.load(model_path)
    anchors = [1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375, 3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875]
    kpu.init_yolo2(task, 0.5, 0.3, 5, anchors)
    
    img = sensor.snapshot()
    
    # Si el billete esta rotado, enderezar la imagen antes de detectar
    img_rot = img.copy()
    # NOTA: La orientacion se determino en classify_bill()
    # Si es_rotado=True, hay que rotar 180° la imagen antes de detectar
    
    objects = kpu.run_yolo2(task, img_rot)
    
    detections = []
    if objects:
        for obj in objects:
            label = YOLO_LABELS[obj.classid()]
            x1, y1, x2, y2 = obj.x(), obj.y(), obj.x() + obj.w(), obj.y() + obj.h()
            conf = obj.value()
            detections.append((label, x1, y1, x2, y2, conf))
    
    kpu.deinit(task)
    return detections

# =============================================================================
# DETERMINAR LADO DEL BILLETE
# =============================================================================

def determine_side(detections):
    """
    Analiza las etiquetas detectadas para determinar el lado.
    Retorna: "anverso_A", "anverso_B", o "reverso"
    """
    labels = {d[0] for d in detections}
    
    has_reverso = bool(labels & FEATURES_REVERSO)
    has_anverso_a = bool(labels & FEATURES_ANVERSO_A)
    has_anverso_b = bool(labels & FEATURES_ANVERSO_B)
    
    if has_reverso:
        return "reverso"
    elif has_anverso_a:
        return "anverso_A"
    elif has_anverso_b:
        return "anverso_B"
    else:
        # Por defecto, si solo tiene valor_ir y nada mas
        return "anverso_B"

# =============================================================================
# RECORTE DE CROPS
# =============================================================================

def crop_features(img, detections):
    """
    Recorta las regiones detectadas de la imagen.
    Retorna lista de (label, bytes_png_crop)
    """
    crops = []
    for label, x1, y1, x2, y2, conf in detections:
        crop = img.copy(roi=(x1, y1, x2 - x1, y2 - y1))
        # Comprimir a PNG en memoria
        crop_data = crop.compress(quality=50)  # JPEG para reducir tamano
        crops.append((label, bytes(crop_data)))
    return crops

# =============================================================================
# PROTOCOLO DE COMUNICACION UART → RASPBERRY PI
# =============================================================================

"""
Formato del paquete (binario):
  Byte 0:     0xAA (inicio de trama)
  Byte 1:     denominacion (0=10, 1=20, 2=50, 3=100, 4=200)
  Byte 2:     orientacion (0=normal, 1=rotado)
  Byte 3:     lado (0=anverso_A, 1=anverso_B, 2=reverso)
  Byte 4-5:   n_crops (uint16 big-endian)
  Para cada crop:
    Byte 0:     len_label (uint8)
    Bytes 1-N:  label (ASCII)
    Byte N+1-2: crop_width (uint16)
    Byte N+3-4: crop_height (uint16)
    Bytes N+5-M: crop_data (JPEG bytes)
  Byte final:  0xBB (fin de trama)
"""

DENOM_MAP = {"10_Bs": 0, "20_Bs": 1, "50_Bs": 2, "100_Bs": 3, "200_Bs": 4}
SIDE_MAP  = {"anverso_A": 0, "anverso_B": 1, "reverso": 2}

def send_to_pi(denom, rotado, lado, crops):
    """Transmite datos a la Raspberry Pi via UART"""
    import struct
    
    # Construir paquete
    buf = bytearray()
    buf.append(0xAA)                          # start
    buf.append(DENOM_MAP.get(denom, 0xFF))    # denominacion
    buf.append(1 if rotado else 0)            # orientacion
    buf.append(SIDE_MAP.get(lado, 0xFF))      # lado
    buf.extend(struct.pack(">H", len(crops))) # n_crops
    
    for label, crop_bytes in crops:
        label_bytes = label.encode("ascii")
        buf.append(len(label_bytes))
        buf.extend(label_bytes)
        buf.extend(struct.pack(">H", len(crop_bytes)))
        buf.extend(crop_bytes)
    
    buf.append(0xBB)  # end
    
    # Enviar en chunks si es necesario (UART buffer limitado)
    CHUNK = 256
    for i in range(0, len(buf), CHUNK):
        uart.write(buf[i:i+CHUNK])
        time.sleep_ms(5)

def receive_from_pi(timeout_ms=5000):
    """
    Espera respuesta de la Pi.
    Retorna: (resultado_str, score_float) o (None, None) si timeout
    """
    start = time.ticks_ms()
    resp = bytearray()
    
    while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
        if uart.any():
            b = uart.read(1)
            if b[0] == 0xCC:  # inicio de respuesta
                resp = bytearray()
                continue
            elif b[0] == 0xDD:  # fin de respuesta
                break
            else:
                resp.extend(b)
        else:
            time.sleep_ms(10)
    
    if len(resp) == 0:
        return None, None
    
    # Decodificar: "resultado:score" → "REAL:0.85"
    try:
        text = resp.decode("ascii")
        resultado, score_str = text.split(":")
        return resultado, float(score_str)
    except:
        return "ERROR", 0.0

# =============================================================================
# CONTROL DE ACTUADORES
# =============================================================================

# NOTA: Inicializar PWMs en setup()
rodillo_1 = None
rodillo_2 = None
compuerta = None

def setup_actuators():
    global rodillo_1, rodillo_2, compuerta
    # PWM 50Hz para servos
    rodillo_1 = PWM(PWM.TIMER0, freq=50, duty=0, pin=PIN_RODILLO_1)
    rodillo_2 = PWM(PWM.TIMER0, freq=50, duty=0, pin=PIN_RODILLO_2)
    compuerta = PWM(PWM.TIMER1, freq=50, duty=0, pin=PIN_COMPUERTA)

def rollers_start(speed_pct=50):
    """Activa rodillos. speed_pct: 0-100"""
    duty = 7.5 + (speed_pct / 100.0) * 2.5  # 5-10% duty = velocidad
    rodillo_1.duty(duty)
    rodillo_2.duty(duty)

def rollers_stop():
    rodillo_1.duty(7.5)  # neutro para servo continuo
    rodillo_2.duty(7.5)

def compuerta_autentico():
    compuerta.duty(2.5)  # 0°

def compuerta_falso():
    compuerta.duty(12.5) # 90°

def compuerta_neutro():
    compuerta.duty(7.5)  # 45°

# =============================================================================
# BUZZER
# =============================================================================

buzzer = None

def setup_buzzer(pin):
    global buzzer
    buzzer = PWM(PWM.TIMER2, freq=0, duty=0, pin=pin)

def beep_aceptacion():
    buzzer.freq(1000)
    buzzer.duty(50)
    time.sleep_ms(200)
    buzzer.duty(0)

def beep_rechazo():
    buzzer.freq(300)
    buzzer.duty(50)
    time.sleep_ms(500)
    buzzer.duty(0)

# =============================================================================
# OLED / LCD
# =============================================================================

def init_display():
    lcd.init()
    lcd.clear(lcd.WHITE)

def show_status(text, color=lcd.BLACK):
    lcd.clear(lcd.WHITE)
    lcd.draw_string(10, 10, text, color, lcd.WHITE)

def show_result(denom, resultado, score):
    lcd.clear(lcd.WHITE)
    lcd.draw_string(10, 10, f"Denom: {denom}", lcd.BLACK, lcd.WHITE)
    color = lcd.GREEN if resultado == "REAL" else lcd.RED
    lcd.draw_string(10, 30, f"{resultado} ({score:.1%})", color, lcd.WHITE)

# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================

def main():
    print("IR-BillVerifier iniciando...")
    
    # Inicializar hardware
    init_camera()
    setup_actuators()
    init_display()
    # setup_buzzer(20)  # pin del buzzer
    
    show_status("IR-BillVerifier\nLISTO")
    
    # Cargar clasificador una vez (se recarga si es necesario)
    classifier_task = kpu.load(MODEL_CLASSIFIER)
    
    while True:
        # --- ESPERAR BILLETE ---
        # while GPIO(PIN_SENSOR_ENTRADA).value() == 0:
        #     time.sleep_ms(100)
        
        show_status("Procesando...")
        
        # --- ACTIVAR RODILLOS ---
        # rollers_start(speed_pct=50)
        # while GPIO(PIN_SENSOR_CAPTURA).value() == 0:
        #     time.sleep_ms(50)
        # rollers_stop()
        # time.sleep_ms(200)  # estabilizar
        
        # --- FASE 1: CLASIFICACION (3 frames, promedio) ---
        denom, rotado, conf_clas = classify_bill(classifier_task, n_frames=3)
        print(f"Clasificacion: {denom} | Rotado: {rotado} | Conf: {conf_clas:.2%}")
        
        # --- LIBERAR CLASIFICADOR ---
        kpu.deinit(classifier_task)
        gc.collect()
        
        # --- FASE 2: DETECCION ---
        detections = detect_features(denom)
        print(f"Detecciones: {len(detections)}")
        for label, x1, y1, x2, y2, conf in detections:
            print(f"  {label}: ({x1},{y1})-({x2},{y2}) conf={conf:.2%}")
        
        # --- DETERMINAR LADO ---
        lado = determine_side(detections)
        print(f"Lado: {lado}")
        
        # --- RECORTAR CROPS ---
        img_raw = sensor.snapshot()
        crops = crop_features(img_raw, detections)
        
        # --- ENVIAR A RASPBERRY PI ---
        send_to_pi(denom, rotado, lado, crops)
        print("Datos enviados a Pi")
        
        # --- ESPERAR RESPUESTA ---
        resultado, score = receive_from_pi(timeout_ms=8000)
        print(f"Respuesta Pi: {resultado} (score={score})")
        
        # --- ACTUAR ---
        if resultado == "REAL":
            compuerta_autentico()
            beep_aceptacion()
        elif resultado == "FALSO":
            compuerta_falso()
            beep_rechazo()
        else:
            compuerta_neutro()
        
        # Mostrar resultado
        if resultado:
            show_result(denom, resultado, score)
        
        time.sleep_ms(1000)
        compuerta_neutro()
        
        # --- RECARGAR CLASIFICADOR PARA SIGUIENTE BILLETE ---
        gc.collect()
        classifier_task = kpu.load(MODEL_CLASSIFIER)
        show_status("IR-BillVerifier\nLISTO")

# =============================================================================
if __name__ == "__main__":
    main()
