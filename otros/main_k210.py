"""
=============================================================================
IR-BillVerifier - Maix Bit (K210) Firmware Principal
=============================================================================
Hardware:
  - Motores DC: IO9 (ENA + ENB) via L298N, PWM 1500Hz (TIMER0)
  - Servo SG90 compuerta: IO10, PWM 50Hz (TIMER1)
  - Encoder RPM FC-03: IO19, GPIOHS0 con IRQ, PULL_DOWN
  - Buzzer: IO20, PWM (TIMER2)
  - TCRT5000 entrada/captura: removidos (control via encoder)
  - UART RPi: IO6 TX / IO7 RX, 115200 baud
  - RGB LED: IO15 R / IO16 G / IO17 B

Flujo:
  1. Espera flanco en encoder (Pin 19) -> alarma + cooldown 5s
  2. Servo baja (abre entrada)
  3. Motores DC avanzan 0.8s (primer jalado)
  4. Servo sube (cierra)
  5. Motores DC avanzan 1.1s (segundo jalado -> billete en zona captura)
  6. Clasifica denominacion + orientacion
  7. Libera clasificador, carga detector YOLO por denominacion
  8. Detecta features IR -> bounding boxes + etiquetas
  9. Recorta crops, transmite via UART a Raspberry Pi
 10. Espera respuesta de RPi (NO avanza hasta tener veredicto)
 11. Tercer avance de motores + compuerta segun veredicto + buzzer
 12. Si encoder se activa durante secuencia: +1 iteracion extra
=============================================================================
"""

import sensor
import image
import lcd
import time
from machine import UART, Timer, PWM
from fpioa_manager import fm
from Maix import GPIO
import KPU as kpu
import gc

# =============================================================================
# CONFIGURACION DE HARDWARE
# =============================================================================

# Pines K210 - Sistema de ingestion con motores DC y encoder RPM
PIN_RPM             = 19   # Encoder optico FC-03 (velocidad + presencia billete)
PIN_MOTORES         = 9    # PWM L298N ENA + ENB (ambos motores DC)
PIN_COMPUERTA       = 10   # PWM servo SG90 compuerta seleccion
PIN_LED_R           = 15
PIN_LED_G           = 16
PIN_LED_B           = 17
PIN_BUZZER          = 20   # Piezo buzzer

# UART para comunicacion con Raspberry Pi
UART_TX = 6
UART_RX = 7

# Configurar pines via FPIOA
fm.register(UART_TX, fm.fpioa.UART1_TX, force=True)
fm.register(UART_RX, fm.fpioa.UART1_RX, force=True)
fm.register(PIN_RPM, fm.fpioa.GPIOHS0, force=True)
uart = UART(UART.UART1, 115200, 8, 0, 0, timeout=1000, read_buf_len=256)

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
        raise ValueError("Sin modelo detector para {}".format(denom))
    
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
motores = None
compuerta = None
rpm_sensor = None
rpm_pulses = 0           # Contador de pulsos del encoder

def setup_actuators():
    global motores, compuerta
    # PWM 1500Hz para motores DC via L298N (TIMER0)
    tim_motores = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_PWM)
    motores = PWM(tim_motores, freq=1500, duty=0, pin=PIN_MOTORES)
    # PWM 50Hz para servo SG90 compuerta (TIMER1)
    tim_servo = Timer(Timer.TIMER1, Timer.CHANNEL0, mode=Timer.MODE_PWM)
    compuerta = PWM(tim_servo, freq=50, duty=7.5, pin=PIN_COMPUERTA)

def rollers_start(speed_pct=50):
    """Activa motores DC. speed_pct: 0-100% del duty cycle"""
    motores.duty(speed_pct)

def rollers_stop():
    """Detiene motores DC (duty 0%)"""
    motores.duty(0)

def compuerta_autentico():
    """Servo a 0° (billete autentico)"""
    compuerta.duty(2.5)

def compuerta_falso():
    """Servo a 180° (billete falso)"""
    compuerta.duty(12.5)

def compuerta_neutro():
    """Servo a 90° (posicion central)"""
    compuerta.duty(7.5)

# =============================================================================
# ENCODER RPM (sensor de velocidad y presencia de billete)
# =============================================================================

def rpm_callback(pin):
    """Interrupcion: cuenta cada pulso del encoder optico"""
    global rpm_pulses
    rpm_pulses += 1

def setup_rpm():
    """Configura el encoder RPM con interrupcion por flanco"""
    global rpm_sensor
    rpm_sensor = GPIO(GPIO.GPIOHS0, GPIO.IN, GPIO.PULL_DOWN)
    rpm_sensor.irq(rpm_callback, GPIO.IRQ_FALLING)

def bill_present(check_ms=200):
    """Verifica si hay billete: retorna True si hay pulsos recientes"""
    global rpm_pulses
    prev = rpm_pulses
    time.sleep_ms(check_ms)
    return rpm_pulses > prev

def get_rpm_speed():
    """Obtiene velocidad en pulsos por segundo (aproximado)"""
    global rpm_pulses
    prev = rpm_pulses
    time.sleep_ms(100)
    return (rpm_pulses - prev) * 10

# =============================================================================
# BUZZER
# =============================================================================

buzzer = None

def setup_buzzer():
    global buzzer
    tim_buzz = Timer(Timer.TIMER2, Timer.CHANNEL0, mode=Timer.MODE_PWM)
    buzzer = PWM(tim_buzz, freq=1000, duty=0, pin=PIN_BUZZER)

def alarma_detectado():
    """Alarma al detectar billete: 3 tonos alternados"""
    for _ in range(3):
        buzzer.freq(523)
        buzzer.duty(50)
        time.sleep(0.15)
        buzzer.freq(587)
        buzzer.duty(50)
        time.sleep(0.15)
    buzzer.duty(0)

def beep_aceptacion():
    """Beep corto agudo: billete autentico"""
    buzzer.freq(823)
    buzzer.duty(50)
    time.sleep_ms(80)
    buzzer.duty(0)

def beep_rechazo():
    """Tonos graves repetidos: billete falso"""
    for _ in range(2):
        buzzer.freq(300)
        buzzer.duty(50)
        time.sleep_ms(300)
        buzzer.duty(0)
        time.sleep_ms(100)

def beep_listo():
    """Beep sistema listo"""
    buzzer.freq(1000)
    buzzer.duty(50)
    time.sleep_ms(50)
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
    lcd.draw_string(10, 10, "Denom: {}".format(denom), lcd.BLACK, lcd.WHITE)
    color = lcd.GREEN if resultado == "REAL" else lcd.RED
    lcd.draw_string(10, 30, "{} ({}%)".format(resultado, int(score * 100) if score else 0), color, lcd.WHITE)

# =============================================================================
# SECUENCIA MECANICA DE INGESTION
# =============================================================================

def ejecutar_fase_ingestion():
    """
    Fase 1 y 2: ingesta del billete hasta posicion de captura.
    1. Servo baja (abre entrada)
    2. Primer avance motores 0.8s
    3. Servo sube (cierra)
    4. Segundo avance motores 1.1s (billete en zona captura)
    Retorna: True si el encoder detecto cambio durante la secuencia
    """
    global rpm_pulses
    rpm_antes = rpm_pulses

    # 1. Bajar servo (abrir compuerta de entrada)
    compuerta.duty(2.5)          # 0° = abierto
    time.sleep(0.22)
    time.sleep(0.2)

    # 2. Primer avance de motores (jalar billete)
    rollers_start(speed_pct=100)
    time.sleep(0.8)
    rollers_stop()
    time.sleep(0.5)

    # 3. Subir servo (cerrar compuerta)
    compuerta.duty(12.5)         # 180° = cerrado
    time.sleep(0.2)
    time.sleep(0.5)

    # 4. Segundo avance de motores (llevar billete a zona captura)
    rollers_start(speed_pct=100)
    time.sleep(1.1)
    rollers_stop()
    compuerta_neutro()

    # Verificar si hubo actividad en el encoder durante la secuencia
    return rpm_pulses > rpm_antes


def ejecutar_tercer_avance(resultado):
    """
    Fase 3: tercer avance de motores DESPUES del veredicto.
    Orientacion de la compuerta segun resultado.
    """
    if resultado == "REAL":
        compuerta_autentico()
    else:
        compuerta_falso()

    rollers_start(speed_pct=100)
    time.sleep(2.0)
    rollers_stop()


def ejecutar_fase_inferencia(classifier_task):
    """
    Pipeline completo: clasificar -> detectar -> enviar a RPi -> esperar respuesta.
    Retorna: (denom, resultado, score)
    """
    # --- CLASIFICACION ---
    denom, rotado, conf_clas = classify_bill(classifier_task, n_frames=3)
    print("Clasificacion: {} | Rotado: {} | Conf: {:.0f}%".format(denom, rotado, conf_clas * 100))

    # --- LIBERAR CLASIFICADOR ---
    kpu.deinit(classifier_task)
    gc.collect()

    # --- DETECCION YOLO ---
    detections = detect_features(denom)
    print("Detecciones: {}".format(len(detections)))
    for label, x1, y1, x2, y2, conf in detections:
        print("  {}: ({},{})-({},{}) conf={:.0f}%".format(label, x1, y1, x2, y2, conf * 100))

    lado = determine_side(detections)
    print("Lado: {}".format(lado))

    # --- RECORTAR CROPS ---
    img_raw = sensor.snapshot()
    crops = crop_features(img_raw, detections)

    # --- ENVIAR A RASPBERRY PI ---
    send_to_pi(denom, rotado, lado, crops)
    print("Datos enviados a Pi")

    # --- ESPERAR RESPUESTA ---
    resultado, score = receive_from_pi(timeout_ms=8000)
    print("Respuesta Pi: {} ({})".format(resultado, score))

    return denom, resultado, score


# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================

def main():
    print("IR-BillVerifier iniciando...")

    # Inicializar hardware
    init_camera()
    setup_actuators()
    setup_rpm()
    setup_buzzer()
    init_display()
    compuerta_neutro()

    show_status("SISTEMA LISTO\nEsperando billete...")
    print("====== SISTEMA ACTIVO - ESPERANDO ENCODER PIN 19 ======")

    # Estado previo del encoder para detectar flancos
    estado_anterior = rpm_sensor.value()

    # Cargar clasificador
    classifier_task = kpu.load(MODEL_CLASSIFIER)

    while True:
        # --- ESPERAR ACTIVACION DEL ENCODER (flanco) ---
        estado_actual = rpm_sensor.value()
        if estado_actual != estado_anterior:
            estado_anterior = estado_actual

            print("\n[!] Cambio detectado en encoder (Pin 19).")
            alarma_detectado()

            # Cooldown 5s
            show_status("Cooldown 5s...")
            print("Cooldown de 5 segundos activo...")
            time.sleep(5.0)

            beep_listo()
            print("====== INICIANDO SECUENCIA DE INGESTION ======")

            interrumpir = False

            while True:
                print("\n--- Iteracion del sistema ---")

                # === FASES 1 y 2: INGESTION MECANICA ===
                show_status("Ingestando\nbillete...")
                hubo_cambio = ejecutar_fase_ingestion()

                # === AQUI: DETECCION E INFERENCIA (despues del 2do avance) ===
                show_status("Clasificando...")
                sensor.run(1)
                denom, resultado, score = ejecutar_fase_inferencia(classifier_task)

                # Mostrar resultado en LCD
                if resultado:
                    show_result(denom, resultado, score)

                # === FASE 3: TERCER AVANCE (SOLO DESPUES DEL VEREDICTO) ===
                if resultado == "REAL":
                    show_status("REAL {}\nExpulsando...".format(denom))
                    ejecutar_tercer_avance("REAL")
                    beep_aceptacion()
                    print("VEREDICTO: AUTENTICO ({}%)".format(int(score * 100)))
                elif resultado == "FALSO":
                    show_status("FALSO {}\nExpulsando...".format(denom))
                    ejecutar_tercer_avance("FALSO")
                    beep_rechazo()
                    print("VEREDICTO: FALSO ({}%)".format(int(score * 100)))
                else:
                    show_status("ERROR\nReintentando...")
                    rollers_stop()
                    compuerta_neutro()

                # --- RECARGAR CLASIFICADOR ---
                gc.collect()
                classifier_task = kpu.load(MODEL_CLASSIFIER)

                # --- VERIFICAR SI HUBO OTRO CAMBIO EN EL ENCODER ---
                nuevo_estado = rpm_sensor.value()
                if nuevo_estado != estado_anterior:
                    print("\n[NUEVO] Cambio en encoder detectado durante la marcha!")
                    estado_anterior = nuevo_estado
                    interrumpir = True
                else:
                    interrumpir = False

                # Si hubo interrupcion, ejecutar una iteracion extra y salir
                if interrumpir:
                    print("Ejecutando iteracion adicional antes del paro...")
                    continue
                else:
                    break

            # Resetear al final del ciclo
            compuerta_neutro()
            rollers_stop()
            beep_listo()

            print("====== SISTEMA LISTO. ESPERANDO PROXIMO BILLETE ======")
            show_status("SISTEMA LISTO\nEsperando billete...")
            time.sleep(1.0)

        time.sleep_ms(20)  # Muestreo del encoder

# =============================================================================
if __name__ == "__main__":
    main()
