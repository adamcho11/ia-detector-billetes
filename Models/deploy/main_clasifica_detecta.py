"""
Maix Bit — Clasifica billete, luego deteccion de features segun denominacion.
"""

import sensor, image, lcd, time
import KPU as kpu
import gc, sys
from fpioa_manager import fm
from machine import UART, Timer, PWM
from Maix import GPIO

# ===================== HARDWARE =====================
PIN_LED_BLANCO = 2
fm.register(PIN_LED_BLANCO, fm.fpioa.TIMER1_TOGGLE1, force=True)
PIN_LED_IR = 6
fm.register(PIN_LED_IR, fm.fpioa.TIMER2_TOGGLE1, force=True)

tim1 = Timer(Timer.TIMER1, Timer.CHANNEL0, mode=Timer.MODE_PWM)
pwm_blanco = PWM(tim1, freq=5000, duty=100, pin=PIN_LED_BLANCO)
tim2 = Timer(Timer.TIMER2, Timer.CHANNEL0, mode=Timer.MODE_PWM)
pwm_ir = PWM(tim2, freq=5000, duty=0, pin=PIN_LED_IR)

PIN_BUTTON = 9
fm.register(PIN_BUTTON, fm.fpioa.GPIOHS0, force=True)
button = GPIO(GPIO.GPIOHS0, GPIO.IN, GPIO.PULL_UP)

# ===================== CLASIFICACION =====================
CLASSIFY_ADDR = 0x300000
CLASSIFY_LABELS = ['10', '20r', '50', '50r', '100', '100r', '200', '200r', '10r', '20']
SENSOR_WINDOW = (224, 224)


def parse_denomination(label):
    return label.rstrip('r')


# ===================== DETECCION (desde SD) =====================
DETECT_MODELS = {
    '10': {
        'model': '/sd/model-10rgb.kmodel',
        'labels': ['serie_a', 'valor_10bs', 'animal_10bs', 'ir_10bs', 'personaje_10bs'],
        'anchors': [0.5, 1.22, 2.34, 1.53, 1.13, 0.87, 1.39, 1.22, 0.84, 0.91],
    },
    '20': {
        'model': '/sd/model-20rgb.kmodel',
        'labels': ['serie_a', 'personaje_20bs', 'valor_20bs', 'animal_20bs', 'ir_20bs'],
        'anchors': [1.31, 0.84, 3.88, 1.22, 3.19, 0.69, 1.31, 0.97, 2.31, 1.75],
    },
    '50': {
        'model': '/sd/model-50rgb.kmodel',
        'labels': ['animal_50bs', 'ir_50bs', 'personaje_50bs', 'serie_a', 'valor_50bs'],
        'anchors': [3.12, 3.31, 2.5, 1.69, 3.19, 0.69, 2.25, 2.69, 1.28, 0.88],
    },
    '100': {
        'model': '/sd/model-100rgb.kmodel',
        'labels': ['serie_a', 'ir_100bs', 'valor_100bs', 'animal_100bs', 'personaje_100bs'],
        'anchors': [2.81, 1.91, 3.12, 0.69, 3.0, 0.59, 1.06, 0.88, 1.56, 0.88],
    },
    '200': {
        'model': '/sd/model-200rgb.kmodel',
        'labels': ['personaje_200bs', 'serie_a', 'valor_200bs', 'ir_200bs', 'animal_200bs'],
        'anchors': [1.13, 0.91, 3.19, 1.75, 1.69, 0.91, 3.09, 0.59, 3.22, 0.69],
    },
}


# ===================== UART =====================
class Comm:
    def __init__(self, uart):
        self.uart = uart

    def send_detect_result(self, objects, labels):
        msg = ""
        for obj in objects:
            pos = obj.rect()
            p = obj.value()
            idx = obj.classid()
            if idx >= len(labels):
                continue
            label = labels[idx]
            msg += "{}:{}:{}:{}:{}:{:.2f}:{}, ".format(pos[0], pos[1], pos[2], pos[3], idx, p, label)
        if msg:
            msg = msg[:-2] + "\n"
        self.uart.write(msg.encode())


def init_uart():
    fm.register(15, fm.fpioa.UART1_TX, force=True)
    fm.register(16, fm.fpioa.UART1_RX, force=True)
    return UART(UART.UART1, 76800, 8, None, 1, timeout=50, read_buf_len=1024)


def lcd_show_except(e):
    import uio
    err_str = uio.StringIO()
    sys.print_exception(e, err_str)
    err_str = err_str.getvalue()
    img = image.Image(size=(224, 224))
    img.draw_string(0, 10, err_str, scale=1, color=(0xff, 0x00, 0x00))
    lcd.display(img)


# ===================== CAMARA =====================
def init_camera():
    try:
        sensor.run(0)
    except Exception:
        pass
    time.sleep_ms(20)
    for _ in range(3):
        try:
            sensor.reset()
            break
        except Exception:
            time.sleep_ms(10)
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing(SENSOR_WINDOW)
    sensor.set_hmirror(False)
    sensor.set_vflip(False)
    sensor.run(1)
    sensor.skip_frames(10)


# ===================== EXPOSICION FIJA =====================
EXP_W  = 110000
GAIN_W = 1.0
EXP_IR = 110000
GAIN_IR = 1.0


# ===================== CLASIFICACION =====================
def classificar(task, img):
    fmap = kpu.forward(task, img)
    plist = list(fmap[:])
    max_val = max(plist)
    max_idx = plist.index(max_val)
    return max_idx, max_val


# ===================== DETECCION =====================
def bucle_deteccion(task, det_labels, anchors, img):
    kpu.init_yolo2(task, 0.5, 0.3, 5, anchors)
    t = time.ticks_ms()
    objects = kpu.run_yolo2(task, img)
    t = time.ticks_ms() - t
    saved_boxes = []
    if objects:
        for obj in objects:
            pos = obj.rect()
            cls_id = obj.classid()
            if cls_id < len(det_labels):
                saved_boxes.append((pos, cls_id, obj.value()))
    return saved_boxes, t


def capturar_y_enviar(saved_boxes, det_labels, t, denom, cls_conf, comm):
    global rpi_connected
    pwm_blanco.duty(0)
    pwm_ir.duty(40)
    sensor.set_auto_exposure(True)
    sensor.set_auto_gain(True)

    sensor.run(1)
    for _ in range(15):
        sensor.snapshot()
        time.sleep_ms(20)
    gc.collect()
    img_ir = sensor.snapshot()       # enviar B&N a la RPi
    pwm_ir.duty(0)
    sensor.set_auto_exposure(False, exposure_us=EXP_W)
    sensor.set_auto_gain(False, gain_db=GAIN_W)
    sensor.run(0)

    num_boxes = len(saved_boxes)
    if num_boxes > 0:
        img_ir.compress(quality=40)
        comm.uart.write("START:{}\n".format(num_boxes).encode())
        comm.uart.write("IMG:{}\n".format(img_ir.size()).encode())
        comm.uart.write(img_ir)

        for pos, cls_id, conf in saved_boxes:
            label = det_labels[cls_id]
            x1, y1, w, h = pos[0], pos[1], pos[2], pos[3]
            x2, y2 = x1 + w, y1 + h
            comm.uart.write("BOX:{}:{}:{}:{}:{}\n".format(
                label, x1, y1, x2, y2).encode())

        # Mostrar pantalla de carga mientras espera RPi
        img_wait = image.Image(size=SENSOR_WINDOW)
        img_wait.draw_string(40, 100, "Procesando...", scale=2, color=(255, 255, 255))
        img_wait.draw_string(60, 140, "%s Bs" % denom, scale=2, color=(0, 255, 0))
        lcd.display(img_wait)

        # Esperar respuesta RPi
        t_start = time.ticks_ms()
        respuesta = ""
        while time.ticks_ms() - t_start < 5000:
            if comm.uart.any():
                data = comm.uart.read()
                try: respuesta += data.decode()
                except: respuesta += str(data)
                if "\n" in respuesta: break
            time.sleep_ms(50)

        respuesta = respuesta.strip()
        es_autentico = "TRUE" in respuesta
        rpi_connected = True  # RPi respondio, marcar como conectada

    # Dibujar en LCD
    pwm_ir.duty(50)
    sensor.run(1)
    time.sleep_ms(30)
    img_display = sensor.snapshot()   # IR se ve B&N en vez de violeta
    pwm_ir.duty(0)
    sensor.run(0)
    sensor.set_auto_exposure(False, exposure_us=EXP_W)
    sensor.set_auto_gain(False, gain_db=GAIN_W)

    for pos, cls_id, conf in saved_boxes:
        img_display.draw_rectangle(pos)
        img_display.draw_string(pos[0], pos[1],
            "%s : %.2f" % (det_labels[cls_id], conf),
            scale=1, color=(255, 0, 0))

    if es_autentico:
        img_display.draw_string(30, 100, "AUTENTICO", scale=3, color=(0, 255, 0))
    else:
        img_display.draw_string(45, 100, "FALSO", scale=3, color=(255, 0, 0))

    img_display.draw_string(0, 0, "%s Bs (%.0f%%)" % (denom, cls_conf * 100), scale=1, color=(0, 255, 0))
    img_display.draw_string(0, 200, "t:%dms" % t, scale=1, color=(255, 0, 0))
    lcd.display(img_display)
    print("Deteccion lista en %dms, %d objetos" % (t, num_boxes))


# ===================== PREVIEW =====================
# ===================== ESTADO RPI =====================
rpi_connected = False


def preview_hasta_boton():
    img_pre = image.Image(size=SENSOR_WINDOW)
    img_pre.draw_string(5, 80, "Presiona boton", scale=2, color=(0, 255, 0))
    img_pre.draw_string(5, 120, "para escanear", scale=2, color=(0, 255, 0))
    if rpi_connected:
        img_pre.draw_circle(210, 14, 6, color=(0, 255, 0), fill=True)  # punto verde
    lcd.display(img_pre)
    pwm_blanco.duty(50)
    pwm_ir.duty(0)
    while button.value() == 1:
        img = sensor.snapshot()
        img.draw_string(5, 190, "Centrar billete", scale=1, color=(255, 255, 255))
        img.draw_string(5, 205, "dentro del cuadro", scale=1, color=(255, 255, 255))
        if rpi_connected:
            img.draw_circle(210, 14, 6, color=(0, 255, 0), fill=True)
        lcd.display(img)
        time.sleep_ms(10)
    while button.value() == 0:
        time.sleep_ms(10)
    pwm_blanco.duty(0)


# ===================== MAIN =====================
def main():
    init_camera()
    sensor.set_auto_exposure(False, exposure_us=EXP_W)
    sensor.set_auto_gain(False, gain_db=GAIN_W)
    lcd.init(type=1)
    lcd.rotation(1)
    lcd.clear(lcd.WHITE)
    uart = init_uart()
    comm = Comm(uart)

    preview_hasta_boton()

    while True:
        pwm_ir.duty(0)
        task_cls = kpu.load(CLASSIFY_ADDR)

        pwm_blanco.duty(50)
        time.sleep_ms(50)
        img = sensor.snapshot()
        img_cls = sensor.snapshot()
        pwm_blanco.duty(0)

        sensor.run(0)
        cls_idx, cls_conf = classificar(task_cls, img_cls)

        cls_label = CLASSIFY_LABELS[cls_idx]
        denom = parse_denomination(cls_label)

        task_cls = None
        gc.collect()

        if denom not in DETECT_MODELS:
            img_err = image.Image(size=SENSOR_WINDOW)
            img_err.draw_string(5, 100, "Sin modelo para", scale=2, color=(255, 0, 0))
            img_err.draw_string(5, 140, "%s Bs" % denom, scale=2, color=(255, 0, 0))
            lcd.display(img_err)
            time.sleep(2)
            sensor.run(1)
            preview_hasta_boton()
            continue

        uart.write("CLASIFICA:{}:{:.2f}\n".format(denom, cls_conf).encode())

        det_cfg = DETECT_MODELS[denom]
        task_det = kpu.load(det_cfg['model'])

        try:
            saved_boxes, t = bucle_deteccion(task_det, det_cfg['labels'], det_cfg['anchors'], img)
        finally:
            kpu.deinit(task_det)
            task_det = None
            gc.collect()

        capturar_y_enviar(saved_boxes, det_cfg['labels'], t, denom, cls_conf, comm)

        while button.value() == 1:
            time.sleep_ms(50)
        while button.value() == 0:
            time.sleep_ms(50)

        sensor.run(1)
        preview_hasta_boton()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.print_exception(e)
        lcd_show_except(e)
    finally:
        try: sensor.run(0)
        except Exception: pass
        gc.collect()
