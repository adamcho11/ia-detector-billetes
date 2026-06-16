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
pwm_blanco = PWM(tim1, freq=5000, duty=0, pin=PIN_LED_BLANCO)
tim2 = Timer(Timer.TIMER2, Timer.CHANNEL0, mode=Timer.MODE_PWM)
pwm_ir = PWM(tim2, freq=5000, duty=50, pin=PIN_LED_IR)          # IR a 30% — subir si muy oscuro

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
        'model': '/sd/model-10_deteccion.kmodel',
        'labels': ['valor_10bs', 'serie_a', 'valor_ir_10bs', 'personaje_10bs', 'animal_10bs'],
        'anchors': [1.78, 2.22, 3.12, 0.78, 3.19, 0.78, 2.06, 1.66, 1.34, 1.12],
    },
    '20': {
        'model': '/sd/model-20_deteccion.kmodel',
        'labels': ['valor_20bs', 'serie_a', 'valor_ir_20bs', 'personaje_20bs', 'animal_20bs'],
        'anchors': [1.78, 2.22, 3.12, 0.78, 3.19, 0.78, 2.06, 1.66, 1.34, 1.12],
    },
    '50': {
        'model': '/sd/model-50_deteccion.kmodel',
        'labels': ['valor_50bs', 'serie_a', 'valor_ir_50bs', 'personaje_50bs', 'animal_50bs'],
        'anchors': [1.78, 2.22, 3.12, 0.78, 3.19, 0.78, 2.06, 1.66, 1.34, 1.12],
    },
    '100': {
        'model': '/sd/model-100_deteccion.kmodel',
        'labels': ['valor_100bs', 'serie_a', 'valor_ir_100bs', 'personaje_100bs', 'animal_100bs'],
        'anchors': [1.78, 2.22, 3.12, 0.78, 3.19, 0.78, 2.06, 1.66, 1.34, 1.12],
    },
    '200': {
        'model': '/sd/model-200_deteccion.kmodel',
        'labels': ['valor_200bs', 'serie_a', 'valor_ir_200bs', 'personaje_200bs', 'animal_200bs'],
        'anchors': [1.78, 2.22, 3.12, 0.78, 3.19, 0.78, 2.06, 1.66, 1.34, 1.12],
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
    fm.register(10, fm.fpioa.UART1_TX, force=True)
    fm.register(11, fm.fpioa.UART1_RX, force=True)
    return UART(UART.UART1, 115200, 8, 0, 0, timeout=1000, read_buf_len=256)


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
    time.sleep_ms(200)
    for _ in range(3):
        try:
            sensor.reset()
            break
        except Exception:
            time.sleep_ms(300)
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing(SENSOR_WINDOW)
    sensor.set_hmirror(False)
    sensor.set_vflip(False)
    sensor.run(1)
    sensor.skip_frames(30)


# ===================== CLASIFICACION =====================
def classificar(task):
    sensor.set_auto_exposure(0, 100000)
    sensor.set_auto_gain(0, 4)
    sums = None
    for _ in range(3):
        img = sensor.snapshot()
        fmap = kpu.forward(task, img)
        plist = list(fmap[:])
        if sums is None:
            sums = plist
        else:
            for i in range(len(plist)):
                sums[i] += plist[i]
        time.sleep_ms(30)

    avg = [s / 3 for s in sums]
    max_val = max(avg)
    max_idx = avg.index(max_val)
    return max_idx, max_val


# ===================== DETECCION =====================
def bucle_deteccion(task, det_labels, anchors, comm):
    kpu.init_yolo2(task, 0.5, 0.3, 5, anchors)

    sensor.set_auto_exposure(0) 
    sensor.set_auto_gain(0)
    sensor.set_auto_exposure(0, 407)
    sensor.set_gainceiling(52) 
    
    img = sensor.snapshot()
    t = time.ticks_ms()
    objects = kpu.run_yolo2(task, img)
    t = time.ticks_ms() - t

    if objects:
        for obj in objects:
            pos = obj.rect()
            cls_id = obj.classid()
            if cls_id < len(det_labels):
                img.draw_rectangle(pos)
                img.draw_string(pos[0], pos[1],
                    "%s : %.2f" % (det_labels[cls_id], obj.value()),
                    scale=1, color=(255, 0, 0))
        comm.send_detect_result(objects, det_labels)
    lcd.display(img)
    print("Deteccion lista en %dms, %d objetos" % (t, len(objects) if objects else 0))


# ===================== PREVIEW =====================
def preview_hasta_boton():
    img_pre = image.Image(size=SENSOR_WINDOW)
    img_pre.draw_string(5, 80, "Presiona boton", scale=2, color=(0, 255, 0))
    img_pre.draw_string(5, 120, "para escanear", scale=2, color=(0, 255, 0))
    lcd.display(img_pre)

    while button.value() == 1:
        img = sensor.snapshot()
        img.draw_rectangle((0, 0, 223, 223), color=(255, 255, 0), thickness=1)        
        img.draw_string(5, 190, "Centrar billete",  scale=1, color=(255, 255, 0))
        img.draw_string(5, 205, "dentro del cuadro", scale=1, color=(255, 255, 0))
        lcd.display(img)
        
    while button.value() == 0:
        time.sleep_ms(30)

    lcd.clear(lcd.WHITE)


# ===================== MAIN =====================
def main():
    init_camera()
    lcd.init(type=1)
    lcd.rotation(1)
    lcd.clear(lcd.WHITE)
    uart = init_uart()
    comm = Comm(uart)

    preview_hasta_boton()

    while True:
        # CLASIFICAR
        img_load = image.Image(size=SENSOR_WINDOW)
        img_load.draw_string(5, 100, "Clasificando...", scale=2, color=(255, 255, 255))
        lcd.display(img_load)

        sensor.run(0)
        pwm_ir.duty(0)
        task_cls = kpu.load(CLASSIFY_ADDR)
        sensor.run(1)

        pwm_blanco.duty(100)
        cls_idx, cls_conf = classificar(task_cls)
        pwm_blanco.duty(0)

        cls_label = CLASSIFY_LABELS[cls_idx]
        denom = parse_denomination(cls_label)

        sensor.run(0)
        task_cls = None
        for _ in range(2):
            gc.collect()
            time.sleep_ms(50)

        if denom not in DETECT_MODELS:
            img_err = image.Image(size=SENSOR_WINDOW)
            img_err.draw_string(5, 100, "Sin modelo para", scale=2, color=(255, 0, 0))
            img_err.draw_string(5, 140, "%s Bs" % denom, scale=2, color=(255, 0, 0))
            lcd.display(img_err)
            time.sleep(2)
            preview_hasta_boton()
            continue

        # MOSTRAR RESULTADO
        print ("se detecto el Billete: %s BS"% denom)

        # CARGAR DETECCION
        det_cfg = DETECT_MODELS[denom]
        print("Cargando deteccion %s Bs: %s" % (denom, det_cfg['model']))
        task_det = kpu.load(det_cfg['model'])
        sensor.run(1)
        pwm_ir.duty(100)
        #pwm_blanco.duty(100)

        try:
            bucle_deteccion(task_det, det_cfg['labels'], det_cfg['anchors'], comm)
        finally:
            sensor.run(0)
            pwm_blanco.duty(0)
            pwm_ir.duty(0)
            kpu.deinit(task_det)
            task_det = None
            gc.collect()

        # Esperar boton y volver a preview
        while button.value() == 1:
            time.sleep_ms(50)
        while button.value() == 0:
            time.sleep_ms(50)

        sensor.run(1)
        pwm_ir.duty(100)
        preview_hasta_boton()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.print_exception(e)
        lcd_show_except(e)
    finally:
        try:
            sensor.run(0)
        except Exception:
            pass
        gc.collect()