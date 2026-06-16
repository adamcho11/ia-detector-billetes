"""
Maix Bit — Colector de features falsas directo a SD.
Guarda cada crop detectado como PNG en /sd/fakes/<denom>/<label>/
"""

import sensor, image, lcd, time
import KPU as kpu
import gc, sys, random, uos
from fpioa_manager import fm
from machine import Timer, PWM
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

CLASSIFY_ADDR = 0x300000
CLASSIFY_LABELS = ['10', '20r', '50', '50r', '100', '100r', '200', '200r', '10r', '20']
SENSOR_WINDOW = (224, 224)

def parse_denomination(label):
    return label.rstrip('r')

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

EXP_W  = 110000
GAIN_W = 1.0
EXP_IR = 110000
GAIN_IR = 1.0


def init_camera():
    try: sensor.run(0)
    except: pass
    time.sleep_ms(20)
    for _ in range(3):
        try: sensor.reset(); break
        except: time.sleep_ms(10)
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing(SENSOR_WINDOW)
    sensor.set_hmirror(False)
    sensor.set_vflip(False)
    sensor.run(1)
    sensor.skip_frames(10)


def classificar(task, img):
    fmap = kpu.forward(task, img)
    plist = list(fmap[:])
    max_val = max(plist)
    max_idx = plist.index(max_val)
    return max_idx, max_val


def preview_hasta_boton():
    # Preview minimal (solo espera boton)
    pwm_blanco.duty(50)
    pwm_ir.duty(0)
    while button.value() == 1:
        time.sleep_ms(5)
    while button.value() == 0:
        time.sleep_ms(5)
    pwm_blanco.duty(0)


def main():
    init_camera()
    sensor.set_auto_exposure(False, exposure_us=EXP_W)
    sensor.set_auto_gain(False, gain_db=GAIN_W)
    lcd.init(type=1)
    lcd.rotation(1)
    lcd.clear(lcd.WHITE)

    # Crear carpeta base en SD
    try: uos.mkdir("/sd/fakes")
    except: pass

    preview_hasta_boton()

    while True:
        pwm_ir.duty(0)
        task_cls = kpu.load(CLASSIFY_ADDR)

        pwm_blanco.duty(50)
        time.sleep_ms(30)
        img = sensor.snapshot()
        img_cls = sensor.snapshot()
        pwm_blanco.duty(0)

        sensor.run(0)
        cls_idx, cls_conf = classificar(task_cls, img_cls)
        denom = parse_denomination(CLASSIFY_LABELS[cls_idx])
        task_cls = None
        gc.collect()

        if denom not in DETECT_MODELS:
            sensor.run(1)
            time.sleep_ms(10)
            # Verificar boton para detener
            if button.value() == 1:
                continue
            else:
                break
            continue

        det_cfg = DETECT_MODELS[denom]
        task_det = kpu.load(det_cfg['model'])

        try:
            kpu.init_yolo2(task_det, 0.5, 0.3, 5, det_cfg['anchors'])
            objects = kpu.run_yolo2(task_det, img)

            saved_boxes = []
            if objects:
                for obj in objects:
                    pos = obj.rect()
                    cls_id = obj.classid()
                    if cls_id < len(det_cfg['labels']):
                        saved_boxes.append((pos, cls_id, obj.value()))

            kpu.deinit(task_det)
            task_det = None
            gc.collect()

            if not saved_boxes:
                sensor.run(1)
                if button.value() == 0:
                    break
                continue

            # Capturar IR
            pwm_blanco.duty(0)
            pwm_ir.duty(40)
            sensor.run(1)
            sensor.set_auto_exposure(True)
            sensor.set_auto_gain(True)
            sensor.skip_frames(5)
            gc.collect()
            img_ir = sensor.snapshot()
            pwm_ir.duty(0)
            sensor.set_auto_exposure(False, exposure_us=EXP_W)
            sensor.set_auto_gain(False, gain_db=GAIN_W)
            sensor.run(0)

            # Guardar
            saved = 0
            for pos, cls_id, conf in saved_boxes:
                label = det_cfg['labels'][cls_id]
                try:
                    base = "/sd/fakes"
                    for sub in [base, base + "/" + denom, base + "/" + denom + "/" + label]:
                        try: uos.mkdir(sub)
                        except: pass
                    crop = img_ir.copy(roi=pos)
                    fname = "{}/{}/{}/{}_{}_{}.jpg".format(
                        base, denom, label, label, random.randint(10000, 99999), int(conf * 100))
                    crop.save(fname, quality=85)
                    saved += 1
                except Exception as e:
                    print("  Error {}: {}".format(label, e))

            print("{} Bs: {}/{} guardados".format(denom, saved, len(saved_boxes)))

        finally:
            pwm_blanco.duty(0)
            pwm_ir.duty(0)
            if task_det is not None:
                kpu.deinit(task_det)
            gc.collect()

        # Detener si se presiona boton
        if button.value() == 0:
            print("Boton presionado - deteniendo")
            break

        sensor.run(1)

    print("Colector detenido")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.print_exception(e)
        img_err = image.Image(size=SENSOR_WINDOW)
        img_err.draw_string(0, 10, str(e), scale=1, color=(255, 0, 0))
        lcd.display(img_err)
    finally:
        try: sensor.run(0)
        except: pass
        gc.collect()
