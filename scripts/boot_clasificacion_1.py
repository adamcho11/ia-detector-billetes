# generado por maixhub, optimizado para hardware resizing con pix_to_ai()
# Modificado por Detector Bills para corregir el cegamiento del sensor e iluminación directa
import sensor, image, lcd, time
import KPU as kpu
from machine import UART, Timer, PWM
import gc, sys
from fpioa_manager import fm
from Maix import GPIO 

# ==========================================
# CONFIGURACIÓN DE POTENCIA Y HARDWARE
# ==========================================
POTENCIA_LED_IR     = 12  # Si se sigue cegando, baja este valor a 10 o 5
PIN_LED_IR          = 2
PIN_PULSADOR        = 9  
# ==========================================

# 1. Configuración de Hardware (LED IR)
fm.register(PIN_LED_IR, fm.fpioa.TIMER1_TOGGLE1, force=True)
tim1 = Timer(Timer.TIMER1, Timer.CHANNEL0, mode=Timer.MODE_PWM)
pwm_ir = PWM(tim1, freq=10000, duty=0, pin=PIN_LED_IR)

# 2. Configuración de Hardware (Pulsador en Pin 9)
fm.register(PIN_PULSADOR, fm.fpioa.GPIO0, force=True)
boton = GPIO(GPIO.GPIO0, GPIO.IN, GPIO.PULL_UP)

input_size = (224, 224)
labels = ['20', '20r', '100', '100r', '200r', '10', '10r', '50', '50r', '200']

def lcd_show_except(e):
    import uio
    err_str = uio.StringIO()
    sys.print_exception(e, err_str)
    err_str = err_str.getvalue()
    img = image.Image(size=input_size)
    img.draw_string(0, 10, err_str, scale=1, color=(0xff,0x00,0x00))
    lcd.display(img)

class Comm:
    def __init__(self, uart):
        self.uart = uart

    def send_classify_result(self, pmax, idx, label):
        msg = "{}:{:.2f}:{}\n".format(idx, pmax, label)
        self.uart.write(msg.encode())

def init_uart():
    fm.register(10, fm.fpioa.UART1_TX, force=True)
    fm.register(11, fm.fpioa.UART1_RX, force=True)
    uart = UART(UART.UART1, 115200, 8, 0, 0, timeout=1000, read_buf_len=256)
    return uart

def procesar_etiqueta(label_str):
    label_str = label_str.strip()
    if label_str.endswith('r'):
        billete = label_str[:-1]
        orientacion = "Reversa (r)"
    else:
        billete = label_str
        orientacion = "Frente"
    return billete, orientacion

def main(labels = None, model_addr="/sd/m.kmodel", sensor_window=input_size, lcd_rotation=0, sensor_hmirror=False, sensor_vflip=False):
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing(sensor_window)
    sensor.set_hmirror(sensor_hmirror)
    sensor.set_vflip(sensor_vflip)
    
    # -----------------------------------------------------------------
    # CONTROL DIRECTO DE EXPOSICIÓN (Evita que la cámara se ciegue)
    # -----------------------------------------------------------------
    sensor.set_auto_gain(False, gain_db=10)      # Desactiva ganancia automática y la fija en un valor estable
    sensor.set_auto_exposure(False, exposure_us=40000) # Fija el tiempo de exposición (en microsegundos)            # Desactiva el balance de blancos automático para evitar cambios de color
    
    sensor.run(1)

    lcd.init(type=1)
    lcd.rotation(lcd_rotation)
    lcd.clear(lcd.BLACK)

    if not labels:
        with open('labels.txt','r') as f:
            exec(f.read())
    if not labels:
        print("no labels.txt")
        img = image.Image(size=(320, 240))
        img.draw_string(90, 110, "no labels.txt", color=(255, 0, 0), scale=2)
        lcd.display(img)
        return 1
    try:
        img = image.Image("startup.jpg")
        lcd.display(img)
    except Exception:
        img = image.Image(size=(320, 240))
        img.draw_string(50, 110, "PULSA EL BOTON PARA ESCANEAR", color=(0, 255, 0), scale=1.5)
        lcd.display(img)

    uart = init_uart()
    comm = Comm(uart)

    try:
        task = None
        task = kpu.load(model_addr)
        
        while(True):
            # Visor en espera
            img_espera = sensor.snapshot()
            img_espera.draw_string(10, 10, "ESTADO: Listo (Esperando boton)", scale=1.2, color=(255, 255, 255))
            lcd.display(img_espera)
            
            # Si se presiona el pulsador
            if boton.value() == 0:
                
                # 1. Control directo de iluminación: Encendemos LED IR
                pwm_ir.duty(POTENCIA_LED_IR)
                
                # Tiramos los primeros 2 frames "basura" por si acaso el sensor tarda unos milisegundos en asentarse
                for _ in range(2):
                    sensor.snapshot()
                
                acumulador_probabilidades = [0.0] * len(labels)
                t_inicio = time.ticks_ms()
                
                # Ejecutamos las 3 inferencias solicitadas con luz estable
                for _ in range(3):
                    img = sensor.snapshot()
                    fmap = kpu.forward(task, img)
                    plist = fmap[:]
                    
                    for idx, prob in enumerate(plist):
                        acumulador_probabilidades[idx] += prob
                
                # 2. Apagamos el LED IR de inmediato
                pwm_ir.duty(0)
                
                t_total = time.ticks_ms() - t_inicio
                
                # Procesado de promedios
                promedios = [p / 3.0 for p in acumulador_probabilidades]
                pmax_promedio = max(promedios)
                max_index = promedios.index(pmax_promedio)
                etiqueta_ganadora = labels[max_index].strip()
                
                valor_billete, orientacion_billete = procesar_etiqueta(etiqueta_ganadora)
                
                # --- MOSTRAR RESULTADO ---
                img_resultado = sensor.snapshot() 
                img_resultado.draw_string(10, 10, "Billete: %s" % valor_billete, scale=1.6, color=(0, 255, 0))
                img_resultado.draw_string(10, 40, "Pos: %s" % orientacion_billete, scale=1.4, color=(255, 255, 0))
                img_resultado.draw_string(10, 70, "Conf Prom: %.2f" % pmax_promedio, scale=1.4, color=(255, 255, 255))
                img_resultado.draw_string(10, 200, "Procesado en: %dms" % t_total, scale=1.2, color=(0, 162, 232))
                
                lcd.display(img_resultado)
                comm.send_classify_result(pmax_promedio, max_index, etiqueta_ganadora)
                
                time.sleep_ms(2000)

    except Exception as e:
        raise e
    finally:
        if not task is None:
            kpu.deinit(task)


if __name__ == "__main__":
    try:
        main(labels=labels, model_addr="/sd/model-279265.kmodel")
    except Exception as e:
        sys.print_exception(e)
        lcd_show_except(e)
    finally:
        gc.collect()
        