import sensor, image, lcd, time, uos
from Maix import GPIO
from fpioa_manager import fm
from machine import SDCard, Timer, PWM

# ==========================================
# CONFIGURACIÓN DE POTENCIA (0 a 100)
# ==========================================
POTENCIA_LED_NORMAL = 50
POTENCIA_LED_IR     = 50
# ==========================================

# 1. Configuración de Hardware
IO_BOOT_BUTTON = 16
PIN_LED = 2
PIN_LED_IR = 6

fm.register(IO_BOOT_BUTTON, fm.fpioa.GPIOHS0)
fm.register(PIN_LED, fm.fpioa.TIMER0_TOGGLE1, force=True)
fm.register(PIN_LED_IR, fm.fpioa.TIMER1_TOGGLE1, force=True)

btn = GPIO(GPIO.GPIOHS0, GPIO.IN, GPIO.PULL_UP)
lcd.init()

# 2. Configuración PWM
tim0 = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_PWM)
pwm_led = PWM(tim0, freq=1000, duty=0, pin=PIN_LED)

tim1 = Timer(Timer.TIMER1, Timer.CHANNEL0, mode=Timer.MODE_PWM)
pwm_ir = PWM(tim1, freq=1000, duty=0, pin=PIN_LED_IR)

# Inicializar cámara
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(True)
sensor.set_vflip(True)
sensor.run(1)

# --- LISTA DE CLASES ---`
clases = [
    "0_Fondo", "10_Bolivianos", "20_Bolivianos", "50_Bolivianos",
    "100_Bolivianos", "200_Bolivianos", "Semaforo_Rojo",
    "Semaforo_Verde", "Calle_Libre", "Objeto_1", "Objeto_2", "Objeto_3"
]

indice_menu = 0
modo_camara = False
clase_seleccionada = ""
foto_count = 0
led_encendido = False

def conectar_sd():
    try:
        if 'sd' not in uos.listdir('/'):
            uos.mount(SDCard(), "/sd")
        return True
    except: return False

def asegurar_ruta(ruta):
    try:
        partes = ruta.split('/')
        camino = ""
        for i in range(len(partes) - 1): # No incluir el nombre del archivo .jpg
            if partes[i] == "": continue
            camino += "/" + partes[i]
            try:
                uos.mkdir(camino)
            except: pass # Si ya existe, ignorar
    except: pass

sd_ok = conectar_sd()

while True:
    if not modo_camara:
        pwm_led.duty(0)
        pwm_ir.duty(0)
        led_encendido = False
        img_menu = image.Image(size=(320, 240))
        img_menu.draw_rectangle(0,0,320,240, fill=True, color=(20,20,20))
        img_menu.draw_string(60, 5, "DATASET COLLECTOR", (255, 255, 255), 1.5)

        for i, c in enumerate(clases):
            color = (0, 255, 0) if i == indice_menu else (150, 150, 150)
            prefix = "> " if i == indice_menu else "  "
            img_menu.draw_string(25, 35 + (i * 14), prefix + c, color, 1)

        lcd.display(img_menu)

        if btn.value() == 0:
            inicio = time.ticks_ms()
            while btn.value() == 0: time.sleep_ms(10)
            duracion = time.ticks_diff(time.ticks_ms(), inicio)

            if duracion > 800:
                clase_seleccionada = clases[indice_menu]
                if "Bolivianos" in clase_seleccionada:
                    pwm_led.duty(POTENCIA_LED_NORMAL)
                    pwm_ir.duty(POTENCIA_LED_IR)
                    led_encendido = True
                modo_camara = True
            else:
                indice_menu = (indice_menu + 1) % len(clases)

    else:
        img = sensor.snapshot()

        if btn.value() == 0:
            inicio = time.ticks_ms()
            while btn.value() == 0: time.sleep_ms(10)
            duracion = time.ticks_diff(time.ticks_ms(), inicio)

            if duracion > 1500:
                modo_camara = False
                pwm_led.duty(0)
                pwm_ir.duty(0)
            elif 500 < duracion <= 1200:
                led_encendido = not led_encendido
                pwm_led.duty(POTENCIA_LED_NORMAL if led_encendido else 0)
                pwm_ir.duty(POTENCIA_LED_IR if led_encendido else 0)
            else:
                if sd_ok:
                    folder_name = clase_seleccionada.replace(" ","_")
                    nombre = "/sd/" + folder_name + "/img_" + str(time.ticks_ms()) + ".jpg"
                    asegurar_ruta(nombre) # <--- AQUÍ SE CREA LA CARPETA SI FALTA
                    img.save(nombre)
                    img.draw_rectangle(0, 0, 320, 240, color=(255, 255, 255), thickness=10)
                    lcd.display(img)
                    foto_count += 1

        img.draw_rectangle(0, 0, 320, 22, color=(0, 0, 0), fill=True)
        if led_encendido:
            info = "LED:{}% | IR:{}%".format(POTENCIA_LED_NORMAL, POTENCIA_LED_IR)
            img.draw_string(5, 4, info, (255, 255, 0), 1)
        else:
            img.draw_string(5, 4, "LEDS: OFF", (100, 100, 100), 1)

        img.draw_string(140, 4, "CLASE: " + clase_seleccionada, (0, 255, 0), 1)
        img.draw_string(10, 30, "Fotos: " + str(foto_count), (255, 255, 255), 1)
        img.draw_string(10, 222, "Clic: Foto | Medio: LEDs | Largo: Menu", (150, 150, 150), 1)
        lcd.display(img)
