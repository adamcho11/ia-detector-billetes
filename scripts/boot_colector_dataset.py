import sensor, image, lcd, time, uos
from Maix import GPIO
from fpioa_manager import fm
from machine import SDCard, Timer, PWM

# ==========================================
# 1. CONFIGURACIÓN DE HARDWARE Y PINES
# ==========================================
PIN_BTN_UP   = 19  # Arriba / Encender IR
PIN_BTN_DOWN = 10  # Abajo / Salir
PIN_BTN_SEL  = 9   # Selección / Foto 

PIN_LED_IR_1 = 2
PIN_LED_IR_2 = 6

POW_LED1 = 20
POW_LED2 = 60

# Registrar pines
fm.register(PIN_BTN_UP, fm.fpioa.GPIOHS0, force=True)
fm.register(PIN_BTN_DOWN, fm.fpioa.GPIOHS1, force=True)
fm.register(PIN_BTN_SEL, fm.fpioa.GPIOHS2, force=True)
fm.register(PIN_LED_IR_1, fm.fpioa.TIMER0_TOGGLE1, force=True)
fm.register(PIN_LED_IR_2, fm.fpioa.TIMER1_TOGGLE1, force=True)

# Configurar botones
btn_up   = GPIO(GPIO.GPIOHS0, GPIO.IN, GPIO.PULL_UP)
btn_down = GPIO(GPIO.GPIOHS1, GPIO.IN, GPIO.PULL_UP)
btn_sel  = GPIO(GPIO.GPIOHS2, GPIO.IN, GPIO.PULL_UP)

# Configurar PWM LEDs IR (Frecuencia a 3000Hz para evitar pitido)
tim0 = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_PWM)
pwm_ir_1 = PWM(tim0, freq=5000, duty=0, pin=PIN_LED_IR_1)
tim1 = Timer(Timer.TIMER1, Timer.CHANNEL0, mode=Timer.MODE_PWM)
pwm_ir_2 = PWM(tim1, freq=5000, duty=0, pin=PIN_LED_IR_2)

lcd.init()
lcd.rotation(0)

# ==========================================
# 2. CONFIGURACIÓN DE CÁMARA (320x224)
# ==========================================
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)      # 320x240 nativo    # Recorte exacto
sensor.set_windowing((48, 0, 240, 240))
sensor.set_hmirror(True)
sensor.set_vflip(True)
sensor.run(1)

# ==========================================
# 3. ESTRUCTURA DEL DATASET
# ==========================================
datasets_list = [
    ("1. BillsRGB - LadoA - Anverso", ["10_Bs", "20_Bs", "50_Bs", "100_Bs", "200_Bs", "Fondo_RGB"]),
    ("2. BillsRGB - LadoB - Anverso", ["10_Bs", "20_Bs", "50_Bs", "100_Bs", "200_Bs", "Fondo_RGB"]),
    ("3. BillsIR - LadoA - Anverso",  ["10_Bs_IR", "20_Bs_IR", "50_Bs_IR", "100_Bs_IR", "200_Bs_IR", "Fondo_IR"]),
    ("4. BillsIR - LadoB - Anverso",  ["10_Bs_IR", "20_Bs_IR", "50_Bs_IR", "100_Bs_IR", "200_Bs_IR", "Fondo_IR"]),
    ("5. BillsRGB - LadoA - Reverso", ["10_Bs", "20_Bs", "50_Bs", "100_Bs", "200_Bs", "Fondo_RGB"]),
    ("6. BillsRGB - LadoB - Reverso", ["10_Bs", "20_Bs", "50_Bs", "100_Bs", "200_Bs", "Fondo_RGB"]),
    ("7. BillsIR - LadoB - Reverso",  ["10_Bs_IR", "20_Bs_IR", "50_Bs_IR", "100_Bs_IR", "200_Bs_IR", "Fondo_IR"])
]

# Extraemos solo los nombres para el primer menú
nombres_tipos = [item[0] for item in datasets_list]
ESTADO_MENU_TIPO = 0
ESTADO_MENU_CLASE = 1
ESTADO_CAMARA = 2

estado_actual = ESTADO_MENU_TIPO
indice_seleccion = 0
tipo_seleccionado = ""
clase_seleccionada = ""
ir_encendido = False
fotos_tomadas = 0

def conectar_sd():
    try:
        if 'sd' not in uos.listdir('/'): uos.mount(SDCard(), "/sd")
        return True
    except: return False

def asegurar_ruta(ruta):
    try: uos.mkdir("/sd/" + ruta)
    except: pass

def esperar_soltar_boton(btn):
    time.sleep_ms(20)
    while btn.value() == 0: time.sleep_ms(10)

sd_ok = conectar_sd()

while True:
    # --- MENÚ TIPO ---
    if estado_actual == ESTADO_MENU_TIPO:
        img = image.Image(size=(320, 240))
        img.draw_rectangle(0,0,320,240, fill=True, color=(0,0,0))
        img.draw_string(40, 10, "DATASET PRINCIPAL", (255, 255, 0), 1.5)
        for i, nombre in enumerate(nombres_tipos):
            color = (0, 255, 0) if i == indice_seleccion else (150, 150, 150)
            img.draw_string(30, 60 + (i * 20), ("> " if i == indice_seleccion else "  ") + nombre, color, 1)
        lcd.display(img)

        if btn_down.value() == 0:
            indice_seleccion = (indice_seleccion + 1) % len(nombres_tipos)
            esperar_soltar_boton(btn_down)
        elif btn_up.value() == 0:
            indice_seleccion = (indice_seleccion - 1) % len(nombres_tipos)
            esperar_soltar_boton(btn_up)
        elif btn_sel.value() == 0:
            tipo_seleccionado = nombres_tipos[indice_seleccion]
            estado_actual = ESTADO_MENU_CLASE
            indice_seleccion = 0
            esperar_soltar_boton(btn_sel)

    # --- MENÚ CLASE ---
    elif estado_actual == ESTADO_MENU_CLASE:
        img = image.Image(size=(320, 240))
        clases_actuales = datasets_list[nombres_tipos.index(tipo_seleccionado)][1]
        img.draw_rectangle(0,0,320,240, fill=False, color=(0,0,0))
        img.draw_string(40, 10, "CLASE DE CAPTURA", (255, 150, 0), 1.5)
        for i, clase in enumerate(clases_actuales):
            color = (0, 255, 255) if i == indice_seleccion else (150, 150, 150)
            img.draw_string(30, 60 + (i * 18), ("> " if i == indice_seleccion else "  ") + clase, color, 1)
        lcd.display(img)

        if btn_down.value() == 0:
            indice_seleccion = (indice_seleccion + 1) % len(clases_actuales)
            esperar_soltar_boton(btn_down)
        elif btn_up.value() == 0:
            indice_seleccion = (indice_seleccion - 1) % len(clases_actuales)
            esperar_soltar_boton(btn_up)
            
        elif btn_sel.value() == 0:
            # --- NUEVA LÓGICA DE PULSACIÓN LARGA ---
            start_time = time.ticks_ms()
            
            # Esperamos a que sueltes el botón para medir el tiempo
            while btn_sel.value() == 0:
                time.sleep_ms(10)
                
            tiempo_presionado = time.ticks_ms() - start_time
            
            if tiempo_presionado > 500: 
                # PULSACIÓN LARGA (> MEDIO SEGUNDO) -> RETROCEDER
                estado_actual = ESTADO_MENU_TIPO
                indice_seleccion = 0 # Reiniciamos el cursor del menú anterior
            else:
                # PULSACIÓN CORTA -> ENTRAR A LA CÁMARA
                clase_seleccionada = clases_actuales[indice_seleccion]
                
                # 1. Asegurar que las carpetas existan en la SD
                carpeta_base = tipo_seleccionado.replace(" ", "_")
                asegurar_ruta(carpeta_base)
                ruta_final = carpeta_base + "/" + clase_seleccionada
                asegurar_ruta(ruta_final)
                
                # 2. LECTURA DE LA SD PARA EL CONTADOR
                if sd_ok:
                    try:
                        archivos = uos.listdir("/sd/" + ruta_final)
                        fotos_tomadas = len([f for f in archivos if f.endswith('.bmp') or f.endswith('.jpg')])
                    except:
                        fotos_tomadas = 0
                else:
                    fotos_tomadas = 0
                
                # 3. Encender IR automáticamente solo si es Dataset IR
                if "IR" in tipo_seleccionado:
                    ir_encendido = True
                    pwm_ir_1.duty(POW_LED1)
                    pwm_ir_2.duty(POW_LED2)
                else:
                    ir_encendido = False
                    pwm_ir_1.duty(0)
                    pwm_ir_2.duty(0)
                    
                estado_actual = ESTADO_CAMARA
            # (No necesitamos esperar_soltar_boton(btn_sel) aquí porque el while de arriba ya lo hizo)

    # --- MODO CÁMARA ---
    elif estado_actual == ESTADO_CAMARA:
        img = sensor.snapshot()

        # Interfaz superpuesta (SIN FRANJA NEGRA, CON SOMBRA PARA LEER FÁCIL)
        # Sombra negra
        img.draw_string(6, 6, clase_seleccionada, (0, 0, 0), 1)
        img.draw_string(181, 6, "N:" + str(fotos_tomadas), (0, 0, 0), 1)
        img.draw_string(6, 206, "SEL:FOTO | ABJ:Salir | ARR:IR", (0, 0, 0), 1.2)
        # Texto principal
        img.draw_string(5, 5, clase_seleccionada, (0, 255, 0), 1)
        img.draw_string(180, 5, "N:" + str(fotos_tomadas), (255, 255, 255), 1)
        img.draw_string(5, 205, "SEL:FOTO | ABJ:Salir | ARR:IR", (255, 255, 0), 1.2)
        
        lcd.display(img)

        if btn_up.value() == 0: # Toggle IR manual
            ir_encendido = not ir_encendido
            pwm_ir_1.duty(POW_LED1 if ir_encendido else 0)   
            pwm_ir_2.duty(POW_LED2 if ir_encendido else 0)  
            esperar_soltar_boton(btn_up)

        elif btn_down.value() == 0: # Salir
            pwm_ir_1.duty(0)
            pwm_ir_2.duty(0)
            ir_encendido = False
            estado_actual = ESTADO_MENU_CLASE
            esperar_soltar_boton(btn_down)

        elif btn_sel.value() == 0: # CAPTURAR
            if sd_ok:
                # Flash blanco en pantalla para confirmar
                img.draw_rectangle(0, 0, 240, 240, color=(255, 255, 255), thickness=10)
                lcd.display(img)

                # 1. Tomar la foto
                img_clean = sensor.snapshot()
                
                # 2. DETENER EL SENSOR FÍSICAMENTE
                sensor.run(0)
                
                # 3. Aplicar Blanco y Negro si es necesario en la misma RAM
                if "IR" in tipo_seleccionado:
                    img_clean.to_grayscale()
                
                path = "/sd/" + tipo_seleccionado.replace(" ", "_") + "/" + clase_seleccionada
                timestamp = str(time.ticks_ms())
                
                # 4. GUARDAR EN BMP PURO
                img_clean.save(path + "/img_" + timestamp + ".bmp")

                # 5. VOLVER A ENCENDER EL SENSOR Y ESTABILIZAR
                sensor.run(1)
                sensor.skip_frames(time = 100)

                # Sumar 1 al contador visual al instante
                fotos_tomadas += 1
            esperar_soltar_boton(btn_sel)