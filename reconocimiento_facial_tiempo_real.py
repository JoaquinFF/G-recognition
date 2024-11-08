import cv2
import face_recognition as fr
from datetime import datetime
import csv
from datetime import timedelta

# Cargar imágenes de control y crear arrays de nombres y codificaciones
nombres_conocidos = ['JOAQUIN', 'FACUNDO']  # Añade los nombres que necesites
caras_conocidas = []

for nombre in nombres_conocidos:
    imagen = fr.load_image_file(f'{nombre}_CARA.jpg')
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    codificacion = fr.face_encodings(imagen_rgb)[0]
    caras_conocidas.append(codificacion)

# Diccionario para almacenar la última entrada de cada persona
ultimas_entradas = {}  # {nombre: {'entrada': datetime, 'salida': datetime}}

# Iniciar captura de video
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Agregar al inicio del archivo, después de los imports
tiempo_desconocido = {}  # Diccionario para rastrear el tiempo que una cara ha sido desconocida

while True:
    # Leer frame de la cámara
    ret, frame = captura.read()
    if not ret:
        break
        
    # Reducir tamaño del frame para mejor rendimiento
    frame_pequeño = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Detectar caras en el frame
    ubicaciones_cara = fr.face_locations(frame_pequeño)
    caras_codificadas = fr.face_encodings(frame_pequeño, ubicaciones_cara)
    
    for (top, right, bottom, left), cara_codificada in zip(ubicaciones_cara, caras_codificadas):
        # Comparar con todas las caras conocidas
        coincidencias = fr.compare_faces(caras_conocidas, cara_codificada)
        distancias = fr.face_distance(caras_conocidas, cara_codificada)
        
        # Encontrar el mejor match
        mejor_coincidencia = None
        if True in coincidencias:
            mejor_indice = distancias.argmin()
            if coincidencias[mejor_indice]:
                mejor_coincidencia = nombres_conocidos[mejor_indice]
                tiempo_actual = datetime.now()
                
                # Verificar si la persona ya tiene un registro hoy
                puede_registrar = True
                puede_registrar_salida = False

                if mejor_coincidencia in ultimas_entradas:
                    ultima_entrada = ultimas_entradas[mejor_coincidencia].get('entrada')
                    ultima_salida = ultimas_entradas[mejor_coincidencia].get('salida')
                    
                    if ultima_entrada:
                        tiempo_transcurrido = tiempo_actual - ultima_entrada
                        
                        # Si ya tiene entrada pero no salida, verificar si puede registrar salida
                        if not ultima_salida:
                            if tiempo_transcurrido >= timedelta(minutes=1) and tiempo_transcurrido <= timedelta(minutes=5): #ESTO ES PARA PRUEBAS, CAMBIAR A 24 HORAS
                                puede_registrar_salida = True
                            puede_registrar = False  # No permitir nueva entrada si aún no hay salida
                        # Si ya tiene entrada y salida, verificar si pueden pasar 24 horas
                        elif tiempo_transcurrido < timedelta(minutes=2):  # 2 minutos para pruebas (cambiar a 24 horas después)
                            puede_registrar = False
                
                # Añadir prints para debugging
                print(f"\nPersona detectada: {mejor_coincidencia}")
                
                # Registrar entrada si es permitido
                if puede_registrar:
                    print("Registrando entrada...")
                    ultimas_entradas[mejor_coincidencia] = {'entrada': tiempo_actual, 'salida': None}
                    with open('registro_entradas.csv', 'a', newline='') as archivo:
                        escritor = csv.writer(archivo)
                        escritor.writerow([mejor_coincidencia, tiempo_actual.strftime("%Y-%m-%d %H:%M:%S"), "ENTRADA"])
                
                # Registrar salida si es permitido
                elif puede_registrar_salida:
                    print("Registrando salida...")
                    ultimas_entradas[mejor_coincidencia]['salida'] = tiempo_actual
                    with open('registro_salidas.csv', 'a', newline='') as archivo:
                        escritor = csv.writer(archivo)
                        escritor.writerow([mejor_coincidencia, tiempo_actual.strftime("%Y-%m-%d %H:%M:%S"), "SALIDA"])
        
        # Ajustar coordenadas al tamaño original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        if mejor_coincidencia:
            color = (0, 255, 0)  # Verde para coincidencia
            texto = f"{mejor_coincidencia} ({distancias[mejor_indice]:.2f})"
        else:
            # Obtener un identificador único para la cara (usando las coordenadas)
            cara_id = f"{top}_{right}_{bottom}_{left}"
            tiempo_actual = datetime.now()
            
            # Inicializar o actualizar el tiempo para esta cara desconocida
            if cara_id not in tiempo_desconocido:
                tiempo_desconocido[cara_id] = tiempo_actual
            
            # Verificar si han pasado 5 segundos
            tiempo_transcurrido = (tiempo_actual - tiempo_desconocido[cara_id]).total_seconds()
            
            if tiempo_transcurrido >= 5:
                print("\n¡Persona desconocida detectada por 5 segundos!")
                nombre = input("Por favor, ingrese el nombre del nuevo cliente: ").upper()
                
                # Guardar la imagen de la cara
                cara_img = frame[top:bottom, left:right]
                cv2.imwrite(f'{nombre}_CARA.jpg', cara_img)
                
                # Registrar nuevo cliente en CSV
                with open('registro_personas.csv', 'a', newline='') as archivo:
                    escritor = csv.writer(archivo)
                    escritor.writerow([nombre, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                
                # Actualizar las listas de caras conocidas
                imagen = fr.load_image_file(f'{nombre}_CARA.jpg')
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                nueva_codificacion = fr.face_encodings(imagen_rgb)[0]
                caras_conocidas.append(nueva_codificacion)
                nombres_conocidos.append(nombre)
                
                print(f"Cliente {nombre} registrado exitosamente!")
                
                mejor_coincidencia = nombre
                color = (0, 255, 0)
                texto = f"{nombre} (Nuevo)"
                
                # Limpiar el tiempo de desconocido después del registro
                tiempo_desconocido.clear()
            else:
                color = (0, 0, 255)  # Rojo para desconocido
                texto = f"Desconocido ({5 - tiempo_transcurrido:.1f}s)"
        
        # Dibujar rectángulo y texto
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, texto, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    
    # Mostrar frame
    cv2.imshow('Reconocimiento Facial en Tiempo Real', frame)
    
    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows() 