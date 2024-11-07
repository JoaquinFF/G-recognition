import cv2
import face_recognition as fr

# Cargar imágenes de control y crear arrays de nombres y codificaciones
nombres_conocidos = ['FLOWERS', 'GUEVARA']  # Añade los nombres que necesites
caras_conocidas = []

for nombre in nombres_conocidos:
    imagen = fr.load_image_file(f'{nombre}_CARA.jpg')
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    codificacion = fr.face_encodings(imagen_rgb)[0]
    caras_conocidas.append(codificacion)

# Iniciar captura de video
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
        
        # Ajustar coordenadas al tamaño original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        if mejor_coincidencia:
            color = (0, 255, 0)  # Verde para coincidencia
            texto = f"{mejor_coincidencia} ({distancias[mejor_indice]:.2f})"
        else:
            color = (0, 0, 255)  # Rojo para no coincidencia
            texto = "Desconocido"
            
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