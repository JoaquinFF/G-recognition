import cv2
import face_recognition as fr

# Cargar imagen de control
foto_control = fr.load_image_file('FLOWERS_CARA.jpg')
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)

# Codificar cara de control
cara_codificada_control = fr.face_encodings(foto_control)[0]

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
        # Comparar con la cara de control
        coincidencias = fr.compare_faces([cara_codificada_control], cara_codificada)
        distancia = fr.face_distance([cara_codificada_control], cara_codificada)
        
        # Ajustar coordenadas al tamaño original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        if coincidencias[0]:
            color = (0, 255, 0)  # Verde para coincidencia
            texto = f"Coincide ({distancia[0]:.2f})"
        else:
            color = (0, 0, 255)  # Rojo para no coincidencia
            texto = f"No coincide ({distancia[0]:.2f})"
            
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