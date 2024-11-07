import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime

# Crear base de datos
ruta = 'Personas_Autorizadas'
mis_imagenes = []
nombres_autorizados = []
lista_personas = os.listdir(ruta)

for nombre in lista_personas:

    imagen_actual = cv2.imread(f"{ruta}/{nombre}")
    mis_imagenes.append(imagen_actual)
    nombres_autorizados.append(os.path.splitext(nombre)[0])


print(nombres_autorizados)

# Codificar imagenes
def codificar(imagenes):

    # Crear una lista nueva
    lista_codificada = []

    # Pasar todas las imagenes a RGB
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # Codificar
        codificado = fr.face_encodings(imagen)[0]

        # Agregar a la lista
        lista_codificada.append(codificado)

    # Devolver lista codificada
    return lista_codificada

# Registrar los ingresos
def registrar_ingresos(persona):
    f = open("registro.csv", "r+")
    lista_datos = f.readlines()
    nombres_registro = []

    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registro.append(ingreso[0])

    if persona not in nombres_registro:
        ahora = datetime.now()
        str_ahora = ahora.strftime('%H:%M:%S')
        f.writelines(f"\n{persona}, {str_ahora}")

def registrar_movimiento(persona, tipo_movimiento):
    f = open("registro_accesos.csv", "a+")
    ahora = datetime.now()
    fecha = ahora.strftime('%Y-%m-%d')
    hora = ahora.strftime('%H:%M:%S')
    f.writelines(f"\n{persona},{fecha},{hora},{tipo_movimiento}")
    f.close()

lista_personas_codificada = codificar(mis_imagenes)

def monitorear_acceso():
    captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while True:
        exito, imagen = captura.read()
        if not exito:
            print("Error al acceder a la cámara")
            break
            
        # Reducir tamaño de imagen para mejor rendimiento
        imagen_pequeña = cv2.resize(imagen, (0,0), fx=0.25, fy=0.25)
        
        # Reconocer cara en captura
        cara_captura = fr.face_locations(imagen_pequeña)
        cara_captura_codificada = fr.face_encodings(imagen_pequeña, cara_captura)
        
        for cara_codif, cara_ubic in zip(cara_captura_codificada, cara_captura):
            coincidencias = fr.compare_faces(lista_personas_codificada, cara_codif)
            distancias = fr.face_distance(lista_personas_codificada, cara_codif)
            
            if len(distancias) > 0:
                indice_coincidencia = numpy.argmin(distancias)
                
                if distancias[indice_coincidencia] < 0.6:
                    nombre = nombres_autorizados[indice_coincidencia]
                    
                    # Multiplicar ubicaciones por 4 ya que redujimos la imagen
                    y1, x2, y2, x1 = [coord * 4 for coord in cara_ubic]
                    
                    # Dibujar rectángulo y nombre
                    cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(imagen, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(imagen, nombre, (x1 + 6, y2 - 6), 
                              cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    
                    registrar_movimiento(nombre, "Entrada")
                else:
                    # Persona no autorizada
                    cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(imagen, "No Autorizado", (x1 + 6, y2 - 6),
                              cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Control de Acceso", imagen)
        
        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    captura.release()
    cv2.destroyAllWindows()
