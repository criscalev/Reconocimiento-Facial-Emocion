import cv2
import pickle
import numpy as np

# Cargar el modelo de Eigenfaces y el diccionario de etiquetas
model_path = 'C:/Users/USUARIO/Desktop/redes/fisherface_model.xml'
labels_dict_path = 'C:/Users/USUARIO/Desktop/redes/labels_dict2.pkl'

model = cv2.face.EigenFaceRecognizer_create()
model.read(model_path)

# Cargar el diccionario de etiquetas
with open(labels_dict_path, 'rb') as f:
    labels_dict = pickle.load(f)
    # Mostrar el diccionario para asegurarse de que se cargó correctamente
    for label_name, label_id in labels_dict.items():
        print(f"{label_name}: {label_id}")

# Inicializar la cámara
cap = cv2.VideoCapture(1)

# Definir el tamaño de las imágenes que el modelo espera (tamaño de entrenamiento)
image_size = (64, 64)  # Ajusta esto según el tamaño usado en el entrenamiento

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraer la región de interés (ROI) para el rostro
        roi_gray = gray[y:y+h, x:x+w]

        # Redimensionar la imagen al tamaño correcto (ajusta el tamaño según tu modelo)
        roi_gray_resized = cv2.resize(roi_gray, image_size)  # Redimensionar a (64, 64) o el tamaño que usaste

        # Predecir el rostro usando el modelo
        label, confidence = model.predict(roi_gray_resized)

        # Asumimos que la confianza normalmente está entre 1000 y 5000
        # Definimos un umbral para la confianza baja (predicción confiable)
        confidence_threshold = 30  # Ajusta según tu modelo

        if confidence <= confidence_threshold:
            # Crear la clave para acceder al diccionario con el formato 'label_nombre'
            person_name = [name for name, id_ in labels_dict.items() if id_ == label][0]
        else:
            person_name = 'Desconocido'

        # Dibujar el rectángulo alrededor del rostro y poner el nombre
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Mostrar el nombre de la persona
        cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Mostrar el valor de la confianza en la imagen
        confidence_text = f'Confianza: {confidence:.2f}'  # Mostrar la confianza con dos decimales
        cv2.putText(frame, confidence_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Muestra siempre el nombre predicho para verificar
        print(f"Nombre predicho: {person_name}, Label: {label}, Confianza: {confidence}")

    # Mostrar la imagen con las predicciones
    cv2.imshow('Reconocimiento Facial', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
