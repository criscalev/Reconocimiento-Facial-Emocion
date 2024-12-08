import os
import cv2
import numpy as np
import pickle

# Verifica si el módulo 'cv2.face' está disponible
try:
    face_module = cv2.face
except AttributeError:
    print("No se pudo acceder al módulo cv2.face. Asegúrate de tener instalada la versión correcta de opencv-contrib-python.")
    exit()

def get_images_and_labels(data_path, image_size=(64, 64)):
    image_paths = []
    labels = []
    labels_dict = {}

    person_id = 0
    for person_name in os.listdir(data_path):
        person_dir = os.path.join(data_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        # Asociar el nombre de la persona con el ID
        labels_dict[person_name] = person_id
        
        # Buscar las imágenes en el directorio de la persona
        for image_filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_filename)
            if os.path.isfile(image_path) and (image_filename.lower().endswith('.png') or image_filename.lower().endswith('.jpg')):
                image_paths.append(image_path)
                labels.append(person_id)
        person_id += 1

    # Leer imágenes y redimensionarlas a 64x64 píxeles
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, image_size)  # Redimensionar a 64x64
        images.append(img_resized)

    return images, np.array(labels), labels_dict

def train_fisherfaces(images, labels):
    model = face_module.FisherFaceRecognizer_create()
    model.train(images, labels)
    return model

# Define el directorio de la base de datos
data_path = 'C:/Users/USUARIO/Desktop/redes/BD_Rostros'

# Llamar a la función y obtener las imágenes, etiquetas y el diccionario de etiquetas
images, labels, labels_dict = get_images_and_labels(data_path)

# Entrenar el modelo FisherFaces
model = train_fisherfaces(images, labels)

# Directorio para guardar el modelo y el diccionario de etiquetas
output_dir = 'C:/Users/USUARIO/Desktop/redes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Guardar el modelo
model_path = os.path.join(output_dir, 'fisherface_model.xml')
model.write(model_path)

# Guardar el diccionario de etiquetas
labels_dict_path = os.path.join(output_dir, 'labels_dict2.pkl')
with open(labels_dict_path, 'wb') as file:
    pickle.dump(labels_dict, file)

print(labels_dict)
print(f"Modelo FisherFace guardado en: {model_path}")
print(f"Diccionario de etiquetas guardado en: {labels_dict_path}")
