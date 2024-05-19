import moviepy.editor as mp
from PIL import Image
import numpy as np
import tensorflow as tf

def predict_video_class(video_path, model):
    # Charger le modèle
    model = tf.keras.models.load_model(model)
    
    # Charger la vidéo
    video_clip = mp.VideoFileClip(video_path)
    
    # Prétraiter chaque image de la vidéo
    frames = []
    for frame in video_clip.iter_frames():
        # Convertir l'image en tableau numpy
        img = Image.fromarray(frame)
        # Redimensionner l'image en 250x250
        img = img.resize((250, 250))
        # Convertir l'image en tableau numpy et normaliser les valeurs de pixel
        img_array = np.array(img) / 255.0
        # Ajouter l'image prétraitée à la liste des images
        frames.append(img_array)
    
    # Convertir la liste d'images en tableau numpy
    frames = np.array(frames)
    
    # Faire les prédictions avec le modèle
    predictions = model.predict(frames)
    
    # Calculer la moyenne des prédictions pour chaque classe
    class1_avg = np.mean(predictions[:, 0])
    class2_avg = np.mean(predictions[:, 1])
    
    # Déterminer la classe prédite en fonction de la moyenne des prédictions
    predicted_class = "old" if class1_avg > class2_avg else 'young'
    
    return predicted_class

# Utilisation de la fonction
video_path = "testing/video.mp4"
model_path = "testing/model.h5"
predicted_class = predict_video_class(video_path, model_path)
print("Classe prédite :", predicted_class)
