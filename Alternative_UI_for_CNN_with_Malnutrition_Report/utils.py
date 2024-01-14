import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model

# Labels Dictionaries for CNN Models

malnutrition_type_labels_dict = {
    "healthy" : 0,
    "kwashiorkor" : 1,
    "marasmus" : 2
}

dry_skin_vitamin_b7_iron_deficiency_labels_dict = {
    'healthy_skin' : 0,
    'dry_skin' : 1
}

vitamin_a_deficiency_labels_dict = {
    "healthy_eyes" : 0,
    "vitamin_a_deficiency" : 1
}

iodine_deficiency_labels_dict = {
    "no_iodine_deficiency" : 0,
    "iodine_deficiency" : 1
}

# Function to make predictions, this function would be called inside the predict function (i.e, function that is being called by the Gradio UI) & it would be used for making predictions for the CNN model (Predictor) chosen on the Gradio UI for the Uploaded image by the user

def prediction(uploaded_image_path, loaded_model, model_labels_dict):
    image = tf.keras.preprocessing.image.load_img(uploaded_image_path, target_size=(180, 180))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_scaled = image_array / 255.0

    # Make predictions using the loaded model
    predictions = loaded_model.predict(image_scaled)

    # Initially we don't know the class / label of the image
    predicted_class = "Unknown"

    # Convert predictions to numerical class labels using argmax
    predicted_class_index = np.argmax(predictions)
    
    # Display the predicted class
    for class_name, class_num_label in model_labels_dict.items():
        if class_num_label == predicted_class_index:
            # Get the predicted class label name
            predicted_class = class_name # Here the value of the predicted_class variable gets updated with the class name to which the image actually belongs

    return image, predicted_class

# Storing Paths of CNN models in Variables

malnutrition_type_model = "C:\\Users\\OMOLP094\\Desktop\\GitHub Repos Of Projects\\Malnutrition-and-Deficiencies-Detection-Using-CNNs\\CNN_Models_for_detecting_Malnutrition_and_Deficiencies\\malnutrition_type_CNN_model\\malnutrition_type_classification_first_CNN_model.h5"

skin_condition_model = "C:\\Users\\OMOLP094\\Desktop\\GitHub Repos Of Projects\\Malnutrition-and-Deficiencies-Detection-Using-CNNs\\CNN_Models_for_detecting_Malnutrition_and_Deficiencies\\dry_healthy_skin_CNN_model\\dry_healthy_skin_classification_first_CNN_model.h5"

eye_condition_model = "C:\\Users\\OMOLP094\\Desktop\\GitHub Repos Of Projects\\Malnutrition-and-Deficiencies-Detection-Using-CNNs\\CNN_Models_for_detecting_Malnutrition_and_Deficiencies\\vitamin_a_deficiency_CNN_model\\vitamin_A_deficient_eyes_classification_first_CNN_model.h5"

iodine_deficiency_model = "C:\\Users\\OMOLP094\\Desktop\\GitHub Repos Of Projects\\Malnutrition-and-Deficiencies-Detection-Using-CNNs\\CNN_Models_for_detecting_Malnutrition_and_Deficiencies\\iodine_deficiency_CNN_model\\healthy_iodine_deficient_classification_first_CNN_model.h5"

# List containing Paths of CNN models & their Labels Dictionaries

model_classes_list = [[malnutrition_type_model, malnutrition_type_labels_dict],
                      [skin_condition_model, dry_skin_vitamin_b7_iron_deficiency_labels_dict],
                      [eye_condition_model, vitamin_a_deficiency_labels_dict],
                      [iodine_deficiency_model, iodine_deficiency_labels_dict]]

# Function to be called by the Gradio UI

def predict(uploaded_image):
    uploaded_image_path = uploaded_image.name
    malnutrition_report = ""

    for model in model_classes_list:
        model_path = model[0]
        model_labels_dict = model[1]
        loaded_model = load_model(model_path)
        # Call the prediction function
        uploaded_image, predicted_class = prediction(uploaded_image_path, loaded_model, model_labels_dict)
        malnutrition_report = malnutrition_report + "\n" + predicted_class
        malnutrition_report = malnutrition_report.replace("_", " ") # removing all the underscores from the malnutrition report string as it contains the names of predicted classes which have "_" in them
        malnutrition_report = malnutrition_report.title() # Converting the malnutrition report string into title case

    return uploaded_image, malnutrition_report
