import os
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from PIL import Image, ImageOps
import numpy as np
import io
import base64
from tensorflow.keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D to handle 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

get_custom_objects()['CustomDepthwiseConv2D'] = CustomDepthwiseConv2D

# Load models and labels
try:
    MODEL_1_PATH = r'D:\BEE FYP Folder\latest V5-Honey bee or Not FOLDER\v5_converted_keras\keras_model.h5'
    LABELS_1_PATH = r"D:\BEE FYP Folder\latest V5-Honey bee or Not FOLDER\v5_converted_keras\labels.txt"

    MODEL_2_PATH = r'D:\BEE FYP Folder\Disease Model Versions\working Diseases v_2 converted_keras\keras_model (1).h5'
    LABELS_2_PATH = r"D:\BEE FYP Folder\Disease Model Versions\working Diseases v_2 converted_keras\labels (1).txt.txt"

    # Load models
    model_1 = load_model(MODEL_1_PATH, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    model_2 = load_model(MODEL_2_PATH, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)

    # Load labels
    class_names_1 = [line.strip() for line in open(LABELS_1_PATH, "r").readlines()]
    class_names_2 = [line.strip() for line in open(LABELS_2_PATH, "r").readlines()]
except Exception as e:
    raise RuntimeError(f"Error loading models or labels: {e}")

def create_confidence_plot(predictions, class_names, title):
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, predictions[0])
        plt.ylabel('Confidence')
        plt.xlabel('Classes')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        buf.close()
        return plot_data
    except Exception as e:
        raise RuntimeError(f"Error generating plot: {e}")

def preprocess_image(image):
    try:
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image)
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        return data
    except Exception as e:
        raise RuntimeError(f"Error during image preprocessing: {e}")

class PredictView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # Ensure file is uploaded
        if 'image' not in request.FILES:
            return Response({"error": "No image uploaded. Please upload an image file."}, status=400)

        # Load and preprocess image
        try:
            uploaded_file = request.FILES['image']
            image = Image.open(uploaded_file).convert("RGB")
            processed_data = preprocess_image(image)
        except Exception as e:
            return Response({"error": f"Error processing uploaded image: {str(e)}"}, status=400)

        try:
            # First model prediction
            prediction_1 = model_1.predict(processed_data)
            index_1 = np.argmax(prediction_1)
            class_name_1 = class_names_1[index_1].strip()
            confidence_1 = float(prediction_1[0][index_1])

            # Create confidence plot for first model
            initial_plot = create_confidence_plot(
                prediction_1,
                class_names_1,
                'Initial Classification Confidence'
            )

            # Check if the first model identifies the image as "Honey Bee or Beehive"
            if index_1 == 0:  # Class 0: Honey Bee or Beehive
                # Proceed to the second model for disease classification
                prediction_2 = model_2.predict(processed_data)
                index_2 = np.argmax(prediction_2)
                class_name_2 = class_names_2[index_2].strip()
                confidence_2 = float(prediction_2[0][index_2])

                # Create disease prediction plot
                disease_plot = create_confidence_plot(
                    prediction_2,
                    class_names_2,
                    'Disease Prediction Confidence'
                )

                return Response({
                    "prediction": class_name_2,
                    "confidence": confidence_2,
                    "initial_plot": initial_plot,
                    "disease_plot": disease_plot,
                    "is_bee": True
                })

            # If class is 1: Not Honey Bee or Not Beehive
            return Response({
                "prediction": "Not a Honey Bee or Beehive",
                "confidence": confidence_1,
                "initial_plot": initial_plot,
                "is_bee": False
            })
        except Exception as e:
            return Response({"error": f"Prediction error: {str(e)}"}, status=500)
