 # Cliente de predicción
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
 # Clase llave para Azure
from msrest.authentication import ApiKeyCredentials
 # dotenv para cargar la clave
from dotenv import load_dotenv
 # Importar el módulo os para leer las variables de entorno
import os

 # Carga los valores de clave y endpoint
load_dotenv()
    
 # Establece los valores en variables
key = os.getenv('KEY')
endpoint = os.getenv('ENDPOINT')
project_id = os.getenv('PROJECT_ID') 
published_name = os.getenv('PUBLISHED_ITERATION_NAME')

 # Configura las credenciales para el cliente
credentials = ApiKeyCredentials(in_headers={'Prediction-key':key})
    
 # Crea el cliente, el cual se utilizará para hacer predicciones
client = CustomVisionPredictionClient(endpoint, credentials)
    
 # Abre el archivo de prueba
with open('../testing-images/american-staffordshire-terrier-10.jpg', 'rb') as image:
     # Realiza la predicción
     results = client.classify_image(project_id, published_name, image.read())
    
     # Debido a que podría haber múltiples predicciones, recorremos cada una de ellas.
     for prediction in results.predictions:
         # Muestra el nombre de la raza y el porcentaje de probabilidad
         print(f'{prediction.tag_name}: {(prediction.probability):.2%}')
