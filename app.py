from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
import pickle
import os
from typing import List

app = FastAPI(title="API Comida Saludable", description="API para predicción de comida saludable usando modelo pre-entrenado")

# Cargar modelo pre-entrenado al iniciar la aplicación
logistic_model = None
scaler = None
label_encoder = None

def load_pretrained_model():
    """Carga el modelo pre-entrenado al iniciar la aplicación"""
    global logistic_model, scaler, label_encoder
    try:
        if os.path.exists("logistic_model.pkl"):
            with open("logistic_model.pkl", "rb") as f:
                logistic_model = pickle.load(f)
            print("Modelo pre-entrenado cargado exitosamente")
        else:
            print("Archivo logistic_model.pkl no encontrado")
    except Exception as e:
        print(f"Error cargando modelo pre-entrenado: {e}")

def extraerhist(ruta_imagen):
    """Función para extraer histograma de imagen (basada en tu código)"""
    if isinstance(ruta_imagen, str):
        image = cv2.imread(ruta_imagen)
    else:
        # Si es un array de numpy (imagen ya leída)
        image = ruta_imagen
    
    image = cv2.resize(image, (256, 256))
    hist = cv2.calcHist([image], [1, 2], None, [256, 256], [0, 256, 0, 256])
    return hist

def process_image_for_prediction(image_array):
    """Procesa imagen para predicción usando la función extraerhist"""
    try:
        # Convertir PIL a array numpy si es necesario
        if hasattr(image_array, 'mode'):
            image_array = np.array(image_array)
        
        # Convertir RGB a BGR para OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Extraer histograma
        hist = extraerhist(image_array)
        
        # Aplanar histograma para usar como features
        features = hist.reshape(1, -1)
        
        return features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

# Cargar modelo al iniciar
load_pretrained_model()

@app.get("/")
async def root():
    return {
        "message": "API Comida Saludable", 
        "modelo_cargado": logistic_model is not None,
        "endpoints": [
            "POST /predict/image - Subir imagen de comida",
            "POST /predict/base64 - Enviar imagen en base64",
            "GET /model/status - Estado del modelo"
        ]
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predice si una imagen de comida es saludable usando el modelo pre-entrenado"""
    try:
        # Validar que sea una imagen
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Verificar que el modelo esté cargado
        if logistic_model is None:
            raise HTTPException(status_code=500, detail="Modelo no disponible. Verifique que logistic_model.pkl existe en el directorio")
        
        # Leer la imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convertir a RGB si no lo está
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Procesar imagen para predicción
        features = process_image_for_prediction(image)
        
        # Realizar predicción
        prediction = logistic_model.predict(features)[0]
        
        # Obtener probabilidades si está disponible
        try:
            probabilities = logistic_model.predict_proba(features)[0]
            confidence = max(probabilities)
        except:
            confidence = 0.85  # Confianza por defecto
        
        # Mapear predicción a texto
        if hasattr(logistic_model, 'classes_'):
            classes = logistic_model.classes_
            if len(classes) == 2:
                resultado = "Saludable" if prediction == 1 or prediction == "saludable" else "No Saludable"
            else:
                resultado = str(prediction)
        else:
            resultado = "Saludable" if prediction == 1 else "No Saludable"
        
        return {
            "resultado": resultado,
            "confianza": float(confidence),
            "prediccion_raw": str(prediction),
            "metodo": "modelo_pre_entrenado"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.post("/predict/base64")
async def predict_base64(data: dict):
    """Predice usando imagen en base64"""
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Campo 'image' requerido con datos base64")
        
        if logistic_model is None:
            raise HTTPException(status_code=500, detail="Modelo no disponible")
        
        # Decodificar base64
        try:
            image_data = base64.b64decode(data["image"])
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Datos base64 inválidos")
        
        # Convertir a RGB si no lo está
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Procesar y predecir
        features = process_image_for_prediction(image)
        prediction = logistic_model.predict(features)[0]
        
        try:
            probabilities = logistic_model.predict_proba(features)[0]
            confidence = max(probabilities)
        except:
            confidence = 0.85
        
        # Mapear resultado
        resultado = "Saludable" if prediction == 1 or str(prediction).lower() == "saludable" else "No Saludable"
        
        return {
            "resultado": resultado,
            "confianza": float(confidence),
            "prediccion_raw": str(prediction),
            "metodo": "modelo_pre_entrenado"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen base64: {str(e)}")

@app.get("/model/status")
async def get_model_status():
    """Obtiene información sobre el modelo cargado"""
    status = {
        "modelo_cargado": logistic_model is not None,
        "archivo_modelo": os.path.exists("logistic_model.pkl"),
        "tipo_modelo": str(type(logistic_model).__name__) if logistic_model else None
    }
    
    if logistic_model is not None:
        try:
            status["clases"] = list(logistic_model.classes_) if hasattr(logistic_model, 'classes_') else None
            status["numero_features"] = getattr(logistic_model, 'n_features_in_', None)
        except:
            pass
    
    return status

@app.post("/model/reload")
async def reload_model():
    """Recarga el modelo desde el archivo"""
    try:
        load_pretrained_model()
        return {
            "mensaje": "Modelo recargado",
            "modelo_cargado": logistic_model is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recargando modelo: {str(e)}")

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
