# API Comida Saludable

API sencilla en FastAPI para identificar, a partir de una imagen de comida, si el plato es "Saludable" o "No Saludable" usando un modelo preentrenado (archivo `logistic_model.pkl`).

## Resumen

Esta API expone endpoints para subir imágenes (multipart/form-data) o enviar imágenes codificadas en Base64. La imagen se procesa extrayendo un histograma (con OpenCV) y se pasa a un modelo de clasificación (LogisticRegression preentrenado) para obtener la predicción y una confianza.

Arquitectura y archivos principales:

- `app.py` - Código principal de la API (FastAPI).
- `logistic_model.pkl` - Archivo del modelo pré-entrenado (no incluido). Debe existir en la raíz del proyecto.
- `requirements.txt` - Dependencias de Python.
- `Dockerfile` - Imagen Docker para desplegar la API.

## Requisitos

- Python 3.10+ recomendado.
- Tener `logistic_model.pkl` en la raíz del proyecto (el servicio no podrá predecir sin este fichero).
- Dependencias listadas en `requirements.txt` (FastAPI, uvicorn, Pillow, numpy, opencv-python, scikit-learn, etc.).

Instalar dependencias localmente:

```powershell
python -m pip install -r requirements.txt
```

## Endpoints

- `GET /` - Información básica y endpoints disponibles.
- `POST /predict/image` - Subir imagen (multipart/form-data). Parámetro: `file`.
- `POST /predict/base64` - Enviar JSON con campo `image` que contiene la imagen codificada en Base64.
- `GET /model/status` - Devuelve si el modelo está cargado, clases y número de features si está disponible.
- `POST /model/reload` - Recarga el modelo desde `logistic_model.pkl`.
- `GET /hello/{name}` - Endpoint de prueba que devuelve un saludo.

### POST /predict/image

Request (multipart/form-data):

- `file`: fichero de imagen (image/*).

Ejemplo con curl (Linux/macOS) o usando PowerShell con curl incluido:

```powershell
curl -X POST "http://localhost:9000/predict/image" -F "file=@C:\ruta\a\tu_imagen.jpg"
```

Respuesta (JSON):

{
  "resultado": "Saludable" | "No Saludable",
  "confianza": 0.85,
  "prediccion_raw": "1",
  "metodo": "modelo_pre_entrenado"
}

### POST /predict/base64

Request (JSON):

{
  "image": "<cadena_base64_de_la_imagen>"
}

Ejemplo usando Python (requests):

```python
import base64
import requests

with open('tu_imagen.jpg','rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

resp = requests.post('http://localhost:9000/predict/base64', json={'image': b64})
print(resp.json())
```

## Despliegue

Local (uvicorn):

```powershell
uvicorn app:app --host 0.0.0.0 --port 9000
```

Con Docker:

```powershell
docker build -t api-comida:latest . ; docker run -p 9000:9000 -v ${PWD}:/app api-comida:latest
```

Notas al usar Docker:

- Asegúrate de montar el directorio que contiene `logistic_model.pkl` o copiarlo al contexto de build para que esté disponible en tiempo de ejecución.
- El `Dockerfile` expone el puerto `9000` (uvicorn), no 80.

## Consideraciones técnicas

- La extracción de features se realiza con una función `extraerhist` que calcula un histograma 2D con OpenCV sobre los canales especificados y lo aplana para usarlo como vector de características para el modelo.
- Se asume que el modelo acepta la forma de features producida por `process_image_for_prediction`.
- Si el modelo no devuelve `predict_proba`, la API usa una confianza por defecto (0.85).

## Errores y troubleshooting

- Si al iniciar la API el campo `modelo_cargado` es `false` en `/`, revisa que exista `logistic_model.pkl` en la raíz.
- Si recibe 400 al subir un archivo, confirma que `Content-Type` del archivo sea `image/*`.
- Si hay errores de OpenCV en Docker por dependencias del sistema, instala las librerías de sistema requeridas (el `Dockerfile` ya incluye las más comunes).

## Pruebas rápidas

Usa `test_main.http` si existe (clientes HTTP integrados como REST Client en VS Code) o usa curl/requests como en los ejemplos.

## Seguridad y límites

- Este servicio no implementa autenticación; no lo expongas sin un proxy/autenticación adecuada.
- Controla el tamaño máximo de los uploads si piensas usarlo en producción.

## Licencia y notas finales

Proyecto de ejemplo — ajusta y valida el pipeline de imagen/modelo antes de usar en producción.

Si quieres, puedo:

- Generar ejemplos de tests unitarios para los endpoints.
- Añadir validaciones de tamaño y tipo de imagen más estrictas.
- Crear un `docker-compose.yml` para desarrollo rápido.

---

Archivo añadido: `README.md` — documentación básica y ejemplos de uso.
