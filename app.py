from ultralytics import RTDETR
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# Inicializar Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# Cargar el modelo entrenado
model = RTDETR("best.pt") 

# Ruta de prueba
@app.route('/test', methods=['GET'])
def hello_world():
    texto = "Me encantó este producto, es excelente"

    return jsonify({'message': texto})


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Endpoint para realizar predicciones con una imagen enviada como multipart/form-data.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No se proporcionó un archivo de imagen"}), 400

    try:
        # Leer la imagen desde la solicitud
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))

        # Realizar predicción
        results = model.predict(image)

        # Obtener el número predicho
        if len(results) > 0:
            result = results[0]  # Tomamos el primer resultado
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                # Asumimos una sola predicción en la imagen
                box = result.boxes.data[0]  # Tomamos la primera caja
                class_id = int(box[-1])  # ID de clase (el número)
                print(f"Número predicho: {class_id}")
            else:
                print("No se detectaron números en la imagen.")
        else:
            print("No se generaron resultados de predicción.")


        # Formatear y devolver la respuesta
        return jsonify({
            "number": class_id,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
