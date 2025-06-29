from flask import Flask, request, jsonify
from flask_cors import CORS
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration de base
KEDRO_PROJECT_NAME = "kedro_road_sign"
UPLOAD_FOLDER = os.path.join("data", "01_raw", "images", "user_uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Kedro Road Sign API!"})

@app.route('/run-pipeline', methods=['POST'])
def run_pipeline():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    pipeline_name = request.form.get("pipeline_name")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not pipeline_name:
        return jsonify({"error": "Missing pipeline_name in form data"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
        #remove all other files in the upload folder
        for f in os.listdir(app.config["UPLOAD_FOLDER"]):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], f)
            if os.path.isfile(file_path) and f != saved_filename:
                os.remove(file_path)
        # Enregistrement du fichier uploadé
        file.save(filepath)

        # Exécution de Kedro
        try:
            project_path = "/home/kedro_road_sign"
            os.chdir(project_path)  # ✅ Forcer le bon contexte

            bootstrap_project(project_path)

            # ✅ Récupération du nom du projet dynamiquement
            import importlib
            pyproject = importlib.import_module("kedro_road_sign")
            configure_project(pyproject.__name__)

            # ✅ Création de session sans nom (détection auto)
            with KedroSession.create() as session:
                context = session.load_context()
                session.run(pipeline_name=pipeline_name)
                
            if pipeline_name == "use_cases":
                # renvoyer les résultats de la prédiction YOLO
                yolo_predict_output_file = f"data/07_model_output/ocr_results/{saved_filename.split('.')[0]}.txt"

            return jsonify({
                "status": "Pipeline executed successfully",
                "pipeline": pipeline_name,
                "saved_file": saved_filename,
                "yolo_predict_output": yolo_predict_output_file if pipeline_name == "use_cases" else None
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    else:
        return jsonify({"error": "File type not allowed"}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

