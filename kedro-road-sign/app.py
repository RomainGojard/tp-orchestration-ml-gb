from flask import Flask, request, jsonify
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Configuration de base
KEDRO_PROJECT_NAME = "kedro-road-sign"
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
        file.save(filepath)

        # Exécution de Kedro
        try:
            project_path = os.getcwd()
            bootstrap_project(project_path)
            configure_project(KEDRO_PROJECT_NAME)

            with KedroSession.create(KEDRO_PROJECT_NAME) as session:
                context = session.load_context()
                session.run(pipeline_name=pipeline_name)

            return jsonify({
                "status": "Pipeline executed successfully",
                "pipeline": pipeline_name,
                "saved_file": saved_filename
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

