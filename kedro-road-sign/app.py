from flask import Flask, request, jsonify
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project
import os

app = Flask(__name__)

KEDRO_PROJECT_NAME = "kedro_road_sign"

@app.route('/run-pipeline', methods=['POST'])
def run_pipeline():
    try:
        # Lire le nom de la pipeline depuis la requête
        data = request.get_json()
        pipeline_name = data.get("pipeline_name")

        if not pipeline_name:
            return jsonify({"error": "Missing 'pipeline_name' in request body"}), 400

        # Bootstrap Kedro
        project_path = os.getcwd()
        bootstrap_project(project_path)
        configure_project(KEDRO_PROJECT_NAME)

        # Créer une session Kedro et exécuter la pipeline
        with KedroSession.create(KEDRO_PROJECT_NAME) as session:
            context = session.load_context()
            session.run(pipeline_name=pipeline_name)

        return jsonify({"status": "Pipeline executed successfully", "pipeline": pipeline_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
