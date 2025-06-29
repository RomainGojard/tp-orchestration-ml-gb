#!/usr/bin/env python3
"""
stress_test.py
Envoie N requêtes concurrentes à l'endpoint /predict d'une application Flask déployée sur Kubernetes.
"""
import argparse
import concurrent.futures
import time
import requests
from io import BytesIO
from PIL import Image
import os

# ------------------------------
# Configuration à adapter
# ------------------------------
URL = "http://localhost:8000/run-pipeline" # Ex. obtenu via `minikube service --url`
IMAGE_PATH = "/Users/romaingojard/Desktop/M2_ESGI/orchestrationML/tp-orchestration-ml-gb/test_image1.jpg"
NUM_REQUESTS = 100
MAX_WORKERS = 10
# ------------------------------

def send_request(_):
  """Charge l'image et poste vers l'API avec le payload de la bonne pipeline."""
  url = "https://m1.direct-signaletique.com/30023-thickbox_default/panneau-d-entree-de-ville-eb10-2-lignes-de-texte-hauteur-de-caracteres-100-mm.jpg"
  response = requests.get(url)
  with Image.open(BytesIO(response.content)) as img:
    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    files = {'file': ('test_image.jpg', buf, 'image/jpeg')}
    data = {'pipeline_name': 'prediction'}
    start = time.time()
    resp = requests.post(URL, files=files, data=data)
    latency = time.time() - start
  return resp.status_code, latency

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--url', default=URL, help="URL de l'API /predict")
  parser.add_argument('--requests', type=int, default=NUM_REQUESTS, help="Nombre total de requêtes")
  parser.add_argument('--workers', type=int, default=MAX_WORKERS, help="Nombre de threads")
  args = parser.parse_args()
  latencies = []
  statuses = []
  print(f"Lancement de {args.requests} requêtes vers {args.url} avec {args.workers} workers…")
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as exe:
    futures = [exe.submit(send_request, i) for i in range(args.requests)]
    for f in concurrent.futures.as_completed(futures):
      status, lat = f.result()
      statuses.append(status)
      latencies.append(lat)
  print("Statuts reçus:", set(statuses))
  print(f"Latence moyenne : {sum(latencies)/len(latencies):.3f}s")
  print(f"Latence p95 : {sorted(latencies)[int(0.95*len(latencies))-1]:.3f}s")

if __name__ == "__main__":
  main()