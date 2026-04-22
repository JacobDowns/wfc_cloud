import os
import uuid
import json
import datetime
from flask import Flask, render_template, request, jsonify
from google.cloud import storage, pubsub_v1, firestore
from google.cloud.exceptions import NotFound
from PIL import Image
import io

TERMINAL_STATUSES = {"COMPLETE", "ERROR", "TIMED OUT", "CANCELLED"}

app = Flask(__name__)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
INPUT_BUCKET = f"wfc-inputs-{PROJECT_ID}"
OUTPUT_BUCKET = f"wfc-outputs-{PROJECT_ID}"
DB_NAME = os.environ.get("FIRESTORE_DB_NAME", "wfc-db")

storage_client = storage.Client()
publisher = pubsub_v1.PublisherClient()
TOPIC_PATH = publisher.topic_path(PROJECT_ID, "wfc-work-queue")
firestore_client = firestore.Client(database=DB_NAME)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'seed_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['seed_image']
    patch_size = int(request.form.get('patch_size', 3))
    output_size = int(request.form.get('output_size', 64))
    
    image_bytes = file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width > 128 or img.height > 128:
            return jsonify({"error": f"Image is {img.width}x{img.height}. Max allowed is 128x128."}), 400
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    filename = f"{job_id}.png"

    # Upload Input
    bucket = storage_client.bucket(INPUT_BUCKET)
    blob = bucket.blob(filename)
    blob.upload_from_string(image_bytes, content_type=file.content_type)
    
    input_url = f"https://storage.googleapis.com/{INPUT_BUCKET}/{filename}"

    # Create Job Document with Timestamp
    doc_ref = firestore_client.collection("wfc_jobs").document(job_id)
    doc_ref.set({
        "status": "PENDING",
        "patch_size": patch_size,
        "output_size": output_size,
        "input_url": input_url,
        "output_url": None,
        "progress": 0.0,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    # Dispatch to Queue
    work_order = {
        "job_id": job_id,
        "input_bucket": INPUT_BUCKET,
        "input_filename": filename,
        "output_bucket": OUTPUT_BUCKET,
        "patch_size": patch_size,
        "output_size": output_size
    }
    
    publisher.publish(TOPIC_PATH, json.dumps(work_order).encode("utf-8"))
    return jsonify({"success": True})

# NEW ROUTE: For the Dashboard to fetch recent jobs
@app.route('/jobs', methods=['GET'])
def get_jobs():
    # Get the 12 most recent jobs
    docs = firestore_client.collection("wfc_jobs").order_by(
        "timestamp", direction=firestore.Query.DESCENDING
    ).limit(12).stream()

    jobs = []
    for d in docs:
        data = d.to_dict()
        data['id'] = d.id
        # Convert timestamp object to string for JSON serialization
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = data['timestamp'].strftime("%H:%M:%S")
        jobs.append(data)

    return jsonify(jobs)


@app.route('/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Mark a PENDING job as CANCELLED. The worker checks this flag before
    starting work and will ack-and-skip if cancelled. Only takes effect if the
    worker has not yet picked up the message; in-flight cancellation will arrive
    in a later change."""
    doc_ref = firestore_client.collection("wfc_jobs").document(job_id)
    snapshot = doc_ref.get()
    if not snapshot.exists:
        return jsonify({"error": "Job not found"}), 404

    current_status = snapshot.get('status')
    if current_status != 'PENDING':
        return jsonify({"error": f"Cannot cancel a job in status {current_status}"}), 409

    doc_ref.update({"status": "CANCELLED"})
    return jsonify({"success": True})


@app.route('/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a terminal-state job: removes the Firestore doc plus its input
    and output blobs in GCS. Refuses to delete a job that's still pending or
    running (the user should cancel first)."""
    doc_ref = firestore_client.collection("wfc_jobs").document(job_id)
    snapshot = doc_ref.get()
    if not snapshot.exists:
        return jsonify({"error": "Job not found"}), 404

    current_status = snapshot.get('status')
    if current_status not in TERMINAL_STATUSES:
        return jsonify({"error": f"Cannot delete a job in status {current_status}; cancel it first"}), 409

    # Best-effort GCS cleanup. Object names follow the upload/worker conventions.
    for bucket_name, blob_name in (
        (INPUT_BUCKET, f"{job_id}.png"),
        (OUTPUT_BUCKET, f"generated-{job_id}.png"),
    ):
        try:
            storage_client.bucket(bucket_name).blob(blob_name).delete()
        except NotFound:
            pass  # already gone — not a problem

    doc_ref.delete()
    return jsonify({"success": True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)