terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = "us-west1"
}

variable "project_id" {
  description = "The GCP Project ID"
  type        = string
}

variable "cloud_run_url" {
  description = "The URL of the deployed WFC Cloud Run worker"
  type        = string
}

# 1. Storage Buckets
resource "google_storage_bucket" "wfc_inputs" {
  name          = "wfc-inputs-${var.project_id}"
  location      = "us-west1"
  force_destroy = true
}

resource "google_storage_bucket" "wfc_outputs" {
  name          = "wfc-outputs-${var.project_id}"
  location      = "us-west1"
  force_destroy = true
}

# Make the output bucket publicly readable for the web frontend
resource "google_storage_bucket_iam_member" "public_read" {
  bucket = google_storage_bucket.wfc_outputs.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# 2. Firestore Database
resource "google_firestore_database" "wfc_db" {
  name        = "wfc-db"
  location_id = "us-west1"
  type        = "FIRESTORE_NATIVE"
}

# 3. Pub/Sub Message Queue
resource "google_pubsub_topic" "wfc_queue" {
  name = "wfc-work-queue"
}

# 4. Security Identity
resource "google_service_account" "pubsub_invoker" {
  account_id   = "wfc-pubsub-invoker"
  display_name = "Pub/Sub Cloud Run Invoker ID"
}

resource "google_cloud_run_v2_service_iam_member" "invoker_binding" {
  project  = var.project_id
  location = "us-west1"
  name     = "wfc-worker" 
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.pubsub_invoker.email}"
}

# 5. Push Subscription
resource "google_pubsub_subscription" "wfc_push_sub" {
  name  = "wfc-worker-sub"
  topic = google_pubsub_topic.wfc_queue.name

  ack_deadline_seconds = 600 

  push_config {
    push_endpoint = var.cloud_run_url
    
    oidc_token {
      service_account_email = google_service_account.pubsub_invoker.email
    }
  }
}

# Public inputs bucket
resource "google_storage_bucket_iam_member" "public_read_inputs" {
  bucket = google_storage_bucket.wfc_inputs.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}
