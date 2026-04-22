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

# ------------------------------------------------------------------------
# VARIABLES & DATA SOURCES
# ------------------------------------------------------------------------
variable "project_id" {
  description = "The GCP Project ID"
  type        = string
}

variable "cloud_run_url" {
  description = "The URL of the deployed WFC Cloud Run worker"
  type        = string
}

variable "github_repo" {
  description = "The GitHub repository in the format username/repo (e.g., octocat/hello-world)"
  type        = string
}

# Fetch the project data to automatically get the numerical Project Number
data "google_project" "project" {
  project_id = var.project_id
}

# ------------------------------------------------------------------------
# ENABLE REQUIRED APIS
# ------------------------------------------------------------------------
locals {
  services = [
    "appengine.googleapis.com",      # To manage App Engine deployments
    "cloudbuild.googleapis.com",     # Used by App Engine to build the code
    "run.googleapis.com",            # To manage Cloud Run deployments
    "firestore.googleapis.com",      # For the database
    "pubsub.googleapis.com",         # For the message queue
    "iamcredentials.googleapis.com", # CRITICAL for Workload Identity Federation
  ]
}

resource "google_project_service" "enabled_apis" {
  for_each           = toset(local.services)
  project            = var.project_id
  service            = each.key
  
  # Prevents Terraform from accidentally disabling these APIs if you run 'destroy'
  disable_on_destroy = false 
}

# ------------------------------------------------------------------------
# 1. STORAGE BUCKETS
# ------------------------------------------------------------------------
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
resource "google_storage_bucket_iam_member" "public_read_outputs" {
  bucket = google_storage_bucket.wfc_outputs.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# Make the input bucket publicly readable for the web frontend dashboard
resource "google_storage_bucket_iam_member" "public_read_inputs" {
  bucket = google_storage_bucket.wfc_inputs.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# ------------------------------------------------------------------------
# 2. FIRESTORE DATABASE
# ------------------------------------------------------------------------
resource "google_firestore_database" "wfc_db" {
  name        = "wfc-db"
  location_id = "us-west1"
  type        = "FIRESTORE_NATIVE"
}

# ------------------------------------------------------------------------
# 3. PUB/SUB MESSAGE QUEUE
# ------------------------------------------------------------------------
resource "google_pubsub_topic" "wfc_queue" {
  name = "wfc-work-queue"
}

# ------------------------------------------------------------------------
# 4. CLOUD RUN SECURITY (ZERO-TRUST)
# ------------------------------------------------------------------------
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

# Push Subscription
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

# ------------------------------------------------------------------------
# 5. GITHUB ACTIONS CI/CD (WORKLOAD IDENTITY FEDERATION)
# ------------------------------------------------------------------------
# Create the Service Account for GitHub Actions
resource "google_service_account" "github_actions" {
  account_id   = "github-actions-sa"
  display_name = "GitHub Actions Deployer"
}

# Grant the Service Account the necessary deployment roles
locals {
  deploy_roles = [
    "roles/appengine.appAdmin",
    "roles/run.admin",
    "roles/iam.serviceAccountUser",
    "roles/cloudbuild.builds.editor",
    "roles/storage.admin"
  ]
}

resource "google_project_iam_member" "github_actions_roles" {
  for_each = toset(local.deploy_roles)
  project  = var.project_id
  role     = each.key
  member   = "serviceAccount:${google_service_account.github_actions.email}"
}

# Create the Workload Identity Pool
resource "google_iam_workload_identity_pool" "github_pool" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions Pool"
}

# Create the OIDC Provider in the Pool
resource "google_iam_workload_identity_pool_provider" "github_provider" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub Provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }

  # This acts as the firewall rule. It MUST use 'assertion'
  attribute_condition = "assertion.repository == '${var.github_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Bind the Service Account to the specific GitHub repository via the Pool
resource "google_service_account_iam_member" "github_actions_wif_bind" {
  service_account_id = google_service_account.github_actions.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/${var.github_repo}"
}

# ------------------------------------------------------------------------
# OUTPUTS (For GitHub Actions YAML variables)
# ------------------------------------------------------------------------
output "github_actions_service_account" {
  description = "Copy this into your GitHub Actions YAML (service_account)"
  value       = google_service_account.github_actions.email
}

output "workload_identity_provider" {
  description = "Copy this into your GitHub Actions YAML (workload_identity_provider)"
  value       = "projects/${data.google_project.project.number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github_pool.workload_identity_pool_id}/providers/${google_iam_workload_identity_pool_provider.github_provider.workload_identity_pool_provider_id}"
}
