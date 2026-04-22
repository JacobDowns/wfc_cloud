import base64
import json
import os
import io
import sys
import time
import multiprocessing
import concurrent.futures
from flask import Flask, request
from google.cloud import storage, firestore
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit
from PIL import Image


def log(job_id, msg):
    """Print a job-id-prefixed line that Cloud Logging can parse per-job."""
    print(f"[{job_id}] {msg}", file=sys.stdout, flush=True)

app = Flask(__name__)

# Cloud Clients (parent process only — the subprocess creates its own).
DB_NAME = os.environ.get("FIRESTORE_DB_NAME", "wfc-db")
storage_client = storage.Client()
firestore_client = firestore.Client(database=DB_NAME)

# Use spawn for the solver subprocess: the subprocess touches Firestore for
# progress/cancel updates, and google-cloud's gRPC clients can't be safely
# forked from a parent that already initialized them. Spawn costs ~1s to
# re-import the module but produces clean isolation.
_MP_CTX = multiprocessing.get_context("spawn")

# ------------------------------------------------------------------------
# 1. PATTERN EXTRACTION (Vectorized + Numba)
# ------------------------------------------------------------------------
@njit(cache=True)
def _compute_rules(patterns):
    """Compute the 4-direction adjacency rule table for a set of unique patterns.

    Only directions 0 (up) and 1 (right) are compared directly; 2 (down) and 3 (left)
    are filled by symmetry, halving the O(P^2) comparison work.

    Note: single-threaded on purpose. The Numba workqueue threading layer is not
    threadsafe across OS threads, and gunicorn runs this handler with multiple
    threads. A concurrent call from two requests used to kill the worker. The
    pure-Numba compile already provides the bulk of the speedup; parallelism on
    top was marginal.
    """
    P = patterns.shape[0]
    N = patterns.shape[1]
    rules = np.zeros((P, 4, P), dtype=np.bool_)
    for i in range(P):
        for j in range(P):
            # dir 0: p1[1:, :] == p2[:-1, :]
            match0 = True
            for a in range(N - 1):
                for b in range(N):
                    for c in range(3):
                        if patterns[i, a + 1, b, c] != patterns[j, a, b, c]:
                            match0 = False
                            break
                    if not match0:
                        break
                if not match0:
                    break
            rules[i, 0, j] = match0

            # dir 1: p1[:, 1:] == p2[:, :-1]
            match1 = True
            for a in range(N):
                for b in range(N - 1):
                    for c in range(3):
                        if patterns[i, a, b + 1, c] != patterns[j, a, b, c]:
                            match1 = False
                            break
                    if not match1:
                        break
                if not match1:
                    break
            rules[i, 1, j] = match1

    for i in range(P):
        for j in range(P):
            rules[i, 2, j] = rules[j, 0, i]
            rules[i, 3, j] = rules[j, 1, i]
    return rules


def extract_patterns_and_rules(image_array, N=3):
    H, W = image_array.shape[:2]
    padded = np.pad(image_array, ((0, N - 1), (0, N - 1), (0, 0)), mode='wrap')
    windows = sliding_window_view(padded, (N, N, 3))[:H, :W, 0]
    flat = np.ascontiguousarray(windows).reshape(H * W, N * N * 3)
    unique_flat, counts = np.unique(flat, axis=0, return_counts=True)
    unique_patterns = np.ascontiguousarray(unique_flat.reshape(-1, N, N, 3))
    weights = counts.astype(np.float64) / counts.sum()
    rules = _compute_rules(unique_patterns)
    return unique_patterns, weights, rules

# ------------------------------------------------------------------------
# 2. WAVE FUNCTION COLLAPSE (chunked + Numba JIT-Compiled)
# ------------------------------------------------------------------------
# The solver is split so the Python caller can poll Firestore between chunks
# for progress reporting and in-flight cancellation. Within a chunk the code
# is identical to the one-shot version: per-cell compatibility-count
# propagation + Shannon entropy H = log(Σw) - Σ(w·log w)/Σw.

STATUS_CONTINUE = 0
STATUS_DONE = 1
STATUS_CONTRADICTION = 2


def _init_wfc_state(grid_size, num_patterns, rules, weights):
    """Allocate the persistent solver state. Runs once per attempt (in Python
    since it's a one-off setup and not performance-critical)."""
    wave = np.ones((grid_size, grid_size, num_patterns), dtype=np.bool_)

    # init_support[t, d] = count of j with rules[j, opp(d), t] True.
    opp = [2, 3, 0, 1]
    init_support = np.zeros((num_patterns, 4), dtype=np.int32)
    for d in range(4):
        init_support[:, d] = rules[:, opp[d], :].sum(axis=0).astype(np.int32)
    support = np.broadcast_to(
        init_support, (grid_size, grid_size, num_patterns, 4)
    ).copy()

    w_log_w = np.zeros(num_patterns, dtype=np.float64)
    nz = weights > 0.0
    w_log_w[nz] = weights[nz] * np.log(weights[nz])

    # Queue is reset to empty after every propagation (see step_wfc), so the
    # peak size is bounded by a single cascade, not cumulative eliminations.
    max_q = grid_size * grid_size * num_patterns
    q_y = np.empty(max_q, dtype=np.int32)
    q_x = np.empty(max_q, dtype=np.int32)
    q_t = np.empty(max_q, dtype=np.int32)
    qht = np.zeros(2, dtype=np.int64)  # [head, tail]
    return wave, support, w_log_w, q_y, q_x, q_t, qht


@njit(cache=True)
def step_wfc(wave, support, w_log_w, weights, rules, grid_size, num_patterns,
             q_y, q_x, q_t, qht, max_collapses):
    """Run up to max_collapses observe+propagate iterations. Mutates wave /
    support / q_* / qht in place so state persists across calls.

    Returns (status, collapses_done):
      STATUS_CONTINUE      — budget exhausted, call again to keep going
      STATUS_DONE          — every cell is collapsed
      STATUS_CONTRADICTION — a cell has zero valid patterns; caller should retry
    """
    dy = np.array([-1, 0, 1, 0], dtype=np.int32)
    dx = np.array([0, 1, 0, -1], dtype=np.int32)
    opp = np.array([2, 3, 0, 1], dtype=np.int32)

    q_head = qht[0]
    q_tail = qht[1]
    collapses = 0

    while collapses < max_collapses:
        # --- Find minimum Shannon entropy cell ---
        min_entropy = 1e18
        min_y = -1
        min_x = -1
        for y in range(grid_size):
            for x in range(grid_size):
                valid_states = 0
                sum_w = 0.0
                sum_wlw = 0.0
                for t in range(num_patterns):
                    if wave[y, x, t]:
                        valid_states += 1
                        sum_w += weights[t]
                        sum_wlw += w_log_w[t]
                if valid_states == 0:
                    qht[0] = q_head
                    qht[1] = q_tail
                    return STATUS_CONTRADICTION, collapses
                elif valid_states > 1:
                    entropy = np.log(sum_w) - sum_wlw / sum_w
                    entropy -= np.random.rand() * 1e-6
                    if entropy < min_entropy:
                        min_entropy = entropy
                        min_y = y
                        min_x = x

        if min_y == -1:
            qht[0] = q_head
            qht[1] = q_tail
            return STATUS_DONE, collapses

        # --- Observe: weighted random pick ---
        total = 0.0
        for t in range(num_patterns):
            if wave[min_y, min_x, t]:
                total += weights[t]
        r = np.random.rand() * total
        acc = 0.0
        chosen = -1
        for t in range(num_patterns):
            if wave[min_y, min_x, t]:
                acc += weights[t]
                if acc >= r:
                    chosen = t
                    break
        if chosen == -1:
            for t in range(num_patterns):
                if wave[min_y, min_x, t]:
                    chosen = t

        # Collapse
        for t in range(num_patterns):
            if wave[min_y, min_x, t] and t != chosen:
                wave[min_y, min_x, t] = False
                q_y[q_tail] = min_y
                q_x[q_tail] = min_x
                q_t[q_tail] = t
                q_tail += 1

        # Propagate
        while q_head < q_tail:
            jy = q_y[q_head]
            jx = q_x[q_head]
            j = q_t[q_head]
            q_head += 1
            for d in range(4):
                ny = jy + dy[d]
                nx = jx + dx[d]
                if 0 <= ny < grid_size and 0 <= nx < grid_size:
                    od = opp[d]
                    for t in range(num_patterns):
                        if wave[ny, nx, t] and rules[j, d, t]:
                            support[ny, nx, t, od] -= 1
                            if support[ny, nx, t, od] == 0:
                                wave[ny, nx, t] = False
                                q_y[q_tail] = ny
                                q_x[q_tail] = nx
                                q_t[q_tail] = t
                                q_tail += 1

        # Propagation for this collapse is complete; reset the queue so its
        # peak usage is one cascade, not cumulative. Saves a lot of memory on
        # large grids.
        q_head = 0
        q_tail = 0
        collapses += 1

    qht[0] = q_head
    qht[1] = q_tail
    return STATUS_CONTINUE, collapses


def _extract_result(wave):
    """For each cell, pick the first remaining pattern index. After a successful
    solve each cell has exactly one True in the pattern axis; argmax gives it."""
    return np.argmax(wave, axis=-1).astype(np.int32)


CANCEL_SENTINEL = "CANCELLED"  # run_wfc_pipeline returns this string when the
                               # user cancelled the job mid-solve.

PROGRESS_INTERVAL_SECONDS = 1.0
CHUNK_COLLAPSES = 64


def run_wfc_pipeline(seed_array, patch_size, grid_size, job_id, db_name):
    """Runs extraction + chunked solve in a subprocess.

    Keeping all Numba work inside the child process matters for two reasons:
    1. Numba's default workqueue threading layer is not threadsafe across OS
       threads, so calling JIT functions from multiple gunicorn threads crashes
       the worker. Isolating to one child process sidesteps this.
    2. It gives us a clean kill target for the 5-minute circuit breaker.

    The solver is driven in chunks so we can poll Firestore between them for:
      - Progress reporting: we write `progress` (0.0..1.0) on the job doc.
      - In-flight cancel: if the user flipped status to CANCELLED while we
        were running, we bail out.

    Returns:
      (patterns, result_grid) on success.
      CANCEL_SENTINEL string on user cancel.
    """
    # The subprocess gets its own Firestore client. Parent-initialized gRPC
    # channels don't survive spawn/fork safely.
    fs_client = firestore.Client(database=db_name)
    doc_ref = fs_client.collection("wfc_jobs").document(job_id)

    t0 = time.time()
    log(job_id, f"Extracting patterns (N={patch_size}) from {seed_array.shape[1]}x{seed_array.shape[0]} seed")
    patterns, weights, rules = extract_patterns_and_rules(seed_array, N=patch_size)
    num_patterns = len(patterns)
    t_extract = time.time() - t0
    rule_density = float(rules.sum()) / (num_patterns * num_patterns * 4) if num_patterns else 0.0
    log(job_id, f"Found {num_patterns} unique patterns in {t_extract:.2f}s (rule density {rule_density:.1%})")

    total_cells = grid_size * grid_size
    attempts = 0
    t_solve_total = 0.0

    while True:
        attempts += 1
        t1 = time.time()
        log(job_id, f"Solving {grid_size}x{grid_size} grid (attempt {attempts})")

        wave, support, w_log_w, q_y, q_x, q_t, qht = _init_wfc_state(
            grid_size, num_patterns, rules, weights)
        explicit_collapses = 0
        last_report = time.time()
        last_reported_progress = -1.0

        while True:
            status, delta = step_wfc(
                wave, support, w_log_w, weights, rules,
                grid_size, num_patterns,
                q_y, q_x, q_t, qht, CHUNK_COLLAPSES)
            explicit_collapses += delta

            now = time.time()
            if status != STATUS_CONTINUE or (now - last_report) >= PROGRESS_INTERVAL_SECONDS:
                # "Collapsed" = cells with exactly one remaining valid pattern.
                # Propagation can collapse cells as a side effect of an explicit
                # observe step, so this is a truer measure of progress than
                # counting observe steps alone.
                if status == STATUS_DONE:
                    progress = 1.0
                else:
                    collapsed_cells = int((wave.sum(axis=-1) == 1).sum())
                    progress = collapsed_cells / total_cells if total_cells else 1.0

                # One Firestore round-trip: read (for cancel) + write (progress).
                try:
                    snap = doc_ref.get()
                    if snap.exists and snap.get('status') == CANCEL_SENTINEL:
                        log(job_id, f"Cancelled mid-solve at {progress:.1%} on attempt {attempts}")
                        return CANCEL_SENTINEL
                    if abs(progress - last_reported_progress) >= 0.005 or status != STATUS_CONTINUE:
                        doc_ref.update({'progress': progress})
                        last_reported_progress = progress
                except Exception as e:
                    # Don't let a transient Firestore blip kill the solve.
                    log(job_id, f"Progress/cancel check failed ({type(e).__name__}: {e})")
                last_report = now

            if status == STATUS_DONE:
                dt = time.time() - t1
                t_solve_total += dt
                log(job_id, f"Solved in {dt:.2f}s (total solve time {t_solve_total:.2f}s across {attempts} attempt(s), {explicit_collapses} explicit collapses)")
                return patterns, _extract_result(wave)
            if status == STATUS_CONTRADICTION:
                dt = time.time() - t1
                t_solve_total += dt
                log(job_id, f"Contradiction after {dt:.2f}s and {explicit_collapses} explicit collapses, retrying")
                break

# ------------------------------------------------------------------------
# 3. THE FLASK WORKER / PIPELINE GLUE
# ------------------------------------------------------------------------
@app.route('/', methods=['POST'])
def pubsub_push():
    envelope = request.get_json()
    if not envelope or 'message' not in envelope:
        return 'Bad Request', 400

    msg_data = base64.b64decode(envelope['message']['data']).decode('utf-8')
    work_order = json.loads(msg_data)

    input_bucket = work_order.get('input_bucket')
    input_file = work_order.get('input_filename')
    output_bucket = work_order.get('output_bucket')
    job_id = work_order.get('job_id')
    patch_size = int(work_order.get('patch_size', 3))
    grid_size = int(work_order.get('output_size', 128))

    if not all([input_bucket, input_file, output_bucket, job_id]):
        return 'Missing required info in payload', 400

    job_start = time.time()
    log(job_id, f"Received: output={grid_size}x{grid_size}, patch={patch_size}, seed=gs://{input_bucket}/{input_file}")

    # Pre-flight: check whether the user cancelled while the message was queued.
    # Returning 200 acks the message so Pub/Sub doesn't redeliver.
    try:
        doc_ref = firestore_client.collection("wfc_jobs").document(job_id)
        snapshot = doc_ref.get()
        if snapshot.exists and snapshot.get('status') == 'CANCELLED':
            log(job_id, "Cancelled before pickup; acking and skipping")
            return 'Cancelled', 200
    except Exception as e:
        log(job_id, f"Pre-flight Firestore read failed ({type(e).__name__}: {e}); proceeding anyway")

    try:
        t0 = time.time()
        in_blob = storage_client.bucket(input_bucket).blob(input_file)
        img_bytes = in_blob.download_as_bytes()
        seed_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        seed_array = np.array(seed_img)
        log(job_id, f"Downloaded seed {seed_img.size[0]}x{seed_img.size[1]} ({len(img_bytes)} bytes) in {time.time()-t0:.2f}s")

        # Run extraction + solve inside a subprocess. Spawn gives clean gRPC
        # isolation since the subprocess opens its own Firestore client for
        # progress/cancel updates. The 5-minute timeout kills the child cleanly
        # if the solver gets stuck.
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=_MP_CTX) as executor:
            future = executor.submit(run_wfc_pipeline, seed_array, patch_size, grid_size, job_id, DB_NAME)
            try:
                result = future.result(timeout=300)
            except concurrent.futures.TimeoutError:
                log(job_id, f"Aborting: exceeded 5-minute solver limit (total elapsed {time.time()-job_start:.1f}s)")
                firestore_client.collection("wfc_jobs").document(job_id).update({"status": "TIMED OUT"})
                return 'Timeout', 200  # 200 so Pub/Sub drops the message

        if result == CANCEL_SENTINEL:
            # Subprocess saw the CANCELLED status mid-solve and bailed. The
            # Firestore doc is already CANCELLED; don't overwrite it.
            log(job_id, f"Cancel honored mid-solve (total elapsed {time.time()-job_start:.1f}s)")
            return 'Cancelled', 200

        patterns, result_grid = result
        t0 = time.time()
        top_left_colors = patterns[:, 0, 0, :]
        final_array = top_left_colors[result_grid]
        final_img = Image.fromarray(final_array, 'RGB')
        out_io = io.BytesIO()
        final_img.save(out_io, format='PNG')
        png_bytes = out_io.getvalue()

        out_name = f"generated-{job_id}.png"
        out_blob = storage_client.bucket(output_bucket).blob(out_name)
        out_blob.upload_from_string(png_bytes, content_type='image/png')
        log(job_id, f"Rendered and uploaded {len(png_bytes)} bytes to gs://{output_bucket}/{out_name} in {time.time()-t0:.2f}s")

        public_url = f"https://storage.googleapis.com/{output_bucket}/{out_name}"
        firestore_client.collection("wfc_jobs").document(job_id).update({
            "status": "COMPLETE",
            "output_url": public_url
        })

        log(job_id, f"Complete. Total wall time {time.time()-job_start:.2f}s")
        return 'Success', 200

    except Exception as e:
        log(job_id, f"Pipeline error ({type(e).__name__}): {e}")
        firestore_client.collection("wfc_jobs").document(job_id).update({"status": "ERROR"})
        return 'Internal Server Error', 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)