import os
import numpy as np
import threading
import logging
from flask import Flask, request, jsonify

from utils import Task, TaskType, pad_matrices, unpad_matrix, create_retry_session

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
session = create_retry_session()

# Environment variables
NODE_ID = os.environ.get('NODE_ID', 'coordinator')
PORT = int(os.environ.get('PORT', 5000))

# Track active workers and tasks
workers = {}  # Map worker_id to URL
active_tasks = {}  # Map task_id to task details
pending_results = {}  # Map task_id to [received_subtasks_count, results_list]
client_tasks = {}  # Map client_task_id to original dimensions

# Lock for thread safety
lock = threading.Lock()

def register_worker(worker_id, worker_url):
    """Register a worker node"""
    with lock:
        workers[worker_id] = worker_url
        print(f"Worker {worker_id} registered at {worker_url}")
    return True

def get_available_worker():
    """Get an available worker (simple round-robin for now)"""
    with lock:
        if not workers:
            return None
        
        # Get the first worker (could be enhanced with load balancing)
        worker_id = list(workers.keys())[0]
        worker_url = workers[worker_id]
        
        # Move this worker to the end of the dict for round-robin
        workers.pop(worker_id)
        workers[worker_id] = worker_url
        
        return worker_id, worker_url

def send_task_to_worker(worker_url, task):
    """Send a task to a worker"""
    try:
        response = session.post(
            f"{worker_url}/process",
            json=task.to_dict(),
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending task to worker: {e}")
        return False

def process_result(task_id, result):
    """Process a completed task result"""
    with lock:
        if task_id not in active_tasks:
            print(f"Received result for unknown task: {task_id}={result}")
            return
        
        task = active_tasks[task_id]

        parent_id = task.parent_id

        combine_task_generated = False
        
        
        if parent_id:
            # This is a subtask, update the parent task's results
            if parent_id not in pending_results:
                pending_results[parent_id] = [0, [None] * 7]
            
            # Find the position of this subtask in the parent's pending results
            pending_results[parent_id][1][task.m_number] = result
            pending_results[parent_id][0] += 1
            # Check if all subtasks are complete
            if pending_results[parent_id][0] == 7:  # 7 for Strassen
                # Create combine task
                all_results = pending_results[parent_id][1]
                combine_task = Task(
                    task_type=TaskType.COMBINE,
                    subtasks_results=all_results,
                    parent_id=parent_id
                )
                
                # Clean up
                del pending_results[parent_id]

                # Register the task
                active_tasks[combine_task.task_id] = combine_task
                combine_task_generated = True
                
                
        else:
            # This is a top-level task
            if task_id in client_tasks:
                # Unpad the result if needed
                original_shapes = client_tasks[task_id]
                if original_shapes:
                    result = unpad_matrix(result, original_shapes[0], original_shapes[1])
                
                print(f"Final result for task {task_id}:\n{result}")
                print(result.tolist())
                with open("/app/data/results.txt", 'w') as f:
                    f.write(";\n".join(" ".join(str(y) for y in x) for x in result.tolist()))

                # Clean up
                del client_tasks[task_id]
            
        # Remove completed task
        del active_tasks[task_id]

    if combine_task_generated: #logic moved to prevent deadlocking
        # Send to an available worker
        worker_id, worker_url = get_available_worker()
        print(f"Sending combine task {combine_task.task_id} to worker {worker_id}")
        send_task_to_worker(worker_url, combine_task)

@app.route('/register', methods=['POST'])
def register():
    """Endpoint for workers to register"""
    data = request.json
    worker_id = data.get('worker_id')
    worker_url = data.get('worker_url')
    
    if not worker_id or not worker_url:
        return jsonify({'error': 'Missing worker ID or URL'}), 400
    
    register_worker(worker_id, worker_url)
    return jsonify({'status': 'registered'}), 200

@app.route('/submit', methods=['POST'])
def submit_task():
    """Endpoint for clients to submit matrix multiplication tasks"""
    data = request.json
    matrix_a = np.array(data.get('matrix_a', []))
    matrix_b = np.array(data.get('matrix_b', []))
    
    if matrix_a.size == 0 or matrix_b.size == 0:
        return jsonify({'error': 'Invalid matrices'}), 400
    
    if matrix_a.shape[1] != matrix_b.shape[0]:
        return jsonify({'error': 'Incompatible matrix dimensions'}), 400
    
    # Pad matrices for Strassen if needed
    padded_a, padded_b, original_a_shape, original_b_shape = pad_matrices(matrix_a, matrix_b)
    
    # Create the task
    task = Task(
        task_type=TaskType.MULTIPLY,
        matrices=[padded_a, padded_b]
    )
    
    # Register the task
    with lock:
        active_tasks[task.task_id] = task
        client_tasks[task.task_id] = (original_a_shape, original_b_shape)

    
    
    # Get an available worker
    worker_id, worker_url = get_available_worker()
    if worker_id is None:
        return jsonify({'error': 'No workers available'}), 503
    
    print(f"Sending task {task.task_id} to worker {worker_id}")
    success = send_task_to_worker(worker_url, task)
    
    if not success:
        return jsonify({'error': 'Failed to send task to worker'}), 500
    
    return jsonify({
        'task_id': task.task_id,
        'status': 'submitted'
    }), 200

@app.route('/return', methods=['POST'])
def return_task():
    """Endpoint for workers to submit matrix multiplication subtasks"""
    data = request.json
    matrix_a = np.array(data.get('matrix_a', []))
    matrix_b = np.array(data.get('matrix_b', []))
    parent_id = data.get('parent_id')
    m_number = data.get('m_number')
    
    # Create the task
    task = Task(
        task_type=TaskType.MULTIPLY,
        matrices=[matrix_a, matrix_b],
        parent_id=parent_id,
        m_number=m_number
    )
    
    # Register the task
    with lock:
        active_tasks[task.task_id] = task    
    
    # Get an available worker
    worker_id, worker_url = get_available_worker()
    if worker_id is None:
        return jsonify({'error': 'No workers available'}), 503
    
    print(f"Sending task {task.task_id} to worker {worker_id}")
    success = send_task_to_worker(worker_url, task)
    
    if not success:
        return jsonify({'error': 'Failed to send task to worker'}), 500
    
    return jsonify({
        'task_id': task.task_id,
        'status': 'returned'
    }), 200


@app.route('/result', methods=['POST'])
def receive_result():
    """Endpoint for workers to submit results"""
    data = request.json
    task_id = data.get('task_id')
    result = np.array(data.get('result', []))
    
    if not task_id or result.size == 0:
        return jsonify({'error': 'Invalid result data'}), 400
    
    process_result(task_id, result)
    return jsonify({'status': 'received'}), 200

if __name__ == '__main__':
    print(f"Starting coordinator node (ID: {NODE_ID}) on port {PORT}")
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=PORT, threaded=True, debug=False)