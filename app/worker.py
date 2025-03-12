import os
import time
import numpy as np
from flask import Flask, request, jsonify
import threading
import logging

from utils import Task, TaskType, split_matrix, join_matrices, create_retry_session


app = Flask(__name__)
app.logger.setLevel(logging.INFO)
session = create_retry_session()

# Environment variables
NODE_ID = os.environ.get('NODE_ID', '0')
PORT = int(os.environ.get('PORT', 5001))
COORDINATOR_HOST = os.environ.get('COORDINATOR_HOST', 'coordinator')
COORDINATOR_PORT = int(os.environ.get('COORDINATOR_PORT', 5000))
COORDINATOR_URL = f"http://{COORDINATOR_HOST}:{COORDINATOR_PORT}"
WORKER_URL = f"http://worker{NODE_ID}:{PORT}" if NODE_ID != 'coordinator' else f"http://localhost:{PORT}"
MIN_MULTIPLY = int(os.environ.get('MIN_MULT', 2))

def register_with_coordinator():
    """Register this worker with the coordinator"""
    try:
        response = session.post(
            f"{COORDINATOR_URL}/register",
            json={
                'worker_id': NODE_ID,
                'worker_url': WORKER_URL
            },
            timeout=10
        )
        if response.status_code == 200:
            print(f"Successfully registered with coordinator as worker {NODE_ID}")
            return True
        else:
            print(f"Failed to register with coordinator: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error registering with coordinator: {e}")
        return False

def send_result_to_coordinator(task_id, result):
    """Send task result back to coordinator"""
    try:
        response = session.post(
            f"{COORDINATOR_URL}/result",
            json={
                'task_id': task_id,
                'result': result.tolist()
            },
            timeout=(5, 120)
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending result to coordinator: {e}")
        return False

def process_multiply_task(task):
    """Process a top-level multiplication task, breaking it down using Strassen's algorithm"""
    matrix_a, matrix_b = task.matrices
    task_id = task.task_id
    
    # Check if we can use direct multiplication (1x1 matrices or base case)
    if matrix_a.shape[0] <= MIN_MULTIPLY or matrix_a.shape[1] <= MIN_MULTIPLY or matrix_b.shape[1] <= MIN_MULTIPLY:
        result = matrix_a @ matrix_b
        return send_result_to_coordinator(task_id, result)
    
    # If not, we perform strassen's algorithm
    # Split matrices into quadrants
    a11, a12, a21, a22 = split_matrix(matrix_a)
    b11, b12, b21, b22 = split_matrix(matrix_b)
    
    # Create the 7 products required by Strassen's algorithm
    products = [
        # M1 = (A11 + A22) * (B11 + B22)
        [np.add(a11, a22), np.add(b11, b22)],
        
        # M2 = (A21 + A22) * B11
        [np.add(a21, a22), b11],
        
        # M3 = A11 * (B12 - B22)
        [a11, np.subtract(b12, b22)],
        
        # M4 = A22 * (B21 - B11)
        [a22, np.subtract(b21, b11)],
        
        # M5 = (A11 + A12) * B22
        [np.add(a11, a12), b22],
        
        # M6 = (A21 - A11) * (B11 + B12)
        [np.subtract(a21, a11), np.add(b11, b12)],
        
        # M7 = (A12 - A22) * (B21 + B22)
        [np.subtract(a12, a22), np.add(b21, b22)]
    ]
    
    for i, product in enumerate(products):
        # Send each subtask back to coordinator for processing
        # In a real implementation, we might want to batch these
        session.post(
            f"{COORDINATOR_URL}/return",
            json={
                'matrix_a': product[0].tolist(),
                'matrix_b': product[1].tolist(),
                'parent_id': task_id,
                'm_number': i
            }
        )
    
    return True

def process_strassen_combine_task(task):
    """Combine the 7 results from Strassen's algorithm subtasks"""
    results = task.subtasks_results
    task_id = task.task_id
    parent_id = task.parent_id
    
    # Get the 7 product matrices
    m1, m2, m3, m4, m5, m6, m7 = results
    
    # Calculate the quadrants of the result matrix
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    # Join the quadrants to form the final result
    result = join_matrices(c11, c12, c21, c22)
    
    # Send the result back to the coordinator
    return send_result_to_coordinator(parent_id if parent_id else task_id, result)

@app.route('/process', methods=['POST'])
def process_task():
    """Endpoint for processing a task"""
    data = request.json
    task = Task.from_dict(data)
    
    print(f"Worker {NODE_ID} processing task {task.task_id} of type {task.task_type.value}")
    
    # Process the task based on its type
    if task.task_type == TaskType.MULTIPLY:
        threading.Thread(target=process_multiply_task, args=(task,)).start()
    elif task.task_type == TaskType.COMBINE:
        threading.Thread(target=process_strassen_combine_task, args=(task,)).start()
    else:
        return jsonify({'error': 'Unknown task type'}), 400
    
    return jsonify({'status': 'processing'}), 200

def register_loop():
    """Keep trying to register with the coordinator"""
    registered = False
    while not registered:
        registered = register_with_coordinator()
        if not registered:
            print(f"Will retry registration in 5 seconds...")
            time.sleep(5)

if __name__ == '__main__':
    print(f"Starting worker node (ID: {NODE_ID}) on port {PORT}")
    
    # Start the registration process in a separate thread
    threading.Thread(target=register_loop).start()
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=PORT, threaded=True, debug=False)