import socket
import pickle
import numpy as np
import threading
import time
import logging
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('worker')

class Worker:
    def __init__(self, coordinator_host='coordinator', coordinator_port=5000, base_case_size=1):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.base_case_size = base_case_size
        self.worker_id = None
        
        logger.info(f"Worker initializing, will connect to coordinator at {coordinator_host}:{coordinator_port}")
    
    def connect_to_coordinator(self):
        """Connect to the coordinator and register as a worker"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.coordinator_host, self.coordinator_port))
            
            # Register with the coordinator
            registration_msg = {
                "type": "worker_register"
            }
            self.socket.sendall(pickle.dumps(registration_msg))
            
            # Get response
            response = pickle.loads(self.socket.recv(4096))
            self.worker_id = response.get("worker_id")
            
            logger.info(f"Successfully registered with coordinator, assigned ID: {self.worker_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            time.sleep(2)  # Wait before retrying
            return False
    
    def start(self):
        """Start the worker"""
        while not self.connect_to_coordinator():
            logger.info("Retrying connection to coordinator...")
        
        try:
            while True:
                # Wait for task from coordinator
                try:
                    data = self.socket.recv(4096)
                    if not data:
                        logger.warning("Connection to coordinator lost, reconnecting...")
                        self.connect_to_coordinator()
                        continue
                    
                    task = pickle.loads(data)
                    threading.Thread(target=self.process_task, args=(task,)).start()
                
                except Exception as e:
                    logger.error(f"Error receiving task: {e}")
                    self.connect_to_coordinator()
        
        except KeyboardInterrupt:
            logger.info("Worker shutting down")
            self.socket.close()
    
    def process_task(self, task):
        """Process a task received from the coordinator"""
        task_type = task.get("type")
        task_id = task.get("task_id")
        
        logger.info(f"Processing task {task_id} of type {task_type}")
        
        try:
            if task_type == "matrix_multiply":
                self.handle_matrix_multiplication(task)
            elif task_type == "combine_results":
                self.handle_combine_results(task)
            else:
                logger.warning(f"Unknown task type: {task_type}")
        
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            # Notify coordinator of failure
            self.send_result({
                "type": "task_result",
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            })
    
    def handle_matrix_multiplication(self, task):
        """Handle a matrix multiplication task"""
        A = np.array(task["A"])
        B = np.array(task["B"])
        task_id = task["task_id"]
        parent_task_id = task["parent_task_id"]
        subtask_id = task["subtask_id"]
        
        # Get matrix dimensions
        n = A.shape[0]
        
        # Base case: small matrix multiplication
        if n <= self.base_case_size:
            result = A @ B
            
            self.send_result({
                "type": "task_result",
                "result_type": "matrix",
                "task_id": task_id,
                "parent_task_id": parent_task_id,
                "subtask_id": subtask_id,
                "result": result
            })
            
            logger.debug(f"Base case multiplication completed for task {task_id}")
        
        # Apply Strassen's algorithm
        else:
            # Check if n is a power of 2, if not, pad the matrices
            if n & (n-1) != 0:  # Not a power of 2
                # Find the next power of 2
                new_n = 1
                while new_n < n:
                    new_n *= 2
                
                # Pad matrices with zeros
                A_padded = np.zeros((new_n, new_n))
                B_padded = np.zeros((new_n, new_n))
                
                A_padded[:n, :n] = A
                B_padded[:n, :n] = B
                
                A = A_padded
                B = B_padded
                n = new_n
            
            # Split matrices into quadrants
            mid = n // 2
            A11 = A[:mid, :mid]
            A12 = A[:mid, mid:]
            A21 = A[mid:, :mid]
            A22 = A[mid:, mid:]
            
            B11 = B[:mid, :mid]
            B12 = B[:mid, mid:]
            B21 = B[mid:, :mid]
            B22 = B[mid:, mid:]
            
            # Create 7 subtasks for Strassen's algorithm
            subtasks = []
            
            # M1 = (A11 + A22) * (B11 + B22)
            subtasks.append({
                "type": "matrix_multiply",
                "task_id": str(uuid.uuid4()),
                "A": A11 + A22,
                "B": B11 + B22,
                "parent_task_id": task_id,
                "subtask_id": "M1"
            })
            
            # M2 = (A21 + A22) * B11
            subtasks.append({
                "type": "matrix_multiply",
                "task_id": str(uuid.uuid4()),
                "A": A21 + A22,
                "B": B11,
                "parent_task_id": task_id,
                "subtask_id": "M2"
            })
            
            # M3 = A11 * (B12 - B22)
            subtasks.append({
                "type": "matrix_multiply",
                "task_id": str(uuid.uuid4()),
                "A": A11,
                "B": B12 - B22,
                "parent_task_id": task_id,
                "subtask_id": "M3"
            })
            
            # M4 = A22 * (B21 - B11)
            subtasks.append({
                "type": "matrix_multiply",
                "task_id": str(uuid.uuid4()),
                "A": A22,
                "B": B21 - B11,
                "parent_task_id": task_id,
                "subtask_id": "M4"
            })
            
            # M5 = (A11 + A12) * B22
            subtasks.append({
                "type": "matrix_multiply",
                "task_id": str(uuid.uuid4()),
                "A": A11 + A12,
                "B": B22,
                "parent_task_id": task_id,
                "subtask_id": "M5"
            })
            
            # M6 = (A21 - A11) * (B11 + B12)
            subtasks.append({
                "type": "matrix_multiply",
                "task_id": str(uuid.uuid4()),
                "A": A21 - A11,
                "B": B11 + B12,
                "parent_task_id": task_id,
                "subtask_id": "M6"
            })
            
            # M7 = (A12 - A22) * (B21 + B22)
            subtasks.append({
                "type": "matrix_multiply",
                "task_id": str(uuid.uuid4()),
                "A": A12 - A22,
                "B": B21 + B22,
                "parent_task_id": task_id,
                "subtask_id": "M7"
            })
            
            # Send subtasks to coordinator
            self.send_result({
                "type": "task_result",
                "result_type": "subtasks",
                "task_id": task_id,
                "subtasks": subtasks
            })
            
            logger.info(f"Task {task_id} decomposed into 7 Strassen subtasks")
    
    def handle_combine_results(self, task):
        """Handle a task to combine Strassen results"""
        parent_task_id = task["parent_task_id"]
        subtask_results = task["subtask_results"]
        
        # We need all 7 Strassen multiplication results
        M1 = subtask_results.get("M1")
        M2 = subtask_results.get("M2")
        M3 = subtask_results.get("M3")
        M4 = subtask_results.get("M4")
        M5 = subtask_results.get("M5")
        M6 = subtask_results.get("M6")
        M7 = subtask_results.get("M7")
        
        if all(M is not None for M in [M1, M2, M3, M4, M5, M6, M7]):
            # Calculate the four quadrants of the result matrix
            C11 = M1 + M4 - M5 + M7
            C12 = M3 + M5
            C21 = M2 + M4
            C22 = M1 - M2 + M3 + M6
            
            # Combine the quadrants into the final result
            n = C11.shape[0]
            C = np.zeros((2*n, 2*n))
            C[:n, :n] = C11
            C[:n, n:] = C12
            C[n:, :n] = C21
            C[n:, n:] = C22
            
            # Send the combined result back to the coordinator
            self.send_result({
                "type": "task_result",
                "result_type": "combine",
                "task_id": task["task_id"],
                "parent_task_id": parent_task_id,
                "result": C
            })
            
            logger.info(f"Combined results for parent task {parent_task_id}")
        else:
            logger.error(f"Missing some Strassen products for task {task['task_id']}")
    
    def send_result(self, result):
        """Send a result back to the coordinator"""
        try:
            # Use a new socket to send results to avoid conflicts
            result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result_socket.connect((self.coordinator_host, self.coordinator_port))
            result_socket.sendall(pickle.dumps(result))
            
            # Wait for acknowledgment
            response = pickle.loads(result_socket.recv(4096))
            if response.get("status") == "result_received":
                logger.debug(f"Result for task {result.get('task_id')} acknowledged by coordinator")
            
            result_socket.close()
        except Exception as e:
            logger.error(f"Error sending result to coordinator: {e}")

def main():
    worker = Worker()
    worker.start()

if __name__ == "__main__":
    main()