import socket
import pickle
import threading
import numpy as np
import time
import logging
from queue import Queue
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('coordinator')

class Coordinator:
    def __init__(self, host='0.0.0.0', port=5000, base_case_size=1):
        self.host = host
        self.port = port
        self.base_case_size = base_case_size
        
        # Dictionary to track tasks
        self.tasks = {}
        self.tasks_lock = threading.Lock()
        
        # Workers management
        self.available_workers = []
        self.workers_lock = threading.Lock()
        
        # Results tracking
        self.pending_results = {}
        self.completed_results = {}

        # Task queue
        self.task_queue = Queue()
        
        # Socket for accepting connections
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        
        logger.info(f"Coordinator initialized on {self.host}:{self.port}")
    
    def start(self):
        """Start the coordinator server and worker threads"""
        self.server_socket.listen(10)
        logger.info("Coordinator started and listening for connections")
        
        # Start thread to process the task queue
        task_processor = threading.Thread(target=self.process_task_queue)
        task_processor.daemon = True
        task_processor.start()
        
        # Main loop to accept connections
        try:
            while True:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"Connection from {addr}")
                
                # Start a new thread to handle the client
                client_handler = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, addr)
                )
                client_handler.daemon = True
                client_handler.start()
        except KeyboardInterrupt:
            logger.info("Coordinator shutting down")
            self.server_socket.close()
    
    def handle_client(self, client_socket, addr):
        """Handle communication with a client"""
        try:
            # Receive initial message
            data = client_socket.recv(4096)
            if not data:
                return
            
            message = pickle.loads(data)
            message_type = message.get("type")
            
            if message_type == "worker_register":
                self.register_worker(client_socket, addr)
            elif message_type == "submit_matrix_multiply":
                self.handle_matrix_multiplication(client_socket, message)
            elif message_type == "task_result":
                self.handle_task_result(message)
                # Send acknowledgment
                client_socket.sendall(pickle.dumps({"status": "result_received"}))
                client_socket.close()
            else:
                logger.warning(f"Unknown message type: {message_type}")
                client_socket.close()
        
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")
            client_socket.close()
    
    def register_worker(self, client_socket, addr):
        """Register a new worker"""
        worker_id = str(uuid.uuid4())
        worker_info = {
            "socket": client_socket,
            "address": addr,
            "id": worker_id
        }
        
        with self.workers_lock:
            self.available_workers.append(worker_info)
        
        logger.info(f"Worker {worker_id} registered from {addr}")
        
        # Send registration confirmation
        client_socket.sendall(pickle.dumps({
            "type": "registration_response",
            "worker_id": worker_id
        }))
        
        # Start a thread to keep connection alive and handle worker messages
        worker_thread = threading.Thread(
            target=self.handle_worker_connection,
            args=(worker_info,)
        )
        worker_thread.daemon = True
        worker_thread.start()
    
    def handle_worker_connection(self, worker_info):
        """Handle ongoing communication with a worker"""
        worker_socket = worker_info["socket"]
        worker_id = worker_info["id"]
        
        try:
            while True:
                # Wait for messages from the worker
                data = worker_socket.recv(4096)
                if not data:
                    logger.info(f"Worker {worker_id} disconnected")
                    self.remove_worker(worker_id)
                    break
                
                message = pickle.loads(data)
                message_type = message.get("type")
                
                if message_type == "worker_ready":
                    with self.workers_lock:
                        if worker_info not in self.available_workers:
                            self.available_workers.append(worker_info)
                    logger.debug(f"Worker {worker_id} is ready for tasks")
                
                elif message_type == "task_result":
                    self.handle_task_result(message)
                    # Send acknowledgment
                    worker_socket.sendall(pickle.dumps({"status": "result_received"}))
                    
                    # Mark worker as available again
                    with self.workers_lock:
                        if worker_info not in self.available_workers:
                            self.available_workers.append(worker_info)
        
        except Exception as e:
            logger.error(f"Error in worker {worker_id} connection: {e}")
            self.remove_worker(worker_id)
    
    def remove_worker(self, worker_id):
        """Remove a worker from the available workers list"""
        with self.workers_lock:
            self.available_workers = [w for w in self.available_workers if w["id"] != worker_id]
        logger.info(f"Worker {worker_id} removed")
    
    def handle_matrix_multiplication(self, client_socket, message):
        """Handle a new matrix multiplication request"""
        A = np.array(message["A"])
        B = np.array(message["B"])
        
        # Generate a task ID for this multiplication
        task_id = str(uuid.uuid4())
        
        # Store the client socket to send the final result
        self.pending_results[task_id] = {
            "client_socket": client_socket,
            "subtasks": {},
            "parent": None
        }
        
        # Create the initial task
        task = {
            "type": "matrix_multiply",
            "task_id": task_id,
            "A": A,
            "B": B,
            "parent_task_id": None,
            "subtask_id": None
        }
        
        # Add task to queue
        self.task_queue.put(task)
        logger.info(f"New matrix multiplication task {task_id} added to queue")
    
    def process_task_queue(self):
        """Process tasks from the queue"""
        while True:
            # Get a task from the queue
            task = self.task_queue.get()
            
            # Try to assign to a worker
            self.assign_task_to_worker(task)
            
            # Wait a bit before checking for the next task
            time.sleep(0.01)
    
    def assign_task_to_worker(self, task):
        """Assign a task to an available worker"""
        # Wait until a worker is available
        worker_info = None
        while worker_info is None:
            with self.workers_lock:
                if self.available_workers:
                    worker_info = self.available_workers.pop(0)
            
            if worker_info is None:
                time.sleep(0.1)  # Wait before checking again
        
        try:
            # Send the task to the worker
            worker_socket = worker_info["socket"]
            worker_socket.sendall(pickle.dumps(task))
            
            logger.info(f"Task {task['task_id']} assigned to worker {worker_info['id']}")
            
            # Register the task
            with self.tasks_lock:
                self.tasks[task["task_id"]] = {
                    "worker_id": worker_info["id"],
                    "status": "assigned",
                    "timestamp": time.time()
                }
        
        except Exception as e:
            logger.error(f"Error assigning task to worker: {e}")
            # Put the task back in the queue
            self.task_queue.put(task)
            # Remove the failed worker
            self.remove_worker(worker_info["id"])
    
    def handle_task_result(self, message):
        """Handle a task result from a worker"""
        task_id = message["task_id"]
        result_type = message.get("result_type", "matrix")
        
        if result_type == "subtasks":
            # This is a Strassen decomposition, add subtasks to queue
            subtasks = message["subtasks"]
            parent_task_id = task_id
            
            # Update task status
            with self.tasks_lock:
                if task_id in self.tasks:
                    self.tasks[task_id]["status"] = "decomposed"
            
            # Add subtasks to the pending results
            if parent_task_id in self.pending_results:
                for subtask in subtasks:
                    subtask_id = subtask["task_id"]
                    self.pending_results[parent_task_id]["subtasks"][subtask_id] = {
                        "completed": False,
                        "result": None
                    }
            
            # Add all subtasks to queue
            for subtask in subtasks:
                self.task_queue.put(subtask)
            
            logger.info(f"Task {task_id} decomposed into {len(subtasks)} subtasks")
        
        elif result_type == "combine":
            # This is a result from combining Strassen subtasks
            final_result = message["result"]
            parent_task_id = message["parent_task_id"]
            
            # Update task status
            with self.tasks_lock:
                if task_id in self.tasks:
                    self.tasks[task_id]["status"] = "completed"
            
            # If this is the root task, send to client
            if parent_task_id is None:
                if task_id in self.pending_results:
                    client_socket = self.pending_results[task_id]["client_socket"]
                    try:
                        response = {
                            "type": "multiplication_result",
                            "result": final_result.tolist()
                        }
                        client_socket.sendall(pickle.dumps(response))
                        client_socket.close()
                        
                        # Clean up
                        del self.pending_results[task_id]
                        logger.info(f"Final result for task {task_id} sent to client")
                    except Exception as e:
                        logger.error(f"Error sending result to client: {e}")
            else:
                # This is a combined result for a parent task
                if parent_task_id in self.pending_results:
                    self.pending_results[parent_task_id]["subtasks"][task_id] = {
                        "completed": True,
                        "result": final_result
                    }
                    
                    # Check if all subtasks for the parent are complete
                    subtasks = self.pending_results[parent_task_id]["subtasks"]
                    all_complete = all(subtask["completed"] for subtask in subtasks.values())
                    
                    if all_complete:
                        # Create a task to combine the results
                        combine_task = {
                            "type": "combine_results",
                            "task_id": str(uuid.uuid4()),
                            "parent_task_id": parent_task_id,
                            "subtask_results": {
                                subtask_id: subtask["result"] 
                                for subtask_id, subtask in subtasks.items()
                            }
                        }
                        self.task_queue.put(combine_task)
                        logger.info(f"All subtasks for {parent_task_id} complete, creating combine task")
        
        elif result_type == "matrix":
            # This is a base case result from a matrix multiplication
            result_matrix = message["result"]
            parent_task_id = message["parent_task_id"]
            subtask_id = message["subtask_id"]
            
            # Update task status
            with self.tasks_lock:
                if task_id in self.tasks:
                    self.tasks[task_id]["status"] = "completed"
            
            # Store the result
            if parent_task_id in self.pending_results:
                if subtask_id in self.pending_results[parent_task_id]["subtasks"]:
                    self.pending_results[parent_task_id]["subtasks"][subtask_id] = {
                        "completed": True,
                        "result": result_matrix
                    }
                    
                    # Check if all subtasks for the parent are complete
                    subtasks = self.pending_results[parent_task_id]["subtasks"]
                    all_complete = all(subtask["completed"] for subtask in subtasks.values())
                    
                    if all_complete:
                        # Create a task to combine the results
                        combine_task = {
                            "type": "combine_results",
                            "task_id": str(uuid.uuid4()),
                            "parent_task_id": parent_task_id,
                            "subtask_results": {
                                subtask_id: subtask["result"] 
                                for subtask_id, subtask in subtasks.items()
                            }
                        }
                        self.task_queue.put(combine_task)
                        logger.info(f"All subtasks for {parent_task_id} complete, creating combine task")
            
            logger.debug(f"Base case result received for task {task_id}")
        
        else:
            logger.warning(f"Unknown result type: {result_type}")

def main():
    coordinator = Coordinator()
    coordinator.start()

if __name__ == "__main__":
    main()