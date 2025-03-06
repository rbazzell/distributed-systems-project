import socket
import json
import time
import os
import sys
import logging
import numpy as np
import hashlib
from util import pad_matrix, split_tasks, assemble_result

logging.basicConfig(level=logging.INFO, format='%(message)s')

class Coordinator:
    def __init__(self, node_id, delay):
        self.node_id = node_id
        self.delay = delay
        self.host = '0.0.0.0'
        self.port = 5000 + node_id
        self.worker_ports = [5001, 5002, 5003, 5004, 5005, 5006, 5007]  # Updated to 7 workers
        self.file_processed = False
        self.worker_index = 0
        self.task_history = {}  # {task_hash: result} for top-level tasks

    def run(self):
        logging.info("Starting coordinator. Use input.txt or CMD interface (type 'exit' to quit).")
        while True:
            matrix_a, matrix_b, n, operation = self.get_input()
            if operation == 'exit' or (matrix_a is None and self.file_processed and not sys.stdin.isatty()):
                logging.info("Exiting coordinator.")
                break
            if matrix_a is not None:
                self.process_operation(n, operation, matrix_a, matrix_b)

    def get_input(self):
        # [Unchanged from previous version, omitted for brevity]
        input_file = 'input.txt'
        if os.path.exists(input_file) and not self.file_processed:
            try:
                with open(input_file, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                if not lines:
                    logging.error("input.txt is empty.")
                    self.file_processed = True
                    return None, None, None, None
                parts = lines[0].split()
                if len(parts) != 2:
                    logging.error("First line of input.txt must be 'size operation'.")
                    self.file_processed = True
                    return None, None, None, None
                n = int(parts[0])
                operation = parts[1].lower()
                valid_operations = ['add', 'subtract', 'multiply', 'complement', 'transpose', 'strassen', 'inverse']
                if operation not in valid_operations:
                    logging.error(f"Invalid operation in input.txt. Choose from {valid_operations}")
                    self.file_processed = True
                    return None, None, None, None

                if operation in ['add', 'subtract', 'multiply', 'strassen']:
                    if len(lines) < 2 * n + 1:
                        logging.error(f"input.txt must contain {n} rows for Matrix A and {n} rows for Matrix B.")
                        self.file_processed = True
                        return None, None, None, None
                    matrix_a = np.array([list(map(int, lines[i].split())) for i in range(1, n + 1)])
                    matrix_b = np.array([list(map(int, lines[i].split())) for i in range(n + 1, 2 * n + 1)])
                    if matrix_a.shape != (n, n) or matrix_b.shape != (n, n):
                        logging.error("Matrix dimensions in input.txt must match specified size.")
                        self.file_processed = True
                        return None, None, None, None
                else:
                    if len(lines) < n + 1:
                        logging.error(f"input.txt must contain {n} rows for Matrix A.")
                        self.file_processed = True
                        return None, None, None, None
                    matrix_a = np.array([list(map(int, lines[i].split())) for i in range(1, n + 1)])
                    matrix_b = None
                    if matrix_a.shape != (n, n):
                        logging.error("Matrix dimensions in input.txt must match specified size.")
                        self.file_processed = True
                        return None, None, None, None
                self.file_processed = True
                return matrix_a, matrix_b, n, operation
            except (ValueError, IndexError) as e:
                logging.error(f"Error parsing input.txt: {e}")
                self.file_processed = True
                return None, None, None, None
        elif sys.stdin.isatty():
            cmd = input("Enter command (e.g., '4 strassen', 'exit'): ").strip().lower()
            if cmd == 'exit':
                return None, None, None, 'exit'
            parts = cmd.split()
            if len(parts) != 2:
                logging.error("Invalid command. Use format: 'size operation'")
                return None, None, None, None
            try:
                n = int(parts[0])
                operation = parts[1]
            except ValueError:
                logging.error("Invalid matrix size. Please enter a number.")
                return None, None, None, None

            valid_operations = ['add', 'subtract', 'multiply', 'complement', 'transpose', 'strassen', 'inverse']
            if operation not in valid_operations:
                logging.error(f"Invalid operation. Choose from {valid_operations}")
                return None, None, None, None

            if operation in ['add', 'subtract', 'multiply', 'strassen']:
                matrix_a = self.get_matrix_input(n, "Matrix A")
                matrix_b = self.get_matrix_input(n, "Matrix B")
            else:
                matrix_a = self.get_matrix_input(n, "Matrix")
                matrix_b = None
            return matrix_a, matrix_b, n, operation
        else:
            logging.error("No input.txt or interactive terminal available.")
            return None, None, None, None

    def get_matrix_input(self, n, name):
        # [Unchanged from previous version, omitted for brevity]
        logging.info(f"Enter {name} ({n}x{n}):")
        matrix = []
        for i in range(n):
            while True:
                try:
                    row = input(f"Row {i+1} (space-separated integers, e.g., '1 2 3'): ").strip()
                    row_vals = [int(x) for x in row.split()]
                    if len(row_vals) != n:
                        logging.error(f"Expected {n} values, got {len(row_vals)}")
                        continue
                    matrix.append(row_vals)
                    break
                except ValueError:
                    logging.error("Invalid input. Use space-separated integers.")
        return np.array(matrix)

    def process_operation(self, n, operation, matrix_a, matrix_b):
        logging.info(f"{operation.capitalize()} Input:")
        logging.info(f"Matrix A:\n{matrix_a}")
        if matrix_b is not None:
            logging.info(f"Matrix B:\n{matrix_b}")

        if operation in ['multiply', 'strassen']:
            padded_n = 1 << (n - 1).bit_length()
            if padded_n != n:
                matrix_a = pad_matrix(matrix_a, n, padded_n)
                matrix_b = pad_matrix(matrix_b, n, padded_n)
                logging.info(f"Padded to {padded_n}x{padded_n} for Strassen's algorithm")
            result_matrix = self.strassen_multiply(matrix_a, matrix_b, padded_n)
            result_matrix = result_matrix[:n, :n]
        elif operation in ['add', 'subtract', 'complement', 'transpose']:
            tasks = split_tasks(matrix_a, matrix_b, n, operation)
            intermediate_results = [self.assign_task(self.next_worker(), task) for task in tasks]
            result_matrix = assemble_result(intermediate_results, n)
        elif operation == 'inverse':
            try:
                result_matrix = np.linalg.inv(matrix_a)
            except np.linalg.LinAlgError:
                logging.error("Matrix is singular and cannot be inverted.")
                return

        logging.info(f"Result of {operation}:\n{result_matrix}")

    def strassen_multiply(self, a, b, n):
        """Distribute top-level Strassen's tasks to workers with history."""
        if n <= 2:
            if n == 1:
                return np.array([[a[0, 0] * b[0, 0]]])
            task_hash = self.hash_task(a, b)
            if task_hash in self.task_history:
                return np.array(self.task_history[task_hash])
            result = self.assign_task(self.next_worker(), {'operation': 'multiply', 'matrix_a': a.tolist(), 'matrix_b': b.tolist(), 'size': n})
            self.task_history[task_hash] = result
            return np.array(result)

        half = n // 2
        a11 = a[:half, :half]
        a12 = a[:half, half:]
        a21 = a[half:, :half]
        a22 = a[half:, half:]
        b11 = b[:half, :half]
        b12 = b[:half, half:]
        b21 = b[half:, :half]
        b22 = b[half:, half:]

        # Local additions/subtractions
        m1_a = a11 + a22
        m1_b = b11 + b22
        m2_a = a21 + a22
        m3_b = b12 - b22
        m4_b = b21 - b11
        m5_a = a12 - a22
        m6_a = a11 - a21
        m7_a = a11 + a12
        m7_b = b21 - b22

        # Distribute recursive Strassen's tasks
        tasks = [
            {'operation': 'strassen', 'matrix_a': m1_a.tolist(), 'matrix_b': m1_b.tolist(), 'size': half},
            {'operation': 'strassen', 'matrix_a': m2_a.tolist(), 'matrix_b': b11.tolist(), 'size': half},
            {'operation': 'strassen', 'matrix_a': a11.tolist(), 'matrix_b': m3_b.tolist(), 'size': half},
            {'operation': 'strassen', 'matrix_a': a22.tolist(), 'matrix_b': m4_b.tolist(), 'size': half},
            {'operation': 'strassen', 'matrix_a': m5_a.tolist(), 'matrix_b': b11.tolist(), 'size': half},
            {'operation': 'strassen', 'matrix_a': m6_a.tolist(), 'matrix_b': b22.tolist(), 'size': half},
            {'operation': 'strassen', 'matrix_a': m7_a.tolist(), 'matrix_b': m7_b.tolist(), 'size': half},
        ]
        task_hashes = [self.hash_task(np.array(task['matrix_a']), np.array(task['matrix_b'])) for task in tasks]
        results = []
        for i, (task, task_hash) in enumerate(zip(tasks, task_hashes)):
            if task_hash in self.task_history:
                results.append(self.task_history[task_hash])
            else:
                result = self.assign_task(self.next_worker(), task)
                self.task_history[task_hash] = result
                results.append(result)
        results = [np.array(r) for r in results]

        # Local quadrant assembly
        c11 = results[0] + results[3] - results[4] + results[6]
        c12 = results[2] + results[4]
        c21 = results[1] + results[3]
        c22 = results[0] + results[2] - results[1] + results[5]

        result = np.zeros((n, n), dtype=float)
        result[:half, :half] = c11
        result[:half, half:] = c12
        result[half:, :half] = c21
        result[half:, half:] = c22
        return result

    def assign_task(self, worker_id, task):
        """Send task to a worker and get result."""
        worker_port = 5000 + worker_id
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('worker' + str(worker_id), worker_port))
            message = json.dumps({'task': task})
            s.sendall(message.encode('utf-8'))
            data = s.recv(4096)
            response = json.loads(data.decode('utf-8'))
            return response['result']

    def next_worker(self):
        """Round-robin worker selection for 7 workers."""
        worker_id = self.worker_index + 1  # 1 to 7
        self.worker_index = (self.worker_index + 1) % 7  # Updated to 7
        return worker_id

    def hash_task(self, a, b):
        """Generate a unique hash for a multiplication task."""
        a_str = str(a.tolist())
        b_str = str(b.tolist())
        return hashlib.md5((a_str + b_str).encode('utf-8')).hexdigest()

if __name__ == '__main__':
    node_id = int(os.getenv('NODE_ID', '0'))
    delay = float(os.getenv('DELAY', '1.0'))
    coordinator = Coordinator(node_id, delay)
    coordinator.run()