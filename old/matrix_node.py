import socket
import json
import time
import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')

class Node:
    def __init__(self, node_id, delay, is_coordinator=False):
        self.node_id = node_id
        self.delay = delay
        self.is_coordinator = is_coordinator
        self.host = '0.0.0.0'
        self.port = 5000 + node_id
        self.worker_ports = [5001, 5002, 5003, 5004, 5005]

    def run(self):
        if self.is_coordinator:
            self.run_coordinator()
        else:
            self.run_worker()

    def run_coordinator(self):
        logging.info("Starting coordinator with CMD interface. Type 'exit' to quit.")
        while True:
            if sys.stdin.isatty():
                cmd = input("Enter command (e.g., '4 strassen', 'exit'): ").strip().lower()
                if cmd == 'exit':
                    logging.info("Exiting coordinator.")
                    break
                parts = cmd.split()
                if len(parts) != 2:
                    logging.error("Invalid command. Use format: 'size operation'")
                    continue
                try:
                    n = int(parts[0])
                    operation = parts[1]
                except ValueError:
                    logging.error("Invalid matrix size. Please enter a number.")
                    continue

                valid_operations = ['add', 'subtract', 'multiply', 'complement', 'transpose', 'strassen']
                if operation not in valid_operations:
                    logging.error(f"Invalid operation. Choose from {valid_operations}")
                    continue

                # Get custom matrices from user as NumPy arrays
                if operation in ['add', 'subtract', 'multiply', 'strassen']:
                    matrix_a = self.get_matrix_input(n, "Matrix A")
                    matrix_b = self.get_matrix_input(n, "Matrix B")
                else:
                    matrix_a = self.get_matrix_input(n, "Matrix")
                    matrix_b = None
            else:
                logging.error("Non-interactive mode not supported for custom inputs.")
                sys.exit(1)

            self.process_operation(n, operation, matrix_a, matrix_b)

    def get_matrix_input(self, n, name):
        """Prompt user to input a matrix and return as NumPy array."""
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
        """Process the specified operation using NumPy arrays."""
        logging.info(f"{operation.capitalize()} Input:")
        logging.info(f"Matrix A:\n{matrix_a}")
        if matrix_b is not None:
            logging.info(f"Matrix B:\n{matrix_b}")

        if operation in ['multiply', 'strassen']:
            # Pad matrices to next power of 2 for Strassen's
            padded_n = 1 << (n - 1).bit_length()
            if padded_n != n:
                matrix_a = self.pad_matrix(matrix_a, n, padded_n)
                matrix_b = self.pad_matrix(matrix_b, n, padded_n)
                logging.info(f"Padded to {padded_n}x{padded_n} for Strassen's algorithm")
            result_matrix = self.strassen_multiply(matrix_a, matrix_b, padded_n)
            # Trim back to original size
            result_matrix = result_matrix[:n, :n]
        elif operation == 'add':
            tasks = self.split_tasks(matrix_a, matrix_b, n, operation)
            intermediate_results = [self.assign_task((i % 5) + 1, task) for i, task in enumerate(tasks)]
            result_matrix = self.assemble_result(intermediate_results, n)
        elif operation == 'subtract':
            tasks = self.split_tasks(matrix_a, matrix_b, n, operation)
            intermediate_results = [self.assign_task((i % 5) + 1, task) for i, task in enumerate(tasks)]
            result_matrix = self.assemble_result(intermediate_results, n)
        elif operation == 'complement':
            tasks = self.split_tasks(matrix_a, None, n, operation)
            intermediate_results = [self.assign_task((i % 5) + 1, task) for i, task in enumerate(tasks)]
            result_matrix = self.assemble_result(intermediate_results, n)
        elif operation == 'transpose':
            tasks = self.split_tasks(matrix_a, None, n, operation)
            intermediate_results = [self.assign_task((i % 5) + 1, task) for i, task in enumerate(tasks)]
            result_matrix = self.assemble_result(intermediate_results, n)

        logging.info(f"Result of {operation}:\n{result_matrix}")

    def pad_matrix(self, matrix, orig_n, new_n):
        """Pad NumPy matrix with zeros to the next power of 2."""
        padded = np.zeros((new_n, new_n), dtype=int)
        padded[:orig_n, :orig_n] = matrix
        return padded

    def strassen_multiply(self, a, b, n):
        """Strassen's matrix multiplication with distributed multiplications using NumPy."""
        if n <= 2:  # Base case
            if n == 1:
                return np.array([[a[0, 0] * b[0, 0]]])
            return self.standard_multiply(a, b)

        # Split matrices into quadrants
        half = n // 2
        a11 = a[:half, :half]
        a12 = a[:half, half:]
        a21 = a[half:, :half]
        a22 = a[half:, half:]
        b11 = b[:half, :half]
        b12 = b[:half, half:]
        b21 = b[half:, :half]
        b22 = b[half:, half:]

        # Compute intermediate matrices locally with NumPy
        m1_a = a11 + a22
        m1_b = b11 + b22
        m2_a = a21 + a22
        m3_b = b12 - b22
        m4_b = b21 - b11
        m5_a = a12 - a22
        m6_a = a11 - a21
        m7_a = a11 + a12
        m7_b = b21 - b22

        # Distribute the 7 multiplication tasks
        tasks = [
            {'operation': 'strassen_multiply', 'matrix_a': m1_a.tolist(), 'matrix_b': m1_b.tolist(), 'size': half},
            {'operation': 'strassen_multiply', 'matrix_a': m2_a.tolist(), 'matrix_b': b11.tolist(), 'size': half},
            {'operation': 'strassen_multiply', 'matrix_a': a11.tolist(), 'matrix_b': m3_b.tolist(), 'size': half},
            {'operation': 'strassen_multiply', 'matrix_a': a22.tolist(), 'matrix_b': m4_b.tolist(), 'size': half},
            {'operation': 'strassen_multiply', 'matrix_a': m5_a.tolist(), 'matrix_b': b11.tolist(), 'size': half},
            {'operation': 'strassen_multiply', 'matrix_a': m6_a.tolist(), 'matrix_b': b22.tolist(), 'size': half},
            {'operation': 'strassen_multiply', 'matrix_a': m7_a.tolist(), 'matrix_b': m7_b.tolist(), 'size': half},
        ]

        p = [None] * 7
        for i, task in enumerate(tasks):
            worker_id = (i % 5) + 1
            p[i] = np.array(self.assign_task(worker_id, task))

        # Compute quadrants locally with NumPy
        c11 = p[0] + p[3] - p[4] + p[6]
        c12 = p[2] + p[4]
        c21 = p[1] + p[3]
        c22 = p[0] + p[2] - p[1] + p[5]

        # Combine quadrants into result matrix
        result = np.zeros((n, n), dtype=int)
        result[:half, :half] = c11
        result[:half, half:] = c12
        result[half:, :half] = c21
        result[half:, half:] = c22
        return result

    def standard_multiply(self, a, b):
        """Standard 2x2 matrix multiplication with NumPy."""
        return a @ b  # NumPy matrix multiplication

    def assign_task(self, worker_id, task):
        """Send task to a worker and get result as a list."""
        worker_port = 5000 + worker_id
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('worker' + str(worker_id), worker_port))
            message = json.dumps({'task': task})
            s.sendall(message.encode('utf-8'))
            data = s.recv(1024)
            response = json.loads(data.decode('utf-8'))
            return response['result']

    def run_worker(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            logging.info(f"Worker {self.node_id} listening on port {self.port}")
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024)
                    if not data:
                        continue
                    task_data = json.loads(data.decode('utf-8'))
                    result = self.compute_task(task_data['task'])
                    response = json.dumps({'result': result})
                    conn.sendall(response.encode('utf-8'))

    def split_tasks(self, matrix_a, matrix_b, n, operation):
        """Split tasks for non-multiplication operations."""
        tasks = []
        if operation in ['add', 'subtract']:
            for i in range(n):
                task = {
                    'operation': operation,
                    'row_a': matrix_a[i].tolist(),
                    'row_b': matrix_b[i].tolist() if matrix_b is not None else None,
                    'row_index': i
                }
                tasks.append(task)
        elif operation == 'complement':
            for i in range(n):
                task = {'operation': 'complement', 'row': matrix_a[i].tolist(), 'row_index': i}
                tasks.append(task)
        elif operation == 'transpose':
            for i in range(n):
                task = {'operation': 'transpose', 'matrix': matrix_a.tolist(), 'column_index': i}
                tasks.append(task)
        return tasks

    def compute_task(self, task):
        """Handle worker tasks with NumPy arrays."""
        operation = task['operation']
        if operation == 'add':
            row_a = np.array(task['row_a'])
            row_b = np.array(task['row_b'])
            result_row = (row_a + row_b).tolist()
        elif operation == 'subtract':
            row_a = np.array(task['row_a'])
            row_b = np.array(task['row_b'])
            result_row = (row_a - row_b).tolist()
        elif operation == 'complement':
            row = np.array(task['row'])
            result_row = (-row).tolist()
        elif operation == 'transpose':
            matrix = np.array(task['matrix'])
            n = len(matrix)
            column_index = task['column_index']
            result_row = matrix[:, column_index].tolist()
        elif operation == 'strassen_multiply':
            matrix_a = np.array(task['matrix_a'])
            matrix_b = np.array(task['matrix_b'])
            size = task['size']
            if size <= 2:
                if size == 1:
                    return [[matrix_a[0, 0] * matrix_b[0, 0]]]
                return (matrix_a @ matrix_b).tolist()
            # Workers handle base case only
            return (matrix_a @ matrix_b).tolist()

        time.sleep(self.delay)
        return {'row_index': task.get('row_index', 0), 'row_result': result_row} if 'row_result' in locals() else matrix_a.tolist()

    def assemble_result(self, intermediate_results, n):
        """Assemble results for non-multiplication operations into NumPy array."""
        result_matrix = np.zeros((n, n), dtype=int)
        for result in intermediate_results:
            row_index = result['row_index']
            result_matrix[row_index] = result['row_result']
        return result_matrix

if __name__ == '__main__':
    node_id = int(os.getenv('NODE_ID', '0'))
    delay = float(os.getenv('DELAY', '1.0'))
    is_coordinator = (node_id == 0)
    node = Node(node_id, delay, is_coordinator)
    node.run()