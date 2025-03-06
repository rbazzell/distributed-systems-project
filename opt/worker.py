import socket
import json
import time
import os
import logging
import numpy as np
from util import standard_multiply

logging.basicConfig(level=logging.INFO, format='%(message)s')

class Worker:
    def __init__(self, node_id, delay):
        self.node_id = node_id
        self.delay = delay
        self.host = '0.0.0.0'
        self.port = 5000 + node_id

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            logging.info(f"Worker {self.node_id} listening on port {self.port}")
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(4096)  # Increased buffer for larger submatrices
                    if not data:
                        continue
                    task_data = json.loads(data.decode('utf-8'))
                    result = self.compute_task(task_data['task'])
                    response = json.dumps({'result': result})
                    conn.sendall(response.encode('utf-8'))

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
        elif operation == 'multiply':
            matrix_a = np.array(task['matrix_a'])
            matrix_b = np.array(task['matrix_b'])
            result = standard_multiply(matrix_a, matrix_b).tolist()
        elif operation == 'strassen':
            matrix_a = np.array(task['matrix_a'])
            matrix_b = np.array(task['matrix_b'])
            size = task['size']
            result = self.strassen_multiply(matrix_a, matrix_b, size).tolist()

        time.sleep(self.delay)
        return {'row_index': task.get('row_index', task.get('column_index', 0)), 'row_result': result_row} if 'row_result' in locals() else result

    def strassen_multiply(self, a, b, n):
        """Recursive Strassen's multiplication on worker."""
        if n <= 2:
            if n == 1:
                return np.array([[a[0, 0] * b[0, 0]]])
            return standard_multiply(a, b)

        half = n // 2
        a11 = a[:half, :half]
        a12 = a[:half, half:]
        a21 = a[half:, :half]
        a22 = a[half:, half:]
        b11 = b[:half, :half]
        b12 = b[:half, half:]
        b21 = b[half:, :half]
        b22 = b[half:, half:]

        m1 = self.strassen_multiply(a11 + a22, b11 + b22, half)
        m2 = self.strassen_multiply(a21 + a22, b11, half)
        m3 = self.strassen_multiply(a11, b12 - b22, half)
        m4 = self.strassen_multiply(a22, b21 - b11, half)
        m5 = self.strassen_multiply(a12 - a22, b11, half)
        m6 = self.strassen_multiply(a11 - a21, b22, half)
        m7 = self.strassen_multiply(a11 + a12, b21 - b22, half)

        c11 = m1 + m4 - m5 + m7
        c12 = m3 + m5
        c21 = m2 + m4
        c22 = m1 + m3 - m2 + m6

        result = np.zeros((n, n), dtype=float)
        result[:half, :half] = c11
        result[:half, half:] = c12
        result[half:, :half] = c21
        result[half:, half:] = c22
        return result

if __name__ == '__main__':
    node_id = int(os.getenv('NODE_ID', '0'))
    delay = float(os.getenv('DELAY', '1.0'))
    worker = Worker(node_id, delay)
    worker.run()