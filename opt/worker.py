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
                    data = conn.recv(1024)
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
        elif operation == 'strassen_multiply':
            matrix_a = np.array(task['matrix_a'])
            matrix_b = np.array(task['matrix_b'])
            size = task['size']
            if size <= 2:
                if size == 1:
                    return [[matrix_a[0, 0] * matrix_b[0, 0]]]
                return standard_multiply(matrix_a, matrix_b).tolist()
            return standard_multiply(matrix_a, matrix_b).tolist()

        time.sleep(self.delay)
        return {'row_index': task.get('row_index', task.get('column_index', 0)), 'row_result': result_row} if 'row_result' in locals() else matrix_a.tolist()

if __name__ == '__main__':
    node_id = int(os.getenv('NODE_ID', '0'))
    delay = float(os.getenv('DELAY', '1.0'))
    worker = Worker(node_id, delay)
    worker.run()