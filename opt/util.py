import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def pad_matrix(matrix, orig_n, new_n):
    """Pad NumPy matrix with zeros to the next power of 2."""
    padded = np.zeros((new_n, new_n), dtype=float)
    padded[:orig_n, :orig_n] = matrix
    return padded

def standard_multiply(a, b):
    """Standard 2x2 matrix multiplication with NumPy."""
    return a @ b

def split_tasks(matrix_a, matrix_b, n, operation):
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
    elif operation in ['complement', 'transpose']:
        for i in range(n):
            task = {
                'operation': operation,
                'row': matrix_a[i].tolist() if operation == 'complement' else None,
                'matrix': matrix_a.tolist() if operation == 'transpose' else None,
                'row_index': i if operation == 'complement' else None,
                'column_index': i if operation == 'transpose' else None
            }
            tasks.append(task)
    return tasks

def assemble_result(intermediate_results, n):
    """Assemble results into NumPy array."""
    result_matrix = np.zeros((n, n), dtype=float)
    for result in intermediate_results:
        row_index = result['row_index']
        result_matrix[row_index] = result['row_result']
    return result_matrix