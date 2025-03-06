import sys
import time
import numpy as np
import requests
import json

def generate_random_matrix(rows, cols):
    """Generate a random matrix with integer values"""
    return np.random.randint(0, 10, size=(rows, cols))

def multiply_matrices(coordinator_url, matrix_a, matrix_b):
    """Submit a matrix multiplication task to the coordinator"""
    try:
        response = requests.post(
            f"{coordinator_url}/submit",
            json={
                'matrix_a': matrix_a.tolist(),
                'matrix_b': matrix_b.tolist()
            },
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"Error submitting task: {response.status_code}")
            return None
        
        result = response.json()
        task_id = result.get('task_id')
        
        print(f"Submitted task with ID: {task_id}")
        return task_id
    
    except Exception as e:
        print(f"Error during submission: {e}")
        return None

def main():
    if len(sys.argv) < 4:
        print("Usage: python client.py <coordinator_url> <matrix_a_size> <matrix_b_size>")
        print("Example: python client.py http://localhost:5000 4,4 4,3")
        return
    
    coordinator_url = sys.argv[1]
    
    # Parse matrix sizes
    a_size = [int(x) for x in sys.argv[2].split(',')]
    b_size = [int(x) for x in sys.argv[3].split(',')]
    
    if len(a_size) != 2 or len(b_size) != 2:
        print("Matrix sizes should be specified as rows,cols")
        return
    
    if a_size[1] != b_size[0]:
        print(f"Incompatible matrix dimensions: {a_size} and {b_size}")
        return
    
    # Generate random matrices
    matrix_a = generate_random_matrix(a_size[0], a_size[1])
    matrix_b = generate_random_matrix(b_size[0], b_size[1])
    #matrix_a = np.matrix("1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1")
    #matrix_b = np.matrix("1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4")
    
    print("Matrix A:")
    print(matrix_a)
    print("\nMatrix B:")
    print(matrix_b)
    
    # Expected result using numpy
    expected = matrix_a @ matrix_b
    print("\nExpected result:")
    print(expected)
    
    # Submit task to coordinator
    print("\nSubmitting task to coordinator...")
    task_id = multiply_matrices(coordinator_url, matrix_a, matrix_b)
    
    if task_id:
        print(f"Task submitted with ID: {task_id}")
        print("Check the coordinator logs for the final result.")
    
if __name__ == "__main__":
    main()