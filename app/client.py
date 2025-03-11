import sys
import time
import numpy as np
import requests
import json
import os
import csv

def generate_random_matrix(rows, cols):
    """Generate a random matrix with integer values"""
    return np.random.randint(0, 10, size=(rows, cols))

def retrieve_matrices_from_file(filename):
    """Returns 2 matrices pulled from a text file"""
    try:
        with open(filename) as f:
            A, B = f.read().split("\n\n")
        A = np.matrix(";".join(A.split("\n")))
        B = np.matrix(";".join(B.split("\n")))
        return A, B
    except Exception as e:
        print(f"Error opening file: {e}")
        return None, None

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
        print("   OR: python client.py <coordinator_url> -f <filepath>")
        print("Example: python client.py http://localhost:5000 4,4 4,3")
        print("     OR: python client.py http://localhost:5000 -f data/matrices.txt")
        return 1
    
    if len(sys.argv) == 6:
        log = True
        log_file = sys.argv[5]
    
    coordinator_url = sys.argv[1]
    
    if sys.argv[2] == "-f":
        filename = sys.argv[3]
        matrix_a, matrix_b = retrieve_matrices_from_file(filename)
        if not isinstance(matrix_a, np.matrix) or not isinstance(matrix_b, np.matrix):
            return 1
    else:
        # Parse matrix sizes
        a_size = [int(x) for x in sys.argv[2].split(',')]
        b_size = [int(x) for x in sys.argv[3].split(',')]
    
        if len(a_size) != 2 or len(b_size) != 2:
            print("Matrix sizes should be specified as rows,cols")
            return 1
        
        if a_size[1] != b_size[0]:
            print(f"Incompatible matrix dimensions: {a_size} and {b_size}")
            return 1
    
        # Generate random matrices
        matrix_a = generate_random_matrix(a_size[0], a_size[1])
        matrix_b = generate_random_matrix(b_size[0], b_size[1])
    
    print("Matrix A:")
    print(matrix_a)
    print("\nMatrix B:")
    print(matrix_b)
    
    # Expected result using numpy
    start_time = time.time_ns()
    expected = matrix_a @ matrix_b
    end_time = time.time_ns()
    print("\nExpected result:")
    print(expected)

    expected_time = end_time - start_time
    
    with open("app/data/expected_results.txt", 'w') as f:
        f.write(";\n".join(" ".join(str(y) for y in x) for x in expected.tolist()))

    
    # Delete previous task's result
    if os.path.exists("app/data/results.txt"):
        os.remove("app/data/results.txt")



    # Submit task to coordinator
    print("\nSubmitting task to coordinator...")
    
    start_time = time.time_ns()
    task_id = multiply_matrices(coordinator_url, matrix_a, matrix_b)

    if not task_id:
        return 1

    print("Final result can be found in app/data/results.txt or in the console log of the coordinator")
    print("Waiting for result to validate...")
    while not os.path.exists("app/data/results.txt"): pass
    end_time = time.time_ns()

    result_time = end_time - start_time
    time.sleep(0.25)
    with open("app/data/results.txt", 'r') as f:
        result = f.read()
    result = np.matrix(result)
    print("\nResult:")
    print(result)
    if log:
        with open(log_file, "a", newline='') as c:
            writer = csv.writer(c, delimiter=',')
            writer.writerow(a_size + b_size + [expected_time / 1e6, result_time / 1e6]) # time in ms

    if result.shape == expected.shape and (result == expected).all():
        print("Results match!")
        return 0
    else:
        print("Results do not match!")
        return 1

    
if __name__ == "__main__":
    code = main()
    exit(code)