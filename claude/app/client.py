import socket
import pickle
import numpy as np
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Client for distributed matrix multiplication')
    parser.add_argument('--host', type=str, default='localhost', 
                        help='Coordinator host (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, 
                        help='Coordinator port (default: 5000)')
    parser.add_argument('--size', type=int, default=4, 
                        help='Matrix size (n x n) (default: 4)')
    args = parser.parse_args()
    
    # Generate random matrices for testing
    n = args.size
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    print(f"Matrix A ({n}x{n}):")
    print(A)
    print(f"\nMatrix B ({n}x{n}):")
    print(B)
    
    # Calculate expected result locally for verification
    expected_result = A @ B
    
    # Connect to coordinator
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"Connecting to coordinator at {args.host}:{args.port}...")
        client_socket.connect((args.host, args.port))
        
        # Send matrix multiplication request
        request = {
            "type": "submit_matrix_multiply",
            "A": A.tolist(),
            "B": B.tolist()
        }
        
        start_time = time.time()
        client_socket.sendall(pickle.dumps(request))
        
        # Receive the result
        data = client_socket.recv(4096)
        end_time = time.time()
        
        if data:
            response = pickle.loads(data)
            if response.get("type") == "multiplication_result":
                result = np.array(response["result"])
                
                print(f"\nDistributed multiplication result:")
                print(result)
                
                # Verify correctness
                error = np.max(np.abs(result - expected_result))
                print(f"\nMax error compared to NumPy result: {error}")
                
                if error < 1e-10:
                    print("Result is correct!")
                else:
                    print("Warning: Result has numerical differences")
                
                print(f"\nTime taken: {end_time - start_time:.4f} seconds")
            else:
                print(f"Unexpected response: {response}")
        else:
            print("No response received from server")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        client_socket.close()

if __name__ == "__main__":
    main()