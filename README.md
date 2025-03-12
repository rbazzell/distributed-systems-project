# Distributed Matrix Multiplication with Strassen's Algorithm

This project implements a distributed system for matrix multiplication using Strassen's algorithm. The system consists of a coordinator node and multiple worker nodes that communicate via HTTP APIs.

## Architecture

* **Coordinator** : Manages tasks, distributes work to workers, and aggregates results
* **Workers** : Process matrix multiplication tasks and return results to the coordinator
* **Docker** : Each node runs in its own Docker container
* **Strassen's Algorithm** : Reduces the number of multiplications from O(n³) to O(n^log₂(7)) ≈ O(n^2.81)

## Directory Structure

```
project-root/
├── app/
│   ├── _pycache_/
│   ├── data/
|   |   ├── expected.txt
|   │   └── results.txt
│   ├── client.py
│   ├── coordinator.py
│   ├── utils.py
│   └── worker.py
├── test/
│   ├── generators/
│   ├── graphs/
│   ├── logs/
│   ├── gen_docker.py
│   ├── graph_logs.py
│   ├── run_tests_linux.py
│   └── run_tests_windows.py
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## How It Works

1. The coordinator receives a matrix multiplication task from a client
2. The coordinator sends the task to a worker
3. The worker applies Strassen's algorithm to divide the task into 7 subtasks
4. The coordinator distributes these subtasks to workers
5. This process continues recursively until the matrices are small enough for direct multiplication
6. Results are passed back up the chain, with the coordinator managing the aggregation
7. The final result is printed in the coordinator logs and to the app/data/results.txt file
8. The client then retrieves this result and compares it with the expected result, optionally logging the computation time

## Task Identification

Each task has a unique identifier based on:

* Task type (multiply, combine)
* Matrix dimensions
* Content hash
* Parent task ID (for subtasks)

This makes debugging easier by providing readable task IDs that help trace the computation flow.

## Running the System

### Prerequisites

* Docker and Docker Compose
* Python 3

### Starting the System

1. (Optional) Generate a new docker-compose.yml file:
  
   ```
   python test/gen_docker.py -f <generator_file>
   ```

   Example generator files can be found in the test/generators directory

2. Build and start the containers:

   ```
   docker-compose up --build
   ```
3. Submit a matrix multiplication task using the client:
   
   For a random task:
   ```
   python app/client.py http://localhost:5000 4,4 4,3
   ```
   Where `4,4` and `4,3` are the dimensions of matrices A and B respectively.


   For pre-determined matrix multiplication tasks:
   ```
   python app/client.py http://localhost:5000 -f <matrix_file>
   ```

### Testing the System

1. Run the testing file for your appropriate operating system:

   ```
   python test/run_tests_<linux or windows>.py
   ```

2. Follow the prompts to select the performance suite you would like to run
3. Observe the data in the tests/logs directory
4. (Optional) Generate graphs of your data:

   ```
   python test/graph_logs.py <log_file> <output_file> <-t "graph title"> <-a>
   ```

   Where ```-t``` sets the title of the graph and ```-a``` puts the average times in the legend.

## Scaling

You can modify the number of workers and the individual strength of each worker using generator files or by directly modifying the docker-compose.yml file.

## Implementation Details

### Strassen's Algorithm

The traditional matrix multiplication requires O(n³) operations. Strassen's algorithm reduces this to approximately O(n^2.81) by:

1. Dividing each matrix into four quadrants
2. Creating 7 products (instead of the naive 8)
3. Combining these products to form the quadrants of the result matrix

### Distributed Processing

* Tasks are distributed among workers to parallelize computation
* The coordinator manages the task distribution and result aggregation
* The system can handle dynamic worker registration
* Tasks are represented as Python objects and serialized to JSON for communication

### Error Handling

* The system includes timeout handling for network requests
* Worker registration is attempted repeatedly until successful
* Task processing is performed in separate threads to prevent blocking

## Performance Considerations

* Matrices are padded to dimensions that are powers of 2 for Strassen's algorithm
* Small matrices (dimensions <= 2) are multiplied directly for efficiency
* Round-robin task distribution is used (can be enhanced with load balancing)
* Threading is used to process tasks asynchronously

## Testing and Performance Analysis

The `test` directory contains tools for:
* Generating Docker Compose environments (`generators/`)
* Running performance tests on Linux and Windows (`run_tests_linux.py`, `run_tests_windows.py`)
* Analyzing and visualizing results (`graph_logs.py`)
* Performance data at different scales (`logs/`)