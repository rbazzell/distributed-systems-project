# Distributed Matrix Multiplication with Strassen's Algorithm

This project implements a distributed system for matrix multiplication using Strassen's algorithm. The system consists of a coordinator node and multiple worker nodes, all running in Docker containers.

## Architecture

* **Coordinator** : Manages the distribution of tasks to workers and combines results
* **Workers** : Execute matrix multiplication tasks and Strassen decomposition
* **Docker** : Each component runs in its own container, orchestrated with Docker Compose

## How It Works

1. The coordinator accepts matrix multiplication requests from clients.
2. Initial matrix multiplication tasks are sent to available workers.
3. Workers apply Strassen's algorithm to decompose large matrix multiplications into 7 smaller multiplications.
4. These subtasks are sent back to the coordinator, which distributes them to available workers.
5. This process continues recursively until matrices reach the base case size (1x1).
6. Base case multiplications are performed directly, and results are returned to the coordinator.
7. The coordinator assigns combine tasks to workers to merge the results according to Strassen's algorithm.
8. The final result is returned to the client.

## Strassen's Algorithm

Strassen's algorithm reduces the time complexity of matrix multiplication from O(n³) to approximately O(n^2.81) by reducing the number of recursive multiplications from 8 to 7. The algorithm works by:

1. Dividing matrices into quadrants
2. Computing 7 matrix products (instead of the naive 8)
3. Combining these products to form the quadrants of the result matrix

## Files

* `coordinator.py`: Implements the coordinator node logic
* `worker.py`: Implements the worker node logic
* `client.py`: Test client to submit matrix multiplication jobs
* `Dockerfile`: Docker configuration for both coordinator and worker nodes
* `docker-compose.yml`: Configuration for the multi-container setup

## Setup and Running

### Prerequisites

* Docker and Docker Compose

### Running the System

1. Clone this repository
2. Build and start the containers:

   ```
   docker-compose up --build
   ```
3. In a separate terminal, run the client to test the system:

   ```
   python client.py --size 8
   ```

   Options:

   * `--host`: Coordinator host (default: localhost)
   * `--port`: Coordinator port (default: 5000)
   * `--size`: Matrix size (n×n) (default: 4)

## Scaling

To increase the number of worker nodes, modify the `docker-compose.yml` file to add more worker services. The system will automatically utilize all available workers.

## Performance Considerations

* Matrices are serialized using pickle for transmission between nodes
* For very large matrices, consider implementing more efficient serialization
* The base case size can be adjusted to optimize performance based on your hardware

## Error Handling

The system includes basic error handling for:

* Worker disconnections
* Task failures
* Network issues

## Future Improvements

* Add support for sparse matrices
* Implement more sophisticated load balancing
* Add real-time monitoring of task progress
* Optimize serialization for better performance with large matrices
