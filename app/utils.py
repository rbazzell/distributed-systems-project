import numpy as np
import hashlib
from enum import Enum
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

class TaskType(Enum):
    MULTIPLY = "multiply"
    COMBINE = "combine"

class Task:
    def __init__(self, task_type, matrices=None, subtasks_results=None, parent_id=None, m_number=None):
        self.task_type = task_type
        self.matrices = matrices  # For MULTIPLY : [A, B]
        self.subtasks_results = subtasks_results  # For COMBINE: [M1, M2, ..., M7]
        self.m_number = m_number # Also for COMBINE
        self.parent_id = parent_id  # ID of the parent task

        # Generate task ID based on content
        self.task_id = self._generate_id()
        
    def _generate_id(self):
        """Generate a readable ID based on task content"""
        if self.task_type == TaskType.MULTIPLY:
            # Use matrix dimensions and hash of content
            matrix_a, matrix_b = self.matrices
            dim_str = f"{matrix_a.shape[0]}x{matrix_a.shape[1]}_{matrix_b.shape[0]}x{matrix_b.shape[1]}"
            content_hash = hashlib.md5(np.array_str(matrix_a).encode() + np.array_str(matrix_b).encode()).hexdigest()[:6]
            return f"{self.task_type.value}_{dim_str}_{content_hash}"
        
        elif self.task_type == TaskType.COMBINE:
            # Use parent_id if available, otherwise timestamp
            base = self.parent_id if self.parent_id else str(int(self.created_at))
            return f"{self.task_type.value}_{base}_combine"
        
        # Fallback
        return f"{self.task_type.value}_{int(self.created_at)}"
    
    def to_dict(self):
        """Convert task to dictionary for JSON serialization"""
        result = {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "parent_id": self.parent_id,
            "m_number": self.m_number
        }
        
        if self.matrices is not None:
            result["matrices"] = [matrix.tolist() for matrix in self.matrices]
            
        if self.subtasks_results is not None:
            result["subtasks_results"] = [matrix.tolist() for matrix in self.subtasks_results]
            
        return result
        
    @classmethod
    def from_dict(cls, data):
        """Create a Task instance from a dictionary"""
        if "matrices" in data:
            matrices = [np.array(matrix) for matrix in data["matrices"]]
        else:
            matrices = None

        if "subtasks_results" in data:
            subtasks_results = [np.array(matrix) for matrix in data["subtasks_results"]]
        else:
            subtasks_results = None

        task = cls(
            task_type=TaskType(data["task_type"]),
            parent_id=data.get("parent_id"),
            matrices=matrices,
            subtasks_results=subtasks_results,
            m_number=data.get("m_number")
        )
        task.task_id = data["task_id"]
        
        return task


def pad_matrices(A, B):
    """Pad matrices to the next power of 2 if needed for Strassen's algorithm"""
    n = max(A.shape[0], A.shape[1], B.shape[0], B.shape[1])
    
    # Find the next power of 2
    m = 1
    while m < n:
        m *= 2
        
    # Pad matrices
    A_padded = np.zeros((m, m), dtype=np.int32)
    B_padded = np.zeros((m, m), dtype=np.int32)
    
    A_padded[:A.shape[0], :A.shape[1]] = A
    B_padded[:B.shape[0], :B.shape[1]] = B
    
    return A_padded, B_padded, A.shape, B.shape

def unpad_matrix(C_padded, original_A_shape, original_B_shape):
    """Extract original sized result from padded result matrix"""
    return C_padded[:original_A_shape[0], :original_B_shape[1]]

def split_matrix(matrix):
    """Split matrix into 4 quadrants"""
    n = matrix.shape[0] // 2
    
    a11 = matrix[:n, :n]
    a12 = matrix[:n, n:]
    a21 = matrix[n:, :n]
    a22 = matrix[n:, n:]
    
    return a11, a12, a21, a22

def join_matrices(c11, c12, c21, c22):
    """Join 4 quadrants into a single matrix"""
    n = c11.shape[0]
    result = np.zeros((2*n, 2*n), dtype=np.int32)
    
    result[:n, :n] = c11
    result[:n, n:] = c12
    result[n:, :n] = c21
    result[n:, n:] = c22
    
    return result

def create_retry_session(retries=3, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504, 104)):
    """Creates a session that automatically retries communication on a failure"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    return session