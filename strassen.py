import numpy as np

def strassen(A, B, n):
    if n == 1:
        return np.matrix(f"{A[0, 0]*B[0, 0]}")

    half = n // 2
    A11 = A[:half, :half]
    A12 = A[:half, half:]
    A21 = A[half:, :half]
    A22 = A[half:, half:]

    B11 = B[:half, :half]
    B12 = B[:half, half:]
    B21 = B[half:, :half]
    B22 = B[half:, half:]

    P1 = A11 + A22
    P2 = B11 + B22
    P3 = A21 + A22
    P4 = B12 - B22
    P5 = B21 - B11
    P6 = A21 - A11
    P7 = A12 - A22
    P8 = B21 + B22

    M1 = strassen(A11 + A22, B11 + B22, half)
    M2 = strassen(A21 + A22, B11, half)
    M3 = strassen(A11, B12 - B22, half)
    M4 = strassen(A22, B21 - B11, half)
    M5 = strassen(A11 + A12, B22, half)
    M6 = strassen(A21 - A11, B11 + B12, half)
    M7 = strassen(A12 - A22, B21 + B22, half)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = np.zeros((n, n), dtype=int)
    C[:half, :half] = C11
    C[:half, half:] = C12
    C[half:, :half] = C21
    C[half:, half:] = C22

    return C

A = np.matrix("5 6 6 3; 3 4 9 8; 2 6 9 4; 7 1 1 4")
B = np.matrix("2 2 2 7; 2 2 5 2; 3 1 9 9; 8 2 3 4")
#A = np.matrix("5 3 2 7; 6 4 6 1; 6 9 9 1; 3 8 4 4")
#B = np.matrix("2 2 3 8; 2 2 1 2; 2 5 9 3; 7 2 9 4")
C = strassen(A, B, 4)
print(C)