import numpy as np

A = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
], dtype=float)

b = np.array([18, 26, 34, 82], dtype=float)


def cramer_method(A, b):
    det_A = np.linalg.det(A)
    if det_A == 0:
        raise ValueError("The matrix is singular, no unique solution exists.")
    n = len(b)
    solutions = []
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b  # Replace column i with vector b
        det_A_i = np.linalg.det(A_i)
        solutions.append(det_A_i / det_A)
    return solutions


def gaussian_elimination(A, b):
    n = len(b)
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))  # Form augmented matrix

    # Forward elimination
    for i in range(n):
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]  # Normalize pivot
        for j in range(i + 1, n):
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j, i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:n])

    return x


def jacobi_method(A, b, tol=1e-6, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)  # Initial guess
    x_new = x.copy()

    for _ in range(max_iterations):
        for i in range(n):
            sum_ = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:  # Convergence check
            return x_new

        x = x_new.copy()

    raise ValueError("Jacobi method did not converge within the maximum number of iterations.")


def gauss_seidel_method(A, b, tol=1e-6, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)  # Initial guess

    for _ in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # Use updated values
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])  # Use old values
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:  # Convergence check
            return x_new

        x = x_new.copy()

    raise ValueError("Gauss-Seidel method did not converge within the maximum number of iterations.")


def print_solution(method_name, solution):
    print(f"\nSolution using {method_name}:")
    for i, value in enumerate(solution, start=1):
        print(f"x{i} = {value:.6f}")


def main():
    try:
        cramer_solution = cramer_method(A, b)
        print_solution("Cramer's Method", cramer_solution)

        gaussian_solution = gaussian_elimination(A.copy(), b.copy())
        print_solution("Gaussian Elimination", gaussian_solution)

        gauss_seidel_solution = gauss_seidel_method(A.copy(), b.copy())
        print_solution("Gauss-Seidel Method", gauss_seidel_solution)

        jacobi_solution = jacobi_method(A.copy(), b.copy())
        print_solution("Jacobi Method", jacobi_solution)

    except ValueError as e:
        print("\nError:", e)


if __name__ == "__main__":
    main()
