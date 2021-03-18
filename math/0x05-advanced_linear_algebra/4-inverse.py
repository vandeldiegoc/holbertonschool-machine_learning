#!/usr/bin/env python3
"""module"""


def determinant(matrix):
    """ that calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or len(matrix) is 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) is 1 and len(matrix[0]) is 0:
        return 1

    for r in matrix:
        if not isinstance(r, list):
            raise TypeError("matrix must be a list of lists")

    for r in matrix:
        if len(r) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    if len(matrix) is 1 and len(matrix[0]) is 1:
        return matrix[0][0]

    if len(matrix) is 2 and len(matrix[0]) is 2:
        det = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return det

    detres = 0
    for i, j in enumerate(matrix[0]):
        row = [r for r in matrix[1:]]
        tmp = []

        for r in row:
            aux = []
            for c in range(len(matrix)):
                if c != i:
                    aux.append(r[c])
            tmp.append(aux)

        detres += j * (-1) ** i * determinant(tmp)

    return detres


def minor(matrix):
    """that calculates the minor matrix of a matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if not matrix:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) is 1 and len(matrix[0]) is 1:
        return [[1]]

    minres = [x[:] for x in matrix]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            minres[i][j] = determinant([row[:j] + row[j+1:]
                                        for row in (matrix[:i]+matrix[i+1:])])
    return minres


def cofactor(matrix):
    """that calculates the cofactor matrix of a matrix"""
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError('matrix must be a list of lists')

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    if matrix is [[]]:
        raise ValueError('matrix must be a non-empty square matrix')

    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a non-empty square matrix")

    cofres = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            cofres[i][j] *= (-1) ** (i+j)
    return cofres


def adjugate(matrix):
    """ that calculates the adjugate matrix of a matrix """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError('matrix must be a list of lists')

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    if matrix is [[]]:
        raise ValueError('matrix must be a non-empty square matrix')

    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a non-empty square matrix")

    adjres = cofactor(matrix)
    for i in range(len(adjres[0])):
        for j in range(len(adjres)):
            matrix[i][j] = adjres[j][i]
    return matrix


def inverse(matrix):
    """ that calculates the inverse of a matrix: """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError('matrix must be a list of lists')

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    if matrix is [[]]:
        raise ValueError('matrix must be a non-empty square matrix')

    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a non-empty square matrix")

    invres = determinant(matrix)
    if not invres:
        return None

    matrix = adjugate(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] /= invres
    return matrix
