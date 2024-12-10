# Copyright 2023, Andreas Waechter
# All rights reserved.
# Author: Andreas Waechter
# Distribution or modification only permitted with written permission
# of the author.

import numpy as np
import scipy as sp
import scipy.sparse
import panuapardiso as p

#######################################################################
# Create the matrix.  Note that we are specifying only the upper
# triangular part of the symmtric indefinite matrix.
#######################################################################

n = 8
ia = np.array([0, 4, 7, 9, 11, 14, 16, 17, 18])
ja = np.array([0, 2, 5, 6, 1, 2, 4, 2, 7, 3, 6, 4, 5, 6, 5, 7, 6, 7])
a = np.array(
    [
        7.0,
        1.0,
        2.0,
        7.0,
        -4.0,
        8.0,
        2.0,
        1.0,
        5.0,
        7.0,
        9.0,
        5.0,
        -1.0,
        5.0,
        0.0,
        5.0,
        11.0,
        5.0,
    ]
)

A_csr = sp.sparse.csr_matrix((a, ja, ia))

# Get the dimension of the matrix
n = A_csr.shape[0]

print("Input matrix of dimension n = {}:".format(n))
print(A_csr)

#######################################################################
# Create the PanuaPardiso object
#######################################################################

# Symmetric indefinite matrix
mtype = -2
# The matrix is already in upper triangular format
is_upper_triangular = True
# The matrix has nonzeros on all diagonal elements
has_full_diagonal = True
# Set this to True if you want to see detailed Parido output
verbose = False

#######################################################################
# Create the Pardiso solver object
#######################################################################

# Unless the shared library is automatically found because some path
# environment variable has been set, we need to provide the path to
# the location of the shared library.
libdir = "/Users/Dimosthenis/Library/CloudStorage/OneDrive-USI/PANUA/panua-pardiso-20230718-mac_arm64/lib"

psolver = p.PanuaPardiso(
    mtype=-2,
    verbose=verbose,
    is_upper_triangular=is_upper_triangular,
    has_full_diagonal=has_full_diagonal,
    libdir=libdir,
)

#######################################################################
# Perform symbolic and numberical optimization
#######################################################################

# Set to true since we have not yet done the symbolic factorization'
# of a matrix with the same sparsity structure as the input matrix
new_structure = True
# This argument does not matter here, since the sparsity structure
# of the matrix is already in the format that Pardiso requires, as
# indicated by the argument for the constructore (is
# upper_triangular=True and has_full_diagonal=True).
already_corrected = True
# Call pardiso
error = psolver.factorize(
    A_csr, new_structure=new_structure, already_corrected=already_corrected
)
print("\nFactorize error = {} ".format(error))
if error != 0:
    raise "Error factorizing the matrix"

#######################################################################
# Create a right-hand side
#######################################################################

b = np.zeros(n, dtype=np.double)
b[1] = 1.0
print("\nRight hand side b:")
print(b)

#######################################################################
# Solve the linear system
#######################################################################

# Set to False since we already factorized this matrix
new_matrix = False
# Set to False since the input matrix has the same sparsity structure
# as the most recently factorized matrix
new_structure = False
# Call Pardiso
(x, error) = psolver.solve(A_csr, b, new_matrix=new_matrix, new_structure=new_structure)

print("\nSolving linear system error = {} ".format(error))
if error != 0:
    raise "Error solving the linear system"
print("\nSolution of the linear system")
print(x)

#######################################################################
# Compute some non-zero elements of the inverse of the matrix
#######################################################################

(A_inv, error) = psolver.selected_inversion(A_csr)

print("\nSelected inversion error = {} ".format(error))
if error != 0:
    raise "Error performing the selected inversion"
print("Some nonzero element of the matrix inverse")
print(A_inv)

# Now extract the diagonal only:
A_inv_diag = A_inv.diagonal()
print("\nDiagonal of the inverse of the matrix")
print(A_inv_diag)

#######################################################################
# Now change some nonzero elements of the matrix
#######################################################################

A_csr[1, 1] = 5.0
print("\nModified matrix with same sparsity structure:")
print(A_csr)

#######################################################################
# Factorize the modified matrix
#######################################################################

# If we now to the factorization of the new matrix, we don't
# need to repeat the symbolic factorization
new_structure = False

# This time, we ask Pardiso to also compute the determinant of the
# matrix, see IPARM(33) in the Pardiso manual
psolver.set_IPARM(33, 1)

# Call pardiso
error = psolver.factorize(
    A_csr, new_structure=new_structure, already_corrected=already_corrected
)
print("\nFactorize modified matrix error = {} ".format(error))
if error != 0:
    raise "Error factorizing the modified matrix"

# Let's get the computed determinant
determinant = psolver.get_DPARM(33)
print("\nDeterminant of that factorized matrix: {}".format(determinant))

#######################################################################
# Compute the diagonal of the inverse for the modified matrix
#######################################################################

(A_inv, error) = psolver.selected_inversion(A_csr)

print("\nSelected inversion of modified matrix error = {} ".format(error))
if error != 0:
    raise "Error performing the selected inversion for the modified matrix"

A_inv_diag = A_inv.diagonal()
print("\nDiagonal of the inverse of the modified matrix")
print(A_inv_diag)
