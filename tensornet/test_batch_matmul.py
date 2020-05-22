import torch
import time


def chain_matmul_square(As):
    """
    Matrix multiplication of chains of square matrices

    Parameters
    --------------
        As: Tensor of shape (L, ..., N, N)

            The list of tensors to multiply. It supports batches.
            
            - L: Lenght of the chain of matrices for the matmul
            - N: Size of the matrix (must be a square matrix) 

    Returns
    ------------
        As_matmul: Tensor of shape (..., N, N)
            The tensor resulting of the chain multiplication

    """

    As_matmul = As
    while As_matmul.shape[0] > 1:
        if As_matmul.shape[0] % 2:
            A_last = As_matmul[-1:]
        else:
            A_last = None
        
        As_matmul = torch.matmul(As_matmul[0:-1:2], As_matmul[1::2])
        if A_last is not None:
            As_matmul = torch.cat([As_matmul, A_last], dim=0)
    
    return As_matmul.squeeze(0)


if __name__ == "__main__":

    # Initialize chain of tensors for matmul
    batch = 4
    chain_length = 64
    size_A = 1000
    As = torch.randn(chain_length, batch, size_A, size_A)

    # Compute using `chain_matmul_square`
    start_time = time.time()
    As_comp = chain_matmul_square(As)
    exec_time = time.time() - start_time
    print('chain_matmul_square time: ', exec_time)

    # Compute by looping the chain in a standard way
    start_time = time.time()
    As_comp2 = As[0:1]
    for ii in range(1, As.shape[0]):
        As_comp2 = torch.matmul(As_comp2, As[ii])
    As_comp2 = As_comp2.squeeze(0)
    exec_time = time.time() - start_time
    print('matmul loop time: ', exec_time)

    # Make sure that the results from both methods are similar
    diff = (As_comp - As_comp2).abs()
    is_near_eq = torch.all((diff / As_comp2.abs().mean()) < 1e-5)
    print(is_near_eq)

    print('Done')
