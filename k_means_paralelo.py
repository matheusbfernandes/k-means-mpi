from mpi4py import MPI
import numpy as np
import pandas as pd


dataset = pd.read_csv('Mall_Customers.csv')
dados = dataset.iloc[:, [3, 4]].values

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    num_processos = comm.Get_size()
    chunk = len(dados) // num_processos

    ultimo_processo = num_processos - 1
    for i in range(1, ultimo_processo):
        comm.send([chunk * i, (chunk * i) + chunk], dest=i)
    comm.send([chunk * ultimo_processo, len(dados)], dest=ultimo_processo)
else:
    indices = comm.recv(source=0)
    print(indices)
