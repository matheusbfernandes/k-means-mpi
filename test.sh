for i in 1 2 3 4 5 6 7 8 9 10
do
    echo execucao $i
    mpiexec -n 8 python3 k_means_paralelo.py
done
