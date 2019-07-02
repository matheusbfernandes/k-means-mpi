from mpi4py import MPI
import numpy as np
import pandas as pd


dataset = pd.read_csv('Mall_Customers.csv')
dados = dataset.iloc[:, [3, 4]].values


np.random.seed(0)


class MKMeans(object):
    def __init__(self, num_centroides, data_set):
        self.num_centroides = num_centroides
        self.dataset = data_set
        self.centroides = self._inicializar_centroides()

    def _inicializar_centroides(self):
        centroides = []
        for _ in range(self.num_centroides):
            pos_aleatoria = np.random.randint(0, self.dataset.shape[0])
            centroides.append(self.dataset[pos_aleatoria])

        return centroides

    def _computar_custo(self):
        custo_total = 0.0
        for centroide in self.centroides:
            custo_total += np.sum(np.linalg.norm(self.dataset - centroide, axis=0))

        return custo_total

    def _posicao_centroide_proximo(self, dado):
        menor_distancia = np.linalg.norm(dado - self.centroides[0], axis=0)
        pos_menor_distancia = 0
        for i in range(1, len(self.centroides)):
            distancia_atual = np.linalg.norm(dado - self.centroides[i], axis=0)
            if distancia_atual < menor_distancia:
                menor_distancia = distancia_atual
                pos_menor_distancia = i

        return pos_menor_distancia

    def treinar(self, limite):
        nao_parar_treinamento = True
        pos_centroides_mais_proximos = np.zeros(self.dataset.shape[0], dtype=int)
        while nao_parar_treinamento:
            j_linha = self._computar_custo()

            for i in range(len(self.dataset)):
                pos_centroides_mais_proximos[i] = self._posicao_centroide_proximo(self.dataset[i])

            for i in range(len(self.centroides)):
                dados_associados_centroide = np.where(pos_centroides_mais_proximos == i)
                novo_centroide = np.zeros(self.dataset.shape[1])
                for pos_dado in dados_associados_centroide[0]:
                    novo_centroide += self.dataset[pos_dado]
                if len(dados_associados_centroide[0]) != 0:
                    self.centroides[i] = novo_centroide / len(dados_associados_centroide[0])

            j = self._computar_custo()
            if (j_linha - j) < limite:
                nao_parar_treinamento = False

    def juntar(self, conjunto_centroides):
        pass


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_processos = comm.Get_size()
    if rank == 0:
        chunk = len(dados) // num_processos

        ultimo_processo = num_processos - 1
        indices = [0, chunk]
        if ultimo_processo > 0:
            for i in range(1, ultimo_processo):
                comm.send([chunk * i, (chunk * i) + chunk], dest=i)
            comm.send([chunk * ultimo_processo, len(dados)], dest=ultimo_processo)
    else:
        indices = comm.recv(source=0)

    k_means = MKMeans(5, dados[indices[0]:indices[1], :])
    k_means.treinar(0.001)

    if rank > 0:
        comm.send(k_means.centroides, dest=0)
    else:
        conjunto_centroides = k_means.centroides
        for i in range(1, num_processos):
            conjunto_centroides += comm.recv(source=i)
        print(np.asarray(conjunto_centroides))


if __name__ == '__main__':
    main()
