from mpi4py import MPI
import numpy as np
import pandas as pd
import time


dataset = pd.read_csv('creditcard.csv')
dados = dataset.values


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
            custo_total += np.sum(np.linalg.norm(self.dataset - centroide, axis=1))

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

    @staticmethod
    def juntar(conjunto_centroides):
        centroides_finais = []
        while conjunto_centroides[0]:
            conjunto_centros = []
            for conjunto in conjunto_centroides:
                conjunto_array = np.asarray(conjunto)
                conjunto_centros.append(np.sum(conjunto_array, axis=0) / len(conjunto_array))

            menor_distancia = None
            for i in range(len(conjunto_centroides)):
                distancia = np.sum(np.linalg.norm(np.asarray(conjunto_centroides[i]) - conjunto_centros[i], axis=1))
                if menor_distancia is None or distancia < menor_distancia:
                    menor_distancia = distancia
                    pos_menor_centro = i
                conjunto_centroides[i].pop(0)

            centroides_finais.append(conjunto_centros[pos_menor_centro])

        return centroides_finais


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_processos = comm.Get_size()
    if rank == 0:
        start = time.time()
        chunk = len(dados) // num_processos

        ultimo_processo = num_processos - 1
        indices = [0, chunk]
        if ultimo_processo > 0:
            for i in range(1, ultimo_processo):
                comm.send([chunk * i, (chunk * i) + chunk], dest=i)
            comm.send([chunk * ultimo_processo, len(dados)], dest=ultimo_processo)
    else:
        indices = comm.recv(source=0)

    k_means = MKMeans(4, dados[indices[0]:indices[1], :])
    k_means.treinar(0.001)

    if rank > 0:
        comm.send(k_means.centroides, dest=0)
    else:
        conjunto_centroides = [k_means.centroides]
        for i in range(1, num_processos):
            conjunto_centroides.append(comm.recv(source=i))
        np.asarray(k_means.juntar(conjunto_centroides))
        end = time.time()
        print("Tempo de execucao = " + str(end - start))


if __name__ == '__main__':
    main()
