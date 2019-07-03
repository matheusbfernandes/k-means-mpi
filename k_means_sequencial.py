import numpy as np
import pandas as pd
import time


np.random.seed(0)


class SKMeans(object):
    def __init__(self, num_clusters, dataset):
        self.num_clusters = num_clusters
        self.dataset = dataset
        self.centroides = self._inicializar_centroides()

    def _inicializar_centroides(self):
        centroides = []
        for _ in range(self.num_clusters):
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
                self.centroides[i] = novo_centroide / len(dados_associados_centroide[0])

            j = self._computar_custo()
            if (j_linha - j) < limite:
                nao_parar_treinamento = False


def main():
    # dataset = pd.read_csv('Mall_Customers.csv')
    # dados = dataset.iloc[:, [3, 4]].values
    start = time.time()
    dataset = pd.read_csv('master.csv')
    dados = dataset.iloc[:, [4, 5]].values
    tempo_load = time.time()

    k_means = SKMeans(5, dados)
    k_means.treinar(0.001)
    print(np.array(k_means.centroides))
    end = time.time()
    print(tempo_load - start)
    print(end - start)



if __name__ == '__main__':
    main()
