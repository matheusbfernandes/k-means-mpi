import numpy as np
import pandas as pd
import time


np.random.seed(0)


class SKMeans(object):

    def __init__(self, num_clusters, dataset):
        self.num_clusters = num_clusters
        self.dataset = dataset
        self.centroides = self._inicializar_centroides()

    # adiciona valores iniciais aleatorios para os centroides,baseado em exemplos do dataset
    def _inicializar_centroides(self):
        centroides = []
        for _ in range(self.num_clusters):
            pos_aleatoria = np.random.randint(0, self.dataset.shape[0])
            centroides.append(self.dataset[pos_aleatoria])

        return centroides

    # computa a a funcao J baseado na distancia euclidiana
    # calculo consiste no somatorio de distancia de todos os exemplos do dataset para todos os centroides
    def _computar_custo(self):
        custo_total = 0.0
        for centroide in self.centroides:
            custo_total += np.sum(np.linalg.norm(self.dataset - centroide, axis=1))

        return custo_total

    # funcao auxiliar responsavel por retornar a posicao onde a distancia euclidiana de o menor valor
    # essa funcao e necessaria para gerar o vetor de centroides mais proximos para cada elemento do dataset
    def _posicao_centroide_proximo(self, dado):
        menor_distancia = np.linalg.norm(dado - self.centroides[0], axis=0)
        pos_menor_distancia = 0
        for i in range(1, len(self.centroides)):
            distancia_atual = np.linalg.norm(dado - self.centroides[i], axis=0)
            if distancia_atual < menor_distancia:
                menor_distancia = distancia_atual
                pos_menor_distancia = i

        return pos_menor_distancia

    # funcao principal, responsavel de fato gerar os k-centroides
    # a versao dessa funcao em pseudocodigo pode ser vista no relatorio entitulada SKMeans
    # a diferenca para o pseudocodigo e que o algoritmo implementado aqui foi quebrado em outras funcoes ja comentadas anteriormente
    def treinar(self, limite):
        nao_parar_treinamento = True
        pos_centroides_mais_proximos = np.zeros(self.dataset.shape[0], dtype=int)
        # codigo executa ate a condicao da linha 72 acontecer
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


def main():
    tempos_carregamento = []
    tempos_totais = []
    tempos_execucao = []

    # executa o codigo 10 vezes
    for i in range(10):
        start = time.time()
        dataset = pd.read_csv('creditcard.csv')
        dados = dataset.values
        tempo_load = time.time()

        k_means = SKMeans(4, dados)
        k_means.treinar(0.001)
        end = time.time()

        tempos_carregamento.append(tempo_load - start)
        tempos_totais.append(end - start)
        tempos_execucao.append(end - tempo_load)

        print("\nExecucao :{}:".format(i))
        print("Tempo de carregamento: " + str(tempos_carregamento[i]))
        print("Tempo de total: " + str(tempos_totais[i]))
        print("Tempo de execucao: " + str(tempos_execucao[i]))
    print("\n\nMedia das execucoes = " + str(np.mean(np.array(tempos_execucao))))  
    print("Media dos tempos totais = " + str(np.mean(np.array(tempos_totais))))
    print("Media dos tempos de carregamento = " + str(np.mean(np.array(tempos_carregamento))))      


if __name__ == '__main__':
    main()
