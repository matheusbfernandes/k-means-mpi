# k-means-mpi

Implementação em Python 3.5 do algorítmo K-means

OBS: base de dados `creditcard.csv` não está no repositório. Se encontra, no entanto, disponível no seguinte link(https://www.kaggle.com/mlg-ulb/creditcardfraud). 

## Execução

Para executar qualquer um dos códigos, é necessário modificar o código para a base de dados que se está lendo e a leitura das features, para que só sejam considerados atributos numéricos. Faz-se essa alteração no trecho `dados = dataset.values` em ambos os códigos. Por, exemplo, para o dataset `iris.csv`, altera-se para `dados = dataset.iloc[:,[0,1,2,3]].values`. 

Para executar o Kmeans sequencial, digite:

```
python3 k_means_sequencial.py
```

Para executar o Kmeans paralelo, digite:

```
mpiexec -n <numero_de_processos> python3 k_means_paralelo.py
```

