import numpy as np

def distanciaEuclidiana(ponto1, ponto2):
    return np.sqrt(np.sum((ponto1 - ponto2)**2))


#Recebe o vetor de vizinhos e retorna o mais frequente
def classeMaisFrequente(kVizinhos):
    valores = {} #Armazena classe:ocorrências
    #Itera pelos vizinhos contabilizando suas ocorrências
    for elemento in kVizinhos:
        if elemento in valores:
            valores[elemento]+=1
        else:
            valores[elemento]=1
    #Retorna a chave com o maior valor no dicionário
    return max(valores, key=valores.get)

#KNN
#Assume que a classe é identificada pelo Index
def KNN(k, X_train, y_train, arrayInstancia):
    dic = {} #Armazena Distancia:Index
    #Itera pela base de dados calculando a distância entre cada instância e o dado passado
    for indexInstancia in range(len(X_train)):
        dic[distanciaEuclidiana(arrayInstancia, X_train[indexInstancia])] = y_train[indexInstancia]
    kMaiores = sorted(dic.keys()) #Organiza as distâncias do menor para o maior
    kVizinhos = [] #Armazena as K classes vizinhas
    for x in range(k):
        kVizinhos.append(dic[kMaiores[x]])
    return classeMaisFrequente(kVizinhos)
