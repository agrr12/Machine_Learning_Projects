import random
import numpy as np
import KNN as K

#Função auxiliar para normalização de um vetor
def normalizarVetor (vetor):
    return (vetor-np.mean(vetor))/np.std(vetor)

#Divide a base de dados em uma porção para treino e outra para teste
def criarTestTrain(porcentagemTest, X, y, randomSeed):
    random.seed(randomSeed) #Cria a seed para tornar o exemplo repetível
    X_train = []
    X_test = X.tolist()
    y_train = []
    y_test = y.tolist()
    #Roda um laço até as quantidades em train e test obedecerem a proporção passada como parâmetro
    while len(X_test)>len(X)*porcentagemTest:
        valorRandom = random.randrange(0, len(X_test)-1) #Seleciona um dos indexes possíveis do conjunto de dados
        valorX = X_test.pop(valorRandom) #Remove o item daquele index do array de X
        valory = y_test.pop(valorRandom) #Remove o item daquele index do array de y
        X_train.append(valorX) #Coloca o item removido no array de treino de X
        y_train.append(valory) #Coloca o item removido no array de treino de y
    return X_train, X_test, y_train, y_test

def Holdout_Acuracia(X_train, X_test, y_train, y_test, k):
    acertos = 0 #Armazena a quantidade de acertos
    classificador = []  #Armazena a classe sugerida pelo KNN e a classe correta (Só para registro)
    for index in range(len(X_test)): #Itera por todas as instâncias de teste
        resultado = K.KNN(k, X_train, y_train, X_test[index]) #Classifica a instância
        classificador.append([resultado,y_test[index]]) #Armazena a classificação feita e a classe correta
        if resultado==y_test[index]: #Verifica se a classificação está correta
            acertos+=1
    return acertos/len(y_test)
