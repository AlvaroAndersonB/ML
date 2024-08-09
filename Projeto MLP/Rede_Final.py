from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('alzheimers_disease_data.csv')

X = data.iloc[:, 1:-2].values
y = data.iloc[:, -2].values

y = y.astype(float)

# Normalização/Padronização Z-Score
X = (X - X.mean(axis=0)) / X.std(axis=0)

np.random.seed()
indices = np.random.permutation(X.shape[0])

# Tamanho
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))
test_size = len(data) - train_size - val_size

# Dividi os dados pelos índice
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Coloca os índices random e agrupa com o rótulo
X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]
X_test, y_test = X[test_indices], y[test_indices]

# Inicializar Parâmetros


def inicializando_parametros(dimensao_entrada, camada_escondida1, camada_escondida2, camada_saida):
    W1 = np.random.randn(dimensao_entrada, camada_escondida1) * 0.01
    b1 = np.zeros((1, camada_escondida1))
    W2 = np.random.randn(camada_escondida1, camada_escondida2) * 0.01
    b2 = np.zeros((1, camada_escondida2))
    W3 = np.random.randn(camada_escondida2, camada_saida) * 0.01
    b3 = np.zeros((1, camada_saida))

    return W1, b1, W2, b2, W3, b3


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivada(Z):
    return Z * (1 - Z)


def tanh(Z):
    return np.tanh(Z)


def tanh_derivada(Z):
    return 1 - np.tanh(Z)**2


def forward_propagation(X, W1, b1, W2, b2, W3, b3):  # Aplica a função de ativação
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = tanh(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def perda(A3, Y):  # Função de perda
    m = Y.size
    A3 = A3.reshape(m)
    Y = Y.reshape(m)
    e = A3 - Y
    custo = np.dot(e, e) / m
    return custo


def back_propagation(X, Y, A1, A2, A3, W2, W3):
    m = X.shape[0]

    dA3 = (A3 - Y.reshape(-1, 1))
    dZ3 = dA3 * tanh_derivada(A3)
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * sigmoid_derivada(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivada(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3


def atualizacao_parametros(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, taxa_de_aprendizado):
    W1 -= taxa_de_aprendizado * dW1
    b1 -= taxa_de_aprendizado * db1
    W2 -= taxa_de_aprendizado * dW2
    b2 -= taxa_de_aprendizado * db2
    W3 -= taxa_de_aprendizado * dW3
    b3 -= taxa_de_aprendizado * db3
    return W1, b1, W2, b2, W3, b3

# Função para embaralhar X e Y a cada iteração


def embaralhar_dados(X, Y):
    indices = np.random.permutation(X.shape[0])
    return X[indices], Y[indices]

# Função para treinamento completo da rede


def treinamento(X_train, Y_train, X_val, Y_val, camada_escondida1, camada_escondida2, iterations, taxa_de_aprendizado):
    dimensao_entrada = X_train.shape[1]
    camada_saida = 1

    W1, b1, W2, b2, W3, b3 = inicializando_parametros(
        dimensao_entrada, camada_escondida1, camada_escondida2, camada_saida)

    # Variáveis para plotar gráfico
    train_costs = []
    val_costs = []

    # For para chamar funções
    for i in range(iterations):
        X_train, Y_train = embaralhar_dados(X_train, Y_train)

        _, A1, _, A2, _, A3 = forward_propagation(
            X_train, W1, b1, W2, b2, W3, b3)
        custo = perda(A3, Y_train)

        dW1, db1, dW2, db2, dW3, db3 = back_propagation(
            X_train, Y_train, A1, A2, A3, W2, W3)

        W1, b1, W2, b2, W3, b3 = atualizacao_parametros(
            W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, taxa_de_aprendizado)

        if i % 1000 == 0:
            val_cost = perda(forward_propagation(
                # Pega a saida A3 "o elemento 5" e o Y_val e calcula o erro
                X_val, W1, b1, W2, b2, W3, b3)[5], Y_val)
            train_costs.append(custo)
            val_costs.append(val_cost)
            print(f"Iteração {i}: Custo de Treinamento {
                  custo}, Custo de Validação {val_cost}")

    # Código para plotar gráfico
    plt.plot(train_costs, label='Custo de Treinamento')
    plt.plot(val_costs, label='Custo de Validação')
    plt.xlabel('Iterações (x1000)')
    plt.ylabel('Custo')
    plt.legend()
    plt.show()
    return W1, b1, W2, b2, W3, b3

# Essa função vai transformar a minha ativação da última camada A3 em previsões binarias


def predicao(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
    return (A3 > 0.5).astype(int)


# Chamando função de treinamento
W1, b1, W2, b2, W3, b3 = treinamento(X_train, y_train, X_val, y_val, camada_escondida1=10,
                                     camada_escondida2=6, iterations=15001, taxa_de_aprendizado=0.28)


# Código para mostar a precisão no console
predicoes = predicao(X_test, W1, b1, W2, b2, W3, b3)
precisao = np.mean(predicoes == y_test.reshape(-1, 1))
print(f"Precisão no conjunto de teste: {precisao * 100:.2f}%")


# Código para plotar matriz de confusão
cm = confusion_matrix(y_test, predicoes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão no Conjunto de Teste')
plt.show()
