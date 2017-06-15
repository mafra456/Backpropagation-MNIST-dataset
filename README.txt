***Testado em Ubuntu 16.04**********************************************
************************************************************************
Dependências: 
Biblioteca Numpy
É necessário ter o arquivo data_tp1.txt na mesma pasta do arquivo tp.py.
************************************************************************
Para executar o programa, digite no terminal:

python tp.py

Dentro do código, para escolher qual algoritmo de minimização utilizar, 
deve se comentar aqueles que não serão usados, exemplo:

gradient_descent(m, X, y,_y, w1, w2, m) #Executa o gradient_descent
#stochastic_gradient_descent(m, X, y,_y, w1, w2, 1)
#mini_batch(m, X, y,_y, w1, w2, 50)

O último parâmetro em cada função refere-se ao tamanho do batch,
no caso do gradient descent, será sempre 5000, do stochastic 1 e mini-batch 10 ou 50.
