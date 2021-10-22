#!/bin/sh

echo "solve Phase Retrieval Problem"
cd PhaseRetrieval
./run_PhaseRetrieval.sh
cd ..

echo "train Neural Network LeNet5 on MNIST"
cd NeuralNetwork_lenet_mnist
./run_lenet_mnist.sh
cd ..

echo "train Neural Network AllCNN on Cifar10"
cd NeuralNetwork_allcnn_cifar10
./run_allcnn_cifar10.sh
cd ..

echo "solve Sparse Bilinear Logistic Regression on MNIST"
cd SparseBilinearLogisticRegression_mnist
./run_SparseBilinearLogisticRegression_mnist.sh
cd ..



