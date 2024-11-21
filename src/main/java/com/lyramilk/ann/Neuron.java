package com.lyramilk.ann;

public class Neuron implements java.io.Serializable {
    // 每个输入的权重
    public double[] weights;

    public Neuron(int inputCount) {
        weights = new double[inputCount];
        for (int i = 0; i < inputCount; i++) {
            weights[i] = Math.random();
        }
    }

    public double calc(double[] inputs) {
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum;
    }

    public void backpropagation(double[] inputs, double error, double rate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += rate * error * inputs[i];
        }
    }
}
