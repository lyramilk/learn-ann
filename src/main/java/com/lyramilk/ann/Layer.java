package com.lyramilk.ann;

public class Layer implements java.io.Serializable {
    public Neuron[] neurons;

    public Layer(int neuronCount, int inputCount) {
        neurons = new Neuron[neuronCount + 1];
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron(inputCount);
        }
    }

    public double[] calc(double[] inputs) {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].calc(inputs);
        }

        return outputs;
    }

    public void backpropagation(double[] inputs, double[] errors, double rate) {
        System.out.println("Layer.backpropagation neurons:" + neurons.length + " inputs: " + inputs.length + " errors: " + errors.length + " rate: " + rate);
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].backpropagation(inputs, errors[i], rate);
        }
    }

}
