package com.lyramilk.ann;

import com.lyramilk.ann.activationfunction.Identify;

public class Layer implements java.io.Serializable {
    public String id;
    public Neuron[] neurons;
    private transient IActivationFunction activationFunction;

    public Layer(int neuronCount, IActivationFunction activationFunction) {
        this.id = String.valueOf(id);
        this.activationFunction = activationFunction;
        neurons = new Neuron[neuronCount];
    }

    public boolean init(int inputCount) {
        for (int i = 0; i < neurons.length; i++) {
            Neuron neuron = new Neuron(inputCount);
            neuron.id = id + "_" + i;

            for (int j = 0; j < neuron.weights.length; j++) {
                neuron.weights[j] = Math.random();
            }
            neuron.bias = Math.random();

            neurons[i] = neuron;
        }
        return true;
    }

    public Vector forward(Vector inputs) {
        if (this.activationFunction == null) {
            this.activationFunction = new Identify();
        }
        Vector outputs = new Vector(neurons.length);
        for (int i = 0; i < neurons.length; i++) {
            Neuron neuron = neurons[i];
            outputs.data[i] = activationFunction.activate(neuron.dot(inputs));
        }
        return outputs;
    }

}
