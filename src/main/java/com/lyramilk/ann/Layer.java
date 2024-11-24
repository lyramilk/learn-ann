package com.lyramilk.ann;

import com.lyramilk.ann.activationfunction.Identify;

public class Layer implements java.io.Serializable {
    public String id;
    public Neuron[] neurons;
    public int inputCount;

    protected transient IActivationFunction activationFunction;

    public Layer(int id,int neuronCount, int inputCount, IActivationFunction activationFunction) {
        this.id = String.valueOf(id);
        this.activationFunction = activationFunction;
        if (this.activationFunction == null) {
            this.activationFunction = Identify.Instance;
        }

        neurons = new Neuron[neuronCount];
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron(inputCount);
            neurons[i].id = id + "_" + i;
        }
        this.inputCount = inputCount;
    }

    public IActivationFunction getActivationFunction() {
        return activationFunction;
    }

    // 允许重新设置激活函数
    public void setActivationFunction(IActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public Vector forward(Vector inputs) {
        Vector outputs = new Vector(neurons.length);
        for (int i = 0; i < neurons.length; i++) {
            Neuron neuron = neurons[i];
            outputs.data[i] = activationFunction.activate(neuron.dot(inputs));
        }
        return outputs;
    }

}
