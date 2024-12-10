package com.lyramilk.ann.bp;

import com.lyramilk.ann.*;

import java.util.ArrayList;
import java.util.List;

public class BP extends ANN {
    private int curEpochs = 0;

    public BP() {
    }

    public void addLayer(int neuronCount, IActivationFunction activationFunction) {
        if (layers.isEmpty()) {
            // 第一层，参数数量为输入参数数量
            layers.add(new Layer(neuronCount, activationFunction));
        } else {
            // 其他层，参数数量为上一层神经元数量
            layers.add(new Layer(neuronCount, activationFunction));
        }
    }

    public void init(int inputCount) {
        for (Layer layer : layers) {
            layer.init(inputCount);
            inputCount = layer.neurons.length;
        }

    }

    public double train(Item item, double rate, IUpdateWeightFunction updateWeightFunction, ILossFunction lossFunction) {


        Vector inputs = new Vector(item.inputs);
        Vector predictions = new Vector(item.predictions);
        return train(inputs, predictions, rate, updateWeightFunction, lossFunction);
    }

    private double leakyRelu(double x) {
        return x > 0 ? x : 0.01 * x;
    }

    public double train(Vector inputs, Vector predictions, double rate, IUpdateWeightFunction updateWeightFunction, ILossFunction lossFunction) {
        ++curEpochs;

        List<Vector> outputsOfAllLayer = forwardAndCache(inputs);
        Vector outputs = outputsOfAllLayer.get(outputsOfAllLayer.size() - 1);
        double loss = loss(lossFunction, predictions, outputs);
        Vector errors = lossFunction.gradient(predictions, outputs);

        // 更新除了第一层之外的其他层
        for (int i = layers.size() - 1; i > 0; i--) {
            Layer layer = layers.get(i);
            Layer prevLayer = layers.get(i - 1);
            Vector currentInputs = outputsOfAllLayer.get(i - 1);

            Vector[] gradient = new Vector[layer.neurons.length];
            for (int j = 0; j < layer.neurons.length; j++) {
                Neuron neuron = layer.neurons[j];
                gradient[j] = new Vector(neuron.weights.length + 1);
            }

            for (int j = 0; j < layer.neurons.length; j++) {
                Neuron neuron = layer.neurons[j];
                double e = errors.data[j];
                for (int k = 0; k < neuron.weights.length; k++) {
                    double o = currentInputs.data[k];
                    double g = e * o;
                    gradient[j].data[k] += g;
                }
                gradient[j].data[neuron.weights.length] += e;
            }
            for (int j = 0; j < layer.neurons.length; j++) {
                Neuron neuron = layer.neurons[j];
                updateWeightFunction.updateWeight(layer, neuron, gradient[j], rate, curEpochs);
            }

            Vector nextErrors = new Vector(prevLayer.neurons.length);
            for (int j = 0; j < prevLayer.neurons.length; j++) {
                Neuron neuron = prevLayer.neurons[j];
                double sum = 0;
                for (int k = 0; k < layer.neurons.length; k++) {
                    Neuron nextNeuron = layer.neurons[k];
                    sum += nextNeuron.weights[j] * errors.data[k];
                }
                nextErrors.data[j] = sum;
            }
            errors = nextErrors;
        }

        // 更新第一层
        Layer layer = layers.get(0);
        Vector currentInputs = inputs;

        Vector[] gradient = new Vector[layer.neurons.length];
        for (int j = 0; j < layer.neurons.length; j++) {
            Neuron neuron = layer.neurons[j];
            gradient[j] = new Vector(neuron.weights.length + 1);
        }


        for (int j = 0; j < layer.neurons.length; j++) {
            Neuron neuron = layer.neurons[j];
            double e = errors.data[j];
            for (int k = 0; k < neuron.weights.length; k++) {
                double o = currentInputs.data[k];
                double g = e * o;
                gradient[j].data[k] += g;
            }
            gradient[j].data[neuron.weights.length] += e;
        }
        for (int j = 0; j < layer.neurons.length; j++) {
            Neuron neuron = layer.neurons[j];
            updateWeightFunction.updateWeight(layer, neuron, gradient[j], rate, curEpochs);
        }

        return loss;
    }

    public List<Vector> forwardAndCache(Vector inputs) {
        List<Vector> outputs = new ArrayList<>();
        Vector inputsForNextLayer = inputs;
        for (Layer layer : layers) {
            inputsForNextLayer = layer.forward(inputsForNextLayer);
            outputs.add(inputsForNextLayer);
        }
        return outputs;
    }
}
