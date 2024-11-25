package com.lyramilk.ann.bp;

import com.lyramilk.ann.*;

import java.util.ArrayList;
import java.util.List;

public class BP extends ANN<UpdatableLayer> {
    private transient final IUpdateWeightFunction updateWeightFunction;
    private int curEpochs = 0;

    public BP(IUpdateWeightFunction updateWeightFunction) {
        this.updateWeightFunction = updateWeightFunction;
    }

    public void addLayer(int neuronCount, IActivationFunction activationFunction) {
        if (layers.isEmpty()) {
            // 第一层，参数数量为输入参数数量
            layers.add(new UpdatableLayer(layers.size(),neuronCount, inputCount, activationFunction, updateWeightFunction));
        } else {
            // 其他层，参数数量为上一层神经元数量
            layers.add(new UpdatableLayer(layers.size(),neuronCount, layers.get(layers.size() - 1).neurons.length, activationFunction, updateWeightFunction));
        }
    }

    public double train(Item item, double rate, ILossFunction lossFunction) {


        Vector inputs = new Vector(item.inputs);
        Vector predictions = new Vector(item.predictions);
        return train(inputs, predictions, rate, lossFunction);
    }

    private double leakyRelu(double x) {
        return x > 0 ? x : 0.01 * x;
    }

    public double train(Vector inputs, Vector predictions, double rate, ILossFunction lossFunction) {
        ++curEpochs;
        List<Vector> outputsOfAllLayer = forwardAndCache(inputs);
        Vector outputs = outputsOfAllLayer.get(outputsOfAllLayer.size() - 1);
        double loss = loss(predictions, outputs);
        System.out.println("loss=" + loss + "predictions=" + predictions +  "outputs=" + outputs + "inputs=" + inputs );
        Vector errors = lossFunction.gradient(predictions, outputs);

        // 更新除了第一层之外的其他层
        for (int i = 1; i < layers.size(); i++) {
            UpdatableLayer layer = layers.get(i);
            UpdatableLayer prevLayer = layers.get(i - 1);
            Vector currentOutputs = outputsOfAllLayer.get(i);
            //Vector prevOutputs = outputsOfAllLayer.get(i - 1);

            IActivationFunction af = layer.getActivationFunction();

            Vector[] gradient = new Vector[layer.neurons.length];
            for (int j = 0; j < layer.neurons.length; j++) {
                Neuron neuron = layer.neurons[j];
                gradient[j] = new Vector(neuron.weights.length + 1);
            }

            for (int j = 0; j < layer.neurons.length; j++) {
                Neuron neuron = layer.neurons[j];
                for (int k = 0; k < neuron.weights.length; k++) {
                    double e = errors.data[j];
                    double o = currentOutputs.data[j];
                    double g = e * o;
                    gradient[j].data[k] += g;
                }
            }
            for (int j = 0; j < layer.neurons.length; j++) {
                Neuron neuron = layer.neurons[j];
                updateWeightFunction.updateWeight(layer,neuron, gradient[j], rate,curEpochs);
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
        UpdatableLayer layer = layers.get(0);
        Vector currentOutputs = outputsOfAllLayer.get(0);

        IActivationFunction af = layer.getActivationFunction();

        Vector[] gradient = new Vector[layer.neurons.length];
        for (int j = 0; j < layer.neurons.length; j++) {
            Neuron neuron = layer.neurons[j];
            gradient[j] = new Vector(neuron.weights.length + 1);
        }


        for (int j = 0; j < layer.neurons.length; j++) {
            Neuron neuron = layer.neurons[j];
            for (int k = 0; k < neuron.weights.length; k++) {
                //System.out.println(neuron.id  +  "神经元" + j  + "/" + layer.neurons.length + "权重" + k + "错误数" + errors.size());
                double e = errors.data[j];
                double o = currentOutputs.data[j];
                double g = e * o;
                gradient[j].data[k] += g;
                //gradient[j].data[k] += errors.data[j] * af.derivative(neuron.dot(currentOutputs)) * currentOutputs.data[k];
            }
            gradient[j].data[neuron.weights.length] += errors.data[j];
        }
        for (int j = 0; j < layer.neurons.length; j++) {
            Neuron neuron = layer.neurons[j];
            updateWeightFunction.updateWeight(layer,neuron, gradient[j], rate,curEpochs);
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
