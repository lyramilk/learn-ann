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
        double[] result = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            // 直接计算，不要调用neurons[i].calc(inputs);
            double sum = 0;
            for (int j = 0; j < inputs.length; j++) {
                double oldsum = sum;
                sum += inputs[j] * neurons[i].weights[j];
                System.out.println("第" + j + "个神经元，sum 从 " + oldsum + " 变为 " + sum  + " inputs[j] = " + inputs[j] + " neurons[i].weights[j] = " + neurons[i].weights[j] + ",增加了" + inputs[j] * neurons[i].weights[j]);
            }
            result[i] = sum + neurons[i].weights[neurons[i].weights.length - 1];
            System.out.println("偏置神经元的权重：" + neurons[i].weights[neurons[i].weights.length - 1]);
        }
        return result;
    }

    public void backpropagation(double[] inputs, double[] errors, double rate) {
        for (int i = 0; i < neurons.length; i++) {
            Neuron neuron = neurons[i];
            double error = errors[i];
            for (int j = 0; j < neuron.weights.length; j++) {
                double old = neuron.weights[j];
                neuron.weights[j] -= rate * error * inputs[j];
                System.out.println("weights[i] 从 " + old + " 变为 " + neuron.weights[j] + " rate = " + rate + " error = " + error + " inputs[i] = " + inputs[j] + ",测试值" + neuron.weights[j] * inputs[j]);
                if (Double.isInfinite(neuron.weights[j]) || Double.isNaN(neuron.weights[j])) {
                    throw new RuntimeException("weights is infinite");
                }
            }
            neuron.weights[neuron.weights.length - 1] -= rate * error;
        }

    }

}
