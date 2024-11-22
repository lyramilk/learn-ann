package com.lyramilk.ann;

import java.util.ArrayList;
import java.util.List;

public class ANN implements java.io.Serializable {
    public final List<Layer> layers = new ArrayList<Layer>();
    private int inputCount;


    public ANN() {
    }

    public void setInputCount(int inputCount) {
        this.inputCount = inputCount;
    }

    /**
     * @param neuronCount 表示神经元数量
     */
    public void addLayer(int neuronCount) {
        if (layers.isEmpty()) {
            // 第一层，参数数量为输入参数数量
            layers.add(new Layer(neuronCount, inputCount));
        } else {
            // 其他层，参数数量为上一层神经元数量
            layers.add(new Layer(neuronCount, layers.get(layers.size() - 1).neurons.length));
        }
    }

    /**
     * @param item 表示输入参数
     * @param rate 表示学习速率
     */
    public void train(Item item, double rate) {
        // 把输入参数格式化成double数组
        // 计算输出并缓存结果
        List<double[]> outputs = calcAndCache(item.input);

        // 计算误差，单独处理输出层。
        double[] errors = new double[layers.get(layers.size() - 1).neurons.length];
        for (int i = 0; i < item.output.length; i++) {
            double dest = item.output[i];
            double real = outputs.get(outputs.size() - 1)[i];
            errors[i] = dest - real;
            // TODO需要处理一下偏置神经元
        }
        System.out.println("输出层第0个输出误差：" + errors[0] + " 输出层第0个输出实际值：" + item.output[0]);

        // 反向传播，误差从输出层向前传播到每一层
        for (int i = layers.size() - 1; i > 0; i--) {
            // 每次循环实际上是处理两层，调整当前层的误差，并计算前一层的误差。
            Layer layer = layers.get(i);
            Layer prevLayer = layers.get(i - 1);

            // 输出比输入多了一层。所以outputs.get(i) 对应layer.get(i-1)
            layer.backpropagation(outputs.get(i), errors, rate);

            double[] delta = new double[prevLayer.neurons.length];
            for (int j = 0; j < prevLayer.neurons.length; j++) {
                double curerr = 0;
                // 要计算第j个误差，先把每个神经元的第j个误差加起来
                for (int k = 0; k < layer.neurons.length; k++) {
                    // j是前一层的神经元，k是当前层的神经元
                    curerr += layer.neurons[k].weights[j] * errors[k];
                }
                delta[j] =  - curerr / layer.neurons.length;
            }

            errors = delta;
        }

        // 处理第一层
        Layer layer = layers.get(0);
        layer.backpropagation(item.input, errors, rate);
    }

    public double[] calc(double[] inputs) {
        double[] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.calc(outputs);
        }
        return outputs;
    }

    private List<double[]> calcAndCache(double[] inputs) {
        List<double[]> result = new ArrayList<double[]>();
        // 在这里输出层比输入层多一层，因为它第一层是输入，而输入在对象中并没有用层对象表示。
        double[] outputs = inputs;
        result.add(outputs);
        for (Layer layer : layers) {
            outputs = layer.calc(outputs);
            System.out.println("outputs: " + outputs.length + " out0: " + outputs[0]);
            result.add(outputs);
        }
        return result;
    }

}
