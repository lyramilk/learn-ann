package com.lyramilk.ann;

import java.util.ArrayList;
import java.util.List;

public class ANN implements java.io.Serializable {
    private final List<Layer> layers = new ArrayList<Layer>();
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
        double[] inputs = new double[inputCount];
        for (int i = 0; i < item.input.length; i++) {
            System.out.println("item.input[" + i + "]:" + item.input.length);
            inputs[i] = item.input[i];
        }

        // 计算输出并缓存结果
        List<double[]> outputs = calcAndCache(inputs);
        // 计算误差
        double[] errors = new double[layers.get(layers.size() - 1).neurons.length];
        for (int i = 0; i < item.output.length; i++) {
            // 需要处理一下偏置神经元
            errors[i] = item.output[i] - outputs.get(outputs.size() - 1)[i];
        }

        // 反向传播
        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer layer = layers.get(i);
            double[] inputs1 = i == 0 ? inputs : outputs.get(i - 1);
            double[] errors1 = new double[layer.neurons.length];
            for (int j = 0; j < layer.neurons.length; j++) {
                errors1[j] = 0;
                for (int k = 0; k < errors.length; k++) {
                    errors1[j] += errors[k] * layers.get(i + 1).neurons[k].weights[j];
                }
            }
            layer.backpropagation(inputs1, errors, rate);
            errors = errors1;
        }
    }

    public double[] calc(double[] inputs) {
        double[] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.calc(outputs);
        }
        return outputs;
    }

    private List<double[]> calcAndCache(double[] inputs) {
        List<double[]> outputs = new ArrayList<double[]>();
        double[] outputs1 = inputs;
        outputs.add(outputs1);
        for (Layer layer : layers) {
            outputs1 = layer.calc(outputs1);
            outputs.add(outputs1);
        }
        return outputs;
    }

}
