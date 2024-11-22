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
/*
    public double calc(double[] inputs) {
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }

        if(Double.isInfinite(sum) || Double.isNaN(sum)) {
            throw new RuntimeException("sum is infinite");
        }
        return sum + 0.1;
    }
/*
    public void backpropagation(double[] inputs, double error, double rate) {
        for (int i = 0; i < weights.length; i++) {
            // error误差是应有输出和实际输出的差值
            // 实际输出是inputs[i] * weights[i]
            // 所以误差对权重的偏导数是error * inputs[i]
            double old = weights[i];
            weights[i] -= rate * error * inputs[i];

            System.out.println("weights[i] 从 " + old + " 变为 " + weights[i] + " rate = " + rate + " error = " + error + " inputs[i] = " + inputs[i] + ",测试值" + weights[i] * inputs[i]);
            if(Double.isInfinite(weights[i]) || Double.isNaN(weights[i])) {
                throw new RuntimeException("weights is infinite");
            }
        }
    }*/
}
