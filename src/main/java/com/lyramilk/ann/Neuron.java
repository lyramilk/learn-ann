package com.lyramilk.ann;

public class Neuron extends Vector {
    public String id;
    // 只是对Vector中的data进行了重命名
    public double[] weights;

    public double bias;

    public double[] momentum;
    public double[] velocity;
    public double biasMomentum;
    public double biasVelocity;

    public Neuron(int inputCount) {
        super(inputCount);
        weights = super.data;
        for (int i = 0; i < inputCount; i++) {
            data[i] = Math.random();
        }
        momentum = new double[inputCount];
        velocity = new double[inputCount];
    }
}
