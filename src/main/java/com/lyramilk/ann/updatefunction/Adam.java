package com.lyramilk.ann.updatefunction;

import com.lyramilk.ann.IUpdateWeightFunction;
import com.lyramilk.ann.Neuron;
import com.lyramilk.ann.Layer;
import com.lyramilk.ann.Vector;

import java.io.ObjectInputStream;

public class Adam implements IUpdateWeightFunction {
    private final static double BETA_1 = 0.9;
    private final static double BETA_2 = 0.999;
    private final static double EPSILON = 1e-8;

    public Adam() {
    }

    @Override
    public void updateWeight(Layer layer,Neuron neuron, Vector gradient, double rate,int t) {
        for (int i = 0; i < neuron.weights.length; i++) {
            double dold = neuron.momentum[i];
            double gradientData = gradient.data[i];
            neuron.momentum[i] = BETA_1 * neuron.momentum[i] + (1 - BETA_1) * gradient.data[i];
            //System.out.println("节点" + neuron.id + "的第" + i + "个权重的动量：" + dold + " -> " + neuron.momentum[i]+ "梯度：" + gradientData);
            neuron.velocity[i] = BETA_2 * neuron.velocity[i] + (1 - BETA_2) * gradient.data[i] * gradient.data[i];
            double mHat = neuron.momentum[i] / (1 - Math.pow(BETA_1, t));
            double vHat = neuron.velocity[i] / (1 - Math.pow(BETA_2, t));
            neuron.weights[i] -= rate * mHat / (Math.sqrt(vHat) + EPSILON);
        }
        //计算偏置权重
        double dold = neuron.biasMomentum;
        neuron.biasMomentum = BETA_1 * neuron.biasMomentum + (1 - BETA_1) * gradient.data[neuron.weights.length];
        //System.out.println("节点" + neuron.id + "的偏置权重的动量：" + dold + " -> " + neuron.biasMomentum + "梯度：" + neuron.weights.length);
        neuron.biasVelocity = BETA_2 * neuron.biasVelocity + (1 - BETA_2) * gradient.data[neuron.weights.length] * gradient.data[neuron.weights.length];
        double mHat = neuron.biasMomentum / (1 - Math.pow(BETA_1, t));
        double vHat = neuron.biasVelocity / (1 - Math.pow(BETA_2, t));
        neuron.bias -= rate * mHat / (Math.sqrt(vHat) + EPSILON);

    }

    @Override
    public void update(Layer layer,Neuron neuron,int i, double gradient, double rate,int t)
    {
            double dold = neuron.momentum[i];
            neuron.momentum[i] = BETA_1 * neuron.momentum[i] + (1 - BETA_1) * gradient;
            System.out.println("节点" + neuron.id + "的第" + i + "个权重的动量：" + dold + " -> " + neuron.momentum[i]+ "梯度：" + gradient);
            neuron.velocity[i] = BETA_2 * neuron.velocity[i] + (1 - BETA_2) * gradient * gradient;
            double mHat = neuron.momentum[i] / (1 - Math.pow(BETA_1, t));
            double vHat = neuron.velocity[i] / (1 - Math.pow(BETA_2, t));
            neuron.weights[i] -= rate * mHat / (Math.sqrt(vHat) + EPSILON);
    }

}



