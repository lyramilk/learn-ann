package com.lyramilk.ann.updatefunction;

import com.lyramilk.ann.IUpdateWeightFunction;
import com.lyramilk.ann.Neuron;
import com.lyramilk.ann.Layer;
import com.lyramilk.ann.Vector;

public class SGD implements IUpdateWeightFunction {
    @Override
    public void updateWeight(Layer layer, Neuron neuron, Vector gradient, double rate,int t) {
        for (int i = 0; i < neuron.weights.length; i++) {
            neuron.weights[i] -= rate * gradient.data[i];
        }
    }

    @Override
    public void update(Layer layer, Neuron neuron, int i, double gradient, double rate, int t) {
        neuron.weights[i] -= rate * gradient;
    }


}
