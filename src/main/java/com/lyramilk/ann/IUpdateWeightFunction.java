package com.lyramilk.ann;

import com.lyramilk.ann.updatefunction.Adam;
import com.lyramilk.ann.updatefunction.SGD;

import java.io.Serializable;

public interface IUpdateWeightFunction extends Serializable {

    void updateWeight(Layer layer, Neuron neuron, Vector gradient, double rate, int t);

    // 计算梯度
    void update(Layer layer, Neuron neuron, int i, double gradient, double rate, int t);

    IUpdateWeightFunction ADAM = new Adam();
    IUpdateWeightFunction SGD = new SGD();

}
