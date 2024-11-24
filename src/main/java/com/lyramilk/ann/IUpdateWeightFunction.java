package com.lyramilk.ann;

import java.io.Serializable;

public interface IUpdateWeightFunction extends Serializable {

    void updateWeight(UpdatableLayer layer,Neuron neuron, Vector gradient, double rate,int t);
    // 计算梯度
    void update(UpdatableLayer layer,Neuron neuron,int i, double gradient, double rate,int t);
}
