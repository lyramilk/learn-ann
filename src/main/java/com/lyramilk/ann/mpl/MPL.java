package com.lyramilk.ann.mpl;

import com.lyramilk.ann.ANN;
import com.lyramilk.ann.IActivationFunction;
import com.lyramilk.ann.Layer;

public class MPL extends ANN<Layer> {
    public void addLayer(int neuronCount, IActivationFunction activationFunction) {
        if (layers.isEmpty()) {
            // 第一层，参数数量为输入参数数量
            layers.add(new Layer(layers.size(),neuronCount, inputCount, activationFunction));
        } else {
            // 其他层，参数数量为上一层神经元数量
            layers.add(new Layer(layers.size(),neuronCount, layers.get(layers.size() - 1).neurons.length + 1, activationFunction));
        }
    }
}
