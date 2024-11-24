package com.lyramilk.ann;

public class UpdatableLayer extends Layer {
    IUpdateWeightFunction updateWeightFunction;

    //public double[] momentum;
    //public double[] velocity;
    public UpdatableLayer(int id,int neuronCount, int inputCount, IActivationFunction activationFunction, IUpdateWeightFunction updateWeightFunction) {
        super(id,neuronCount, inputCount, activationFunction);
        this.updateWeightFunction = updateWeightFunction;
        //momentum = new double[inputCount];
        //velocity = new double[inputCount];
    }
}
