package com.lyramilk.ann;
import com.lyramilk.ann.lossfunction.MSE;

public interface ILossFunction {
    double loss(Vector predictions, Vector outputs);

    Vector gradient(Vector predictions, Vector outputs);

    ILossFunction MSE = new MSE();
}
