package com.lyramilk.ann;

public interface ILossFunction {
    double loss(Vector predictions, Vector outputs);

    Vector gradient(Vector predictions, Vector outputs);
}
