package com.lyramilk.ann.lossfunction;

import com.lyramilk.ann.ILossFunction;
import com.lyramilk.ann.Vector;

public class MSE implements ILossFunction {
    @Override
    public double loss(Vector predictions, Vector outputs) {
        return predictions.copy().sub(outputs).pow(2).sum() / predictions.size();
    }

    @Override
    public Vector gradient(Vector predictions, Vector outputs) {
        return predictions.copy().sub(outputs).mul(-2);
    }
}
