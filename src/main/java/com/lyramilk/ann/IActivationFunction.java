package com.lyramilk.ann;

import com.lyramilk.ann.activationfunction.Identify;
import com.lyramilk.ann.activationfunction.Relu;

public interface IActivationFunction {
    double activate(double x);

    double derivative(double x);

    IActivationFunction IDENTIFY = new Identify();
    IActivationFunction RELU = new Relu();
}
