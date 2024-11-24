package com.lyramilk.ann.activationfunction;

public class Relu implements com.lyramilk.ann.IActivationFunction {
    public static Relu Instance = new Relu();

    @Override
    public double activate(double x) {
        return Math.max(0, x);
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }

}