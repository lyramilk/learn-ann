package com.lyramilk.ann.activationfunction;

public class Identify implements com.lyramilk.ann.IActivationFunction {
    public static Identify Instance = new Identify();

    @Override
    public double activate(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1;
    }
}
