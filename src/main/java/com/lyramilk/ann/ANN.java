package com.lyramilk.ann;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.lyramilk.ann.bp.ANNWrapper;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class ANN implements java.io.Serializable {
    public final List<Layer> layers = new ArrayList<Layer>();

    public ANN() {
    }


    public static ANNWrapper loadJSON(String json) {
        Gson gson = new Gson();
        return gson.fromJson(json, ANNWrapper.class);
    }

    public static ANNWrapper loadBin(byte[] bytes) {
        try {
            return (ANNWrapper) new ObjectInputStream(new ByteArrayInputStream(bytes)).readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    public String toJSON() {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(this);
    }

    public byte[] toBin() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(this);
        oos.close();
        return baos.toByteArray();
    }

    public double[] calc(double[] inputs) {
        Vector outputs = forward(new Vector(inputs));
        return outputs.data;
    }

    public Vector forward(Vector inputs) {
        Vector inputsForNextLayer = inputs;
        for (Layer layer : layers) {
            inputsForNextLayer = layer.forward(inputsForNextLayer);
        }
        return inputsForNextLayer;
    }

    public double loss(ILossFunction lossFunction, Vector predictions, Vector outputs) {
        return lossFunction.loss(predictions, outputs);
    }
}
