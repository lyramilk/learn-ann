package com.lyramilk.ann.bp;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.lyramilk.ann.*;
import com.lyramilk.ann.lossfunction.MSE;
import com.lyramilk.ann.updatefunction.Adam;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ANNWrapper implements java.io.Serializable {
    private final BP bp = new BP();
    private final List<Integer> layers = new ArrayList<>();
    private final Map<String, Integer> inputMapping = new HashMap<>();
    private final Map<String, Integer> outputMapping = new HashMap<>();
    private int tokenCount = -1;

    public ANNWrapper() {

    }

    public static ANNWrapper loadJSON(String json) {
        Gson gson = new GsonBuilder().create();
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

    public void addLayer(int neuronCount) {
        layers.add(neuronCount);
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

    public void train(List<Data> dataList, double rate,int epoch) {
        for (Data data : dataList) {
            for (Map.Entry<String, Double> entry : data.inputs.entrySet()) {
                String word = entry.getKey();
                if (!inputMapping.containsKey(word)) {
                    inputMapping.put(word, inputMapping.size());
                }
            }

            for (Map.Entry<String, Double> entry : data.predictions.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();
                if (!outputMapping.containsKey(word)) {
                    outputMapping.put(word, outputMapping.size());
                }
            }
        }


        List<Item> trainData = new ArrayList<>();
        for (Data data : dataList) {
            Item item = new Item();
            item.inputs = new double[inputMapping.size()];
            item.predictions = new double[outputMapping.size()];
            for (Map.Entry<String, Double> entry : data.inputs.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();

                item.inputs[inputMapping.get(word)] = value;
            }
            for (Map.Entry<String, Double> entry : data.predictions.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();
                item.predictions[outputMapping.get(word)] = value;
            }
            trainData.add(item);
        }

        // 添加隐藏层
        for (int neuronCount : layers) {
            bp.addLayer(neuronCount, IActivationFunction.RELU);
        }
        // 添加输出层
        bp.addLayer(outputMapping.size(), IActivationFunction.RELU);
        bp.init(inputMapping.size());


        System.out.println("即将提交训练共有" + inputMapping.size() + "个参数和" + outputMapping.size() + "个输出");

        for(int q=0;q<epoch;++q) {
            for (int i = 0; i < trainData.size(); ++i) {
                double loss = bp.train(trainData.get(i), rate, IUpdateWeightFunction.ADAM, ILossFunction.MSE);
                System.out.println("第" + q + "轮，第" + i + "次训练，loss=" + loss);
                if(loss < 0.001){
                    System.out.println("loos小于0.001，停止训练");
                    return;
                }

            }
        }
    }

    public void train(List<Data> dataList, int tokenCount, double rate) {
        if (tokenCount <= 0) {
            throw new IllegalArgumentException("tokenCount must be greater than 0");
        }
        for (Data data : dataList) {
            for (Map.Entry<String, Double> entry : data.inputs.entrySet()) {
                String word = entry.getKey();
                if (!inputMapping.containsKey(word)) {
                    inputMapping.put(word, inputMapping.size() % tokenCount);
                }
            }

            for (Map.Entry<String, Double> entry : data.predictions.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();
                if (!outputMapping.containsKey(word)) {
                    outputMapping.put(word, outputMapping.size());
                }
            }
        }


        List<Item> trainData = new ArrayList<>();
        for (Data data : dataList) {
            Item item = new Item();
            item.inputs = new double[tokenCount];
            item.predictions = new double[outputMapping.size()];
            for (Map.Entry<String, Double> entry : data.inputs.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();

                // 如果tokenCount大于0，则值所有参数符合这个token的参数值的和
                item.inputs[inputMapping.get(word) % tokenCount] += value;
            }
            for (Map.Entry<String, Double> entry : data.predictions.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();
                item.predictions[outputMapping.get(word)] = value;
            }
            trainData.add(item);
        }

        // 设置参数数目如果tokenCount大于0，表示使用tokenCount作为输入参数数目
        this.tokenCount = tokenCount;

        // 添加隐藏层
        for (int neuronCount : layers) {
            bp.addLayer(neuronCount, IActivationFunction.RELU);
        }
        // 添加输出层
        bp.addLayer(outputMapping.size(),IActivationFunction.IDENTIFY);
        bp.init(inputMapping.size() % tokenCount);

        System.out.println("即将提交训练共有" + inputMapping.size() + "个参数和" + outputMapping.size() + "个输出");
        for (int i = 0; i < bp.layers.size(); ++i) {
            System.out.println("第" + i + "层神经元数量" + bp.layers.get(i).neurons.length);
        }


        for (int i = 0; i < trainData.size(); ++i) {
            double loss = bp.train(trainData.get(i), rate, IUpdateWeightFunction.ADAM, ILossFunction.MSE);
            System.out.println("第" + i + "次训练，loss=" + loss);
        }
    }

    public Map<String, Double> calc(Map<String, Double> inputs) {
        double[] input;
        if (tokenCount > 0) {
            input = new double[tokenCount];
        } else {
            input = new double[inputMapping.size()];
        }
        List<String> matchedWords = new ArrayList<>();
        for (Map.Entry<String, Double> entry : inputs.entrySet()) {
            String word = entry.getKey();
            double value = entry.getValue();
            matchedWords.add(word);
            if (tokenCount > 0) {
                input[inputMapping.get(word) % tokenCount] += value;
            } else {
                input[inputMapping.get(word)] = value;
            }
        }

        System.out.println("预测中，匹配到词组:" + String.join(",", matchedWords));
        double[] output = bp.calc(input);
        Map<String, Double> result = new HashMap<>();
        for (Map.Entry<String, Integer> entry : outputMapping.entrySet()) {
            String word = entry.getKey();
            int index = entry.getValue();
            result.put(word, output[index]);
        }
        return result;
    }

    public Map<String, Double> calc(Data data) {
        return calc(data.inputs);
    }

    public Map<String, Double> predict(List<String> inputs) {
        Map<String, Double> result = new HashMap<>();
        for (String word : inputs) {
            if (!inputMapping.containsKey(word)) {
                continue;
            }
            if (!result.containsKey(word)) {
                result.put(word, 1.0);
            } else {
                result.put(word, result.get(word) + 1);
            }
        }

        return calc(result);
    }
}
