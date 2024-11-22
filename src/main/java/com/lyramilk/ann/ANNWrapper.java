package com.lyramilk.ann;

import com.google.gson.Gson;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ANNWrapper implements java.io.Serializable {
    private final ANN ann = new ANN();
    private List<Integer> layers = new ArrayList<>();
    private final Map<String, Integer> inputMapping = new HashMap<>();
    private final Map<String, Integer> outputMapping = new HashMap<>();
    private int tokenCount = -1;

    public ANNWrapper() {

    }


    public void addLayer(int neuronCount) {
        layers.add(neuronCount);
    }

    public static ANNWrapper loadJSON(String json) {
        Gson gson = new Gson();
        return gson.fromJson(json, ANNWrapper.class);
    }

    public String toJSON() {
        Gson gson = new Gson();
        return gson.toJson(this);
    }

    public static ANNWrapper loadBin(byte[] bytes) {
        try {
            return (ANNWrapper) new ObjectInputStream(new ByteArrayInputStream(bytes)).readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }


    public byte[] toBin() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(this);
        oos.close();
        return baos.toByteArray();
    }

    public void train(List<Data> dataList, double rate) {
        for (Data data : dataList) {
            for (Map.Entry<String, Double> entry : data.inputs.entrySet()) {
                String word = entry.getKey();
                if (!inputMapping.containsKey(word)) {

                    inputMapping.put(word, inputMapping.size());

                }
            }

            for (Map.Entry<String, Double> entry : data.outputs.entrySet()) {
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
            item.input = new double[inputMapping.size()];
            item.output = new double[outputMapping.size()];
            for (Map.Entry<String, Double> entry : data.inputs.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();

                item.input[inputMapping.get(word)] = value;
            }
            for (Map.Entry<String, Double> entry : data.outputs.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();
                item.output[outputMapping.get(word)] = value;
            }
            trainData.add(item);
        }

        // 设置参数数目
        ann.setInputCount(inputMapping.size());

        // 添加隐藏层
        for (int neuronCount : layers) {
            ann.addLayer(neuronCount);
        }
        // 添加输出层
        ann.addLayer(outputMapping.size());

        for (Item item : trainData) {
            ann.train(item, rate);
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

            for (Map.Entry<String, Double> entry : data.outputs.entrySet()) {
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
            item.input = new double[tokenCount];
            item.output = new double[outputMapping.size()];
            for (Map.Entry<String, Double> entry : data.inputs.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();

                // 如果tokenCount大于0，则值所有参数符合这个token的参数值的和
                item.input[inputMapping.get(word) % tokenCount] += value;
            }
            for (Map.Entry<String, Double> entry : data.outputs.entrySet()) {
                String word = entry.getKey();
                double value = entry.getValue();
                item.output[outputMapping.get(word)] = value;
            }
            trainData.add(item);
        }

        // 设置参数数目如果tokenCount大于0，表示使用tokenCount作为输入参数数目
        ann.setInputCount(tokenCount);
        this.tokenCount = tokenCount;

        // 添加隐藏层
        for (int neuronCount : layers) {
            ann.addLayer(neuronCount);
        }
        // 添加输出层
        ann.addLayer(outputMapping.size());
System.out.println("即将提交训练共有" + inputMapping.size() + "个参数和" + outputMapping.size() + "个输出");
for(int i=0;i<ann.layers.size();++i){
    System.out.println("第" + i + "层神经元数量" + ann.layers.get(i).neurons.length);
}


        for(int i=0;i<trainData.size();++i){
            System.out.println("第" + i + "次训练");
            ann.train(trainData.get(i), rate);
        }
    }

    public Map<String, Double> calc(Map<String, Double> inputs) {
        double[] input = new double[inputMapping.size()];
        for (Map.Entry<String, Double> entry : inputs.entrySet()) {
            String word = entry.getKey();
            double value = entry.getValue();
            input[inputMapping.get(word)] = value;
        }

        double[] output = ann.calc(input);
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
}
