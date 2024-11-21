package com.lyramilk.ann;

import com.alibaba.fastjson2.JSON;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ANNWrapper implements java.io.Serializable {
    private ANN ann;
    private List<Integer> layers = new ArrayList<>();
    private final Map<String, Integer> inputMapping = new HashMap<>();
    private final Map<String, Integer> outputMapping = new HashMap<>();
    private int tokenCount = -1;

    public ANNWrapper() {
        ann = new ANN();
    }


    public void addLayer(int neuronCount) {
        layers.add(neuronCount);
    }

    public static ANNWrapper load(String json) {
        return JSON.parseObject(json, ANNWrapper.class);
    }

    public String train(List<Data> dataList, double rate) {
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

        return JSON.toJSONString(this);
    }

    public String train(List<Data> dataList, int tokenCount, double rate) {
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

        // 添加隐藏层
        for (int neuronCount : layers) {
            ann.addLayer(neuronCount);
        }
        // 添加输出层
        ann.addLayer(outputMapping.size());

        for (Item item : trainData) {
            ann.train(item, rate);
        }

        return JSON.toJSONString(this);
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
