package org.example;


import com.google.gson.Gson;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.lyramilk.ann.Data;

import java.util.*;

public class DataSet {
    private final List<InputData> data = new ArrayList<>();
    private static final JiebaSegmenter segmenter = new JiebaSegmenter();

    static DataSet fromJson(String json) {
        Gson gson = new Gson();
        List<Map> parseArray = (List<Map>)gson.fromJson(json, List.class);

        DataSet dataSet = new DataSet();
        for (Map map : parseArray) {
            InputData inputData = new InputData();
            String input = (String) map.get("input");
            String output = (String) map.get("output");

            String[] outputs = output.split(" ");

            inputData.input = segmenter.sentenceProcess(input);
            inputData.output = Arrays.asList(outputs);
            dataSet.data.add(inputData);
        }

        return dataSet;
    }

    public static List<String> segment(String text) {
        return segmenter.sentenceProcess(text);
    }

    public List<Data> toData() {
        int inputCount = 0;
        List<Data> dataList = new ArrayList<>();
        for (InputData inputData : data) {
            if(++inputCount > 2)  break;
            Data data = new Data();
            data.inputs = new HashMap<>();
            data.predictions = new HashMap<>();
            for (String word : inputData.input) {
                data.inputs.put(word, 1.0);
            }
            for (String word : inputData.output) {
                String[] split = word.split("(?<=\\D)(?=\\d)");
                if (split.length == 2) {
                    String output = split[0];
                    String outputValue = split[1];
                    // 把outputValue转成double，注意如果包含非数字字符，直接停止转换
                    outputValue = outputValue.replaceAll("[^\\d.]", "");
                    double value = Double.parseDouble(outputValue);
                    data.predictions.put(output, value);
                }
            }
            dataList.add(data);
        }
        return dataList;
    }


}
