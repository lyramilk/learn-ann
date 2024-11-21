package org.example;

import com.alibaba.fastjson2.JSON;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.lyramilk.ann.Data;

import java.util.*;

public class DataSet {
    private List<InputData> data = new ArrayList<>();

    static DataSet fromJson(String json) {
        List<Map> parseArray = JSON.parseArray(json, Map.class);

        JiebaSegmenter segmenter = new JiebaSegmenter();

        DataSet dataSet = new DataSet();
        for (Map map : parseArray) {
            InputData inputData = new InputData();
            String input = (String) map.get("input");
            String output = (String) map.get("output");

            inputData.input = segmenter.sentenceProcess(input);
            inputData.output = Arrays.stream(output.split(" ")).toList();
            dataSet.data.add(inputData);
        }

        return dataSet;
    }

    public List<Data> toData() {
        List<Data> dataList = new ArrayList<>();
        for (InputData inputData : data) {
            Data data = new Data();
            data.inputs = new HashMap<>();
            data.outputs = new HashMap<>();
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
                    data.outputs.put(output, value);
                }
            }
            dataList.add(data);
        }
        return dataList;
    }


}
