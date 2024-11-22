package org.example;


import com.google.gson.Gson;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.lyramilk.ann.Data;

import java.util.*;

public class DataSet {
    private List<InputData> data = new ArrayList<>();

    static DataSet fromJson(String json) {
        Gson gson = new Gson();
        //List<Map> parseArray = JSON.parseArray(json, Map.class);
        List<Map> parseArray = (List<Map>)gson.fromJson(json, List.class);

        JiebaSegmenter segmenter = new JiebaSegmenter();

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

    public List<Data> toData() {
        List<Data> dataList = new ArrayList<>();
        int datacount = 0;
        for (InputData inputData : data) {
            if(datacount++ > 0) break;
            Data data = new Data();
            data.inputs = new HashMap<>();
            data.outputs = new HashMap<>();
            for (String word : inputData.input) {
                data.inputs.put(word, 30.0);
                break;
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
                    break;
                }
            }
            dataList.add(data);
            dataList.add(data);
        }
        return dataList;
    }


}
