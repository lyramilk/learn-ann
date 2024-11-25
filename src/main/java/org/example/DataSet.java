package org.example;


import com.google.gson.Gson;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.lyramilk.ann.Data;

import java.util.*;

public class DataSet {
    private final List<InputData> data = new ArrayList<>();
    private static final JiebaSegmenter segmenter = new JiebaSegmenter();
    private final String mask = "痛,高,疼,后,年,愈,基,多,差,减,慢,好,男,女,疗效,性别,信息,病例,年龄,治疗,巩固";

    static DataSet fromJson(String json) {
        Gson gson = new Gson();
        List<Map> parseArray = (List<Map>)gson.fromJson(json, List.class);

        Map<String,Integer> wordFrequency = new HashMap<>();

        DataSet dataSet = new DataSet();
        for (Map map : parseArray) {
            InputData inputData = new InputData();
            String input = (String) map.get("input");
            String output = (String) map.get("output");

            String[] outputs = output.split(" ");

            inputData.input = new ArrayList<>();

            List<String> segs = segmenter.sentenceProcess(input);
            for(String seg : segs) {
                boolean isAllIdeographic = true;
                for(int i = 0; i < seg.length(); i++) {
                    if(!Character.isIdeographic(seg.charAt(0))){
                        isAllIdeographic = false;
                    }
                }
                if(isAllIdeographic){
                    if(wordFrequency.containsKey(seg)) {
                        wordFrequency.put(seg, wordFrequency.get(seg) + 1);
                    }else{
                        wordFrequency.put(seg, 1);
                    }
                    inputData.input.add(seg);
                }
            }

            inputData.output = Arrays.asList(outputs);
            dataSet.data.add(inputData);
        }
        return dataSet;
    }

    public static List<String> segment(String text) {
        return segmenter.sentenceProcess(text);
    }

    public List<Data> toData(int bound) {
        Set<String> wordMask = new HashSet<>(Arrays.asList(mask.split(",")));

        int inputCount = 0;
        List<Data> dataList = new ArrayList<>();
        for (InputData inputData : data) {
            if(inputCount++ > bound)  break;
            Data data = new Data();
            data.inputs = new HashMap<>();
            data.predictions = new HashMap<>();
            for (String word : inputData.input) {
                // 清洗关键词，去掉一些高频无效词
                if(wordMask.contains(word)) continue;
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
