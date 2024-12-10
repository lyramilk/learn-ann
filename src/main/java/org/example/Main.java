package org.example;

import com.lyramilk.ann.*;
import com.lyramilk.ann.bp.ANNWrapper;
import com.lyramilk.ann.bp.BP;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Stream;

public class Main {
    private DataSet dataSet;

    public void loadDataSet() throws IOException {
        System.out.print("正在处理数据集...");
        ClassLoader classLoader = Main.class.getClassLoader();
        InputStream inputStream = classLoader.getResourceAsStream("SFT_structPrescription_92896.json");

        if (inputStream == null) {
            throw new FileNotFoundException("File not found");
        }

        // String jsonDataSet = new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
        StringBuilder jsonDataSet = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                jsonDataSet.append(line);
            }
        }
        dataSet = DataSet.fromJson(jsonDataSet.toString());
        System.out.println("数据集处理完成");
    }

    ANNWrapper ann = new ANNWrapper();

    public void trainPrescription(int samplesCount, int epoch, double rate) {
        System.out.println("正在训练模型...");



        List<Data> dataList = dataSet.toData(samplesCount);

        List<Data> trainDataList = new ArrayList<>();
        Set<String> blacklist = new HashSet<>();
        blacklist.add("男");
        blacklist.add("年龄");
        blacklist.add("性别");
        blacklist.add("个");
        blacklist.add("病例");

        for (int i = 0; i < samplesCount; ++i) {
            trainDataList.add(dataList.get(i));
        }

        ann.addLayer(1000);
        ann.train(dataList, rate, epoch);
        System.out.println("模型训练完成");
    }

    public void saveModel(String path) throws IOException {
        File file = new File(path);
        try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
            fileOutputStream.write(ann.toJSON().getBytes(StandardCharsets.UTF_8));
        }
        System.out.println("模型已保存到" + path);
    }

    public void loadModel(String path) {
        File file = new File(path);
        try (FileInputStream fileInputStream = new FileInputStream(file)) {
            StringBuilder jsonModel = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fileInputStream, StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    jsonModel.append(line);
                }
            }
            ann = ANNWrapper.loadJSON(jsonModel.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("模型已加载");
    }

    public void predictPrescription(String input) {
        Map<String, Double> reuslt = ann.predict(DataSet.segment(input));

        // 按预测值从大到小排序
        // reuslt = reuslt.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
        Stream<Map.Entry<String, Double>> obj = reuslt.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed());
        List<Map.Entry<String, Double>> output = new ArrayList<>();
        obj.forEach(output::add);
        for (Map.Entry<String, Double> entry : output) {
            if (entry.getValue() > 0) {
                System.out.println(entry.getKey() + " : " + entry.getValue());
            }
        }
    }

    BP bpmultiply = new BP();

    public void trainmultiply(int samplesCount, int epoch,double rate) {
        System.out.println("正在训练模型...");

        bpmultiply.addLayer(10, IActivationFunction.RELU);
        bpmultiply.addLayer(10, IActivationFunction.RELU);
        bpmultiply.addLayer(4, IActivationFunction.RELU);
        bpmultiply.init(2);

        List<double[]> samples = new ArrayList<>();
        for (int i = 0; i < samplesCount; ++i) {
            double x = (int) (Math.random() * 1000);
            double y = (int) (Math.random() * 1000);
            if (y < 1.0f) {
                y = 1.0f;
            }
            if (x < 1.0f) {
                x = 1.0f;
            }
            double[] input = {x, y};
            samples.add(input);
        }

        for (int t = 0; t < epoch; ++t) {
            double loss = 0;
            for (double[] input : samples) {
                double x = input[0];
                double y = input[1];
                double[] output = {x * y, x + y, x / y, 1000 * x + y};

                Item item = new Item();
                item.inputs = input;
                item.predictions = output;
                loss = bpmultiply.train(item, rate, IUpdateWeightFunction.ADAM, ILossFunction.MSE);
            }
            System.out.println("第" + t + "轮训练，loss=" + loss);
        }

        System.out.println("模型训练完成");
    }

    public void predictmultiply(double x, double y) {
        double[] input = {x, y};
        double[] output = bpmultiply.calc(input);
        System.out.print("预测结果：");
        System.out.println("x=" + x + ",y=" + y);
        System.out.println("x*y预测值" + output[0] + "，实际值" + x * y + "，误差比率" + (output[0] / (x * y)));
        System.out.println("x+y预测值" + output[1] + "，实际值" + (x + y) + "，误差比率" + (output[1] / (x + y)));
        System.out.println("x/y预测值" + output[2] + "，实际值" + (x / y) + "，误差比率" + (output[2] / (x / y)));
        System.out.println("1000*x+y预测值" + output[3] + "，实际值" + (1000 * x + y) + "，误差比率" + (output[3] / (1000 * x + y)));
    }


    public static void main(String[] args) throws IOException {
        Main main = new Main();

/*
        main.loadDataSet();
        main.trainPrescription(20,100,0.001);
        main.saveModel("E:\\model.json");
        main.loadModel("E:\\model.json");
        main.predictPrescription("高血压");
*/


        main.trainmultiply(100, 2000,0.001);
        main.predictmultiply(12000, 3);

    }
}