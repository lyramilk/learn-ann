package org.example;

import com.lyramilk.ann.*;
import com.lyramilk.ann.activationfunction.Identify;
import com.lyramilk.ann.activationfunction.Relu;
import com.lyramilk.ann.bp.ANNWrapper;
import com.lyramilk.ann.bp.BP;
import com.lyramilk.ann.lossfunction.MSE;
import com.lyramilk.ann.updatefunction.Adam;
import com.lyramilk.ann.updatefunction.SGD;

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

    public void trainPrescription(int samplesCount,int epoch)
    {
        System.out.println("正在训练模型...");

        ann.addLayer(1000);

        List<Data> dataList = dataSet.toData(samplesCount);

        List<Data> trainDataList = new ArrayList<>();
        Set<String> blacklist = new HashSet<>();
        blacklist.add("男");
        blacklist.add("年龄");
        blacklist.add("性别");
        blacklist.add("个");
        blacklist.add("病例");

        for(int i=0;i<samplesCount;++i){
            trainDataList.add(dataList.get(i));
        }


        ann.train(dataList, 0.01,epoch);
        System.out.println("模型训练完成");
    }

    public void saveModel(String path) throws IOException {
        File file = new File(path);
        try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
            fileOutputStream.write(ann.toJSON().getBytes(StandardCharsets.UTF_8));
        }
        System.out.println("模型已保存到" + path);
    }

    public void loadModel(String path)
    {
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
        Map<String,Double> reuslt = ann.predict(DataSet.segment(input));

        // 按预测值从大到小排序
        // reuslt = reuslt.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
        Stream<Map.Entry<String,Double>> obj =  reuslt.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed());
        List<Map.Entry<String,Double>> output =  new ArrayList<>();
        obj.forEach(output::add);
        for (Map.Entry<String,Double> entry : output) {
            if (entry.getValue() > 0) {
                System.out.println(entry.getKey() + " : " + entry.getValue());
            }
        }
    }

    BP bpmultiply = new BP();
    public void trainmultiply(int samplesCount,int epoch)
    {
        System.out.println("正在训练模型...");

        bpmultiply.addLayer(64, IActivationFunction.RELU);
        bpmultiply.addLayer(2, IActivationFunction.RELU);
        bpmultiply.init(2);

        for(int t=0;t<epoch;++t){
            double loss = 0;
            for(int i=0;i<samplesCount;++i){
                int x = (int)(Math.random()*100);
                int y = (int)(Math.random()*100);
                double[] input = {x, y};
                double[] output = {x * y, x + y};

                Item item = new Item();
                item.inputs = input;
                item.predictions = output;
                loss = bpmultiply.train(item, 0.01, IUpdateWeightFunction.ADAM,ILossFunction.MSE);
            }
            System.out.println("第" + t + "轮训练，loss=" + loss);
        }

        System.out.println("模型训练完成");
    }





    public static void main(String[] args) throws IOException {
        Main main = new Main();

        main.loadDataSet();
        main.trainPrescription(100,10000000);

        main.saveModel("E:\\model.json");
        //main.loadModel("E:\\model.json");
        main.predictPrescription("高血压");

    }
}