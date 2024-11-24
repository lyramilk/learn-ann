package org.example;

import com.lyramilk.ann.ILossFunction;
import com.lyramilk.ann.Item;
import com.lyramilk.ann.activationfunction.Relu;
import com.lyramilk.ann.bp.ANNWrapper;
import com.lyramilk.ann.bp.BP;
import com.lyramilk.ann.lossfunction.MSE;
import com.lyramilk.ann.updatefunction.Adam;
import com.lyramilk.ann.updatefunction.SGD;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Map;

public class Main {

    public static void main(String[] args) throws IOException {
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

        System.out.print("正在处理数据集...");
        DataSet dataSet = DataSet.fromJson(jsonDataSet.toString());

        System.out.println("正在训练模型...");
        /**
        ANNWrapper ann = new ANNWrapper();
        ann.train(dataSet.toData(), 0.01);



        String key = "肾结节";
        Map<String,Double> reuslt = ann.predict(DataSet.segment(key));

        for(Map.Entry<String,Double> entry : reuslt.entrySet()){
            if(entry.getValue() > 0) {
                System.out.println("预测结果：" + entry.getKey() + "：" + entry.getValue());
            }
        }

        /*/
        BP ann = new BP(new Adam());
        ann.setInputCount(2);
        ann.addLayer(64, Relu.Instance);
        ann.addLayer(2, Relu.Instance);



        for(double i = 10;i<300;i+=1){
            for(double j = 30;j<40;j+=1){
                double[] input = {i,j};
                double[] output = {i*j,i+j};

                Item item = new Item();
                item.inputs = input;
                item.predictions = output;
                double loss = ann.train(item,0.001, MSE.Instance);
                System.out.println("第" + i + "次训练，loss=" + loss);
            }
        }


        double[] input = {201,30};
        double[] result = ann.calc(input);
        System.out.println("预测结果：" + result[0] + "," + result[1]);
        /**/


        File file = new File("E:\\model.json");
/*
        try (FileWriter fileWriter = new FileWriter(file)) {
            //fileWriter.write(ann.toJSON());
        }*/

        try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
            fileOutputStream.write(ann.toJSON().getBytes(StandardCharsets.UTF_8));
        }
        System.out.println("模型已保存到E:\\model.json");
        inputStream.close();
        System.out.println("Hello World!");
    }
}