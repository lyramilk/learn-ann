package org.example;

import com.lyramilk.ann.ANNWrapper;

import java.io.*;
import java.nio.charset.StandardCharsets;

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
        ANNWrapper ann = new ANNWrapper();
        ann.addLayer(1);
        ann.train(dataSet.toData(),1, 0.01);

        System.out.println("训练完成");
        File file = new File("E:\\model.json");

        try (FileWriter fileWriter = new FileWriter(file)) {
            fileWriter.write(ann.toJSON());
        }
        /*
        try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
            fileOutputStream.write(ann.toBin());
        }*/
        System.out.println("模型已保存到E:\\model.json");
        inputStream.close();
        System.out.println("Hello World!");
    }
}