package ustc.edu.cn;

import scala.Tuple2;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.SparkConf;


public class NaiveBayesDemo {
  public static void main(String[] args) {
    SparkConf sparkConf = new SparkConf().setAppName("NaiveBayesDemo").setMaster("local[3]");
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);
    //加载数据
    String base = "C:\\Users\\lgzkd\\IdeaProjects\\Spark-ML\\src\\main\\resources\\";
    String path = base+"NaiveBayesdata.txt";
    JavaRDD<String> lines =jsc.textFile(path);
    JavaRDD<LabeledPoint> parsedData = lines.map(new Function<String, LabeledPoint>() {
      public LabeledPoint call(String s) {
        String[] sarray = s.split(",");
        Double dLabel=Double.valueOf(sarray[0]);
        String[] sFeatures=sarray[1].split(" ");
        double[] values = new double[sFeatures.length];
        for (int i = 0; i < sFeatures.length; i++) {
          values[i] = Double.parseDouble(sFeatures[i]);
        }
        System.out.println(dLabel+":"+Vectors.dense(values));
        LabeledPoint lp=new LabeledPoint(dLabel,Vectors.dense(values));
        return lp;
      }
    }
    );
    parsedData.cache();
    //JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();
    // 把数据的60%作为训练集，40%作为测试集.
    JavaRDD<LabeledPoint>[] tmp = parsedData.randomSplit(new double[]{0.6, 0.4});
    JavaRDD<LabeledPoint> training = tmp[0]; // training set
    JavaRDD<LabeledPoint> test = tmp[1]; // test set
    //获得训练模型,第一个参数为数据，第二个参数为平滑参数，默认为1，可改
    final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
    //对模型进行准确度分析
    JavaPairRDD<Double, Double> predictionAndLabel =
      test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
        @Override
        public Tuple2<Double, Double> call(LabeledPoint p) {
          return new Tuple2<>(model.predict(p.features()), p.label());
        }
      });
    double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
      @Override
      public Boolean call(Tuple2<Double, Double> pl) {
        return pl._1().equals(pl._2());
      }
    }).count() / (double) test.count();
    System.out.println("accuracy-->"+accuracy);
    //保存和加载训练模型
    model.save(jsc.sc(), base+"myNaiveBayesModel");
    NaiveBayesModel sameModel = NaiveBayesModel.load(jsc.sc(), base+"myNaiveBayesModel");
    //对新的事件进行概率预测
    System.out.println("Prediction of (0.0, 2.0, 0.0, 1.0):"+sameModel.predict(Vectors.dense(0.0,2.0,0.0,1.0)));

    jsc.stop();
  }
}
