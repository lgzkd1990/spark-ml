package ustc.edu.cn.chap1

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by lei on 2016/10/31.
  * 购买次数
  * 客户总个数
  * 总收入
  * 最畅销的产品
  */
object easy {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[3]").setAppName("easy")
    val sc = new SparkContext(conf)
    val data = sc.textFile("data/UserPurchaseHistory.csv").map(line => {
      val field = line.split(",")
      (field(0), field(1), field(2).toDouble)
    })
    val num = data.count()
    println("num="+num)
    val customer = data.map(_._1).distinct().count()
    println("customer:"+customer)
    val totalIncome = data.map(_._3).sum()//reduce(_ + _)
    println("totalIncome="+totalIncome)
    val bestSeller = data.map(x => {
      (x._2, 1)
    }).reduceByKey(_ + _).sortBy(_._2,false)
    println(bestSeller.collect().toBuffer)
  }
}
