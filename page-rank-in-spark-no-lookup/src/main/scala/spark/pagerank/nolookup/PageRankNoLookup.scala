package spark.pagerank.nolookup

import org.apache.logging.log4j.{LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

object PageRankNoLookup {

  def main(args: Array[String]) {
    val logger: Logger = LogManager.getRootLogger

    if (args.length != 3) {
      logger.error("Usage:\nassignment4.part1.PageRankNoLookup <k - count of linear graph chains> <n - number of page rank iterations> <output file>")
      System.exit(1)
    }
    val k = args(0).toInt
    val n = args(1).toInt
    val output = args(2)

    // spark context
    val conf: SparkConf = new SparkConf().setAppName("Page Rank in Spark (without lookup)")
    val sc = new SparkContext(conf)

    // Create synthetic graph
    // this is graph structure represented only by edges (v,u)
    val graphEdges = mutable.MutableList[(Long, Long)]()
    val dummyVertex = 0
    for (vertex <- 1 to k * k) {
      if ((vertex % k) == 0)
        graphEdges += ((vertex, dummyVertex))
      else
        graphEdges += ((vertex, vertex + 1))
    }

    sc.broadcast(k)
    sc.broadcast(dummyVertex)

    // get RDD of graph edges
    val graphEdgesRDD = sc.parallelize(graphEdges)


    // get RDD of graph with adjacency list representation
    // we cache this, because the same graph will be used again and again
    val graph = graphEdgesRDD.distinct().groupByKey().cache()
    // initial page rank of each vertex
    var ranks = graph.mapValues(adjList => 1.0 / (k * k))

    // initial dangling PR mass
//    var danglingPRMass = 0.0

    // calculate page rank  for 'n' iterations
    for (iterCount <- 1 to n) {
      // - join graph with existing ranks to get tuples with adjList and PR for each vertex
      // - for each vertex v, return a tuple of form (m,pr(v)/c(v)),
      //    where c(v) is the count of outlinks of v, and m is an outlink of v
      // - reduceBy key to get aggregate PR for each m
      val tempRanks = graph.join(ranks)
          .flatMap {
            case (v, (adjList, pr)) => adjList.map(dest => (dest, pr / adjList.size)).++(Iterable[(Long, Double)]((v, 0.0)))
          }
          .reduceByKey((x,y) => x + y)

      // filter out the dangling mass
      val danglingRank = tempRanks.filter{
        case (v,pr) => v == dummyVertex
      }.map {
        case (v,dr) => dr / (k*k)
      }

      // distribute
      ranks = tempRanks.cartesian(danglingRank).map {
        case ((v, pr), danglingPR) =>
          if (v == dummyVertex)
            (v, pr)
          else
            (v, pr + danglingPR)
      }.cache()
    }

    println("----------------------------------------------")
    println(ranks.toDebugString)
    println("----------------------------------------------")
//    logger.info("DUMMY NODE (node 0) FINAL PR: " + danglingPRMass)
    ranks.saveAsTextFile(output)

  }

}
