#!/usr/bin/env python
# coding: utf-8

# # 1. Loading CSV to Hive using Spark

# In[1]:


# import os
# from cmlbootstrap import CMLBootstrap
from pyspark.sql import SparkSession
# Change to the appropriate Datalake directory location

# DATALAKE_DIRECTORY = os.environ["STORAGE"]
DATALAKE_DIRECTORY = "s3a://go01-demo"

spark = (
  SparkSession.builder.appName("MyApp")
  .config("spark.jars", "/opt/spark/optional-lib/iceberg-spark-runtime.jar")
  .config("spark.sql.hive.hwc.execution.mode", "spark")
  .config( "spark.sql.extensions", "com.qubole.spark.hiveacid.HiveAcidAutoConvertExtension, org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
  .config("spark.sql.catalog.spark_catalog.type", "hive")
  .config( "spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
  .config("spark.yarn.access.hadoopFileSystems", DATALAKE_DIRECTORY)
  .getOrCreate()
  )


# In[2]:


datafile=spark.read.csv("/home/cdsw/data/raw/preprocessed_flight_data.csv",header=True, inferSchema=True)


# In[3]:


datafile.show(5)


# In[4]:


spark.sql("drop table if exists airlines.flights_wseol")


# In[5]:


datafile.write.saveAsTable("airlines.flights_wseol")


# In[6]:


spark.sql("select * from airlines.flights_wseol limit 5").show()


# In[ ]:




