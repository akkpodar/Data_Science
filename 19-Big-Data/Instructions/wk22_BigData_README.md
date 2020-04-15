# Week 22 
## Objectives
* Identify the pieces of the Hadoop ecosystem.
* Identify the differences and similarities between Hadoop and Spark.
* Write MapReduce jobs locally with MRjob.
* Manipulate data using PySpark DataFrames.
* Explain why NLP is necessary in a big data tool kit.
* Apply transformations resulting from NLP data processing to PySpark DataFrames.
* Explain and utilize PySpark text processing methods like tokenization, stop words, n-grams, and term and document frequency.
* Utilize an NLP data processing pipeline to create a spam filter.

### What is the relevance of Big Data?
Our methods of data collection have grown quite a bit in the past few years. We now have a massive amount of data available, but adequately parsing, processing, and analyzing this data is difficult to do using our traditional applications. 2.3 trillion gigabytes of new data is created each day in both structured and unstructured formats. Knowing how to process and analyze this data is a very in-demand skill.

Using Big Data, businesses can gain understanding in where, when, and why their customers buy their product. They can provide targeted advertisement in a more efficient way. They can predict market trends and anticipate future production needs. They can have an edge on the competition or their own future improvement. They can identify new sources of revenue. Big Data can be a game changer to a company.

### How much Big Data knowledge will I walk away with?
Similar to Machine Learning, Big Data is a very deep well of knowledge. We could have an entire course dedicated to understanding how to handle Big Data effectively and still walk away wanting to know more. This course is meant to provide you with an introduction into Big Data and give you the tools and familiarity to continue this education on your own. Take advantage of in-class time, office hours, and your network of peers to continue learning more about Big Data and its real world use cases.


### GOOGLE COLAB
* [GOOGLE COLAB HOW TO](https://medium.com/@sushantgautam_930/apache-spark-in-google-collaboratory-in-3-steps-e0acbba654e6)

### NLP CURRENT PROJECT HELPFUL LINKS - PLUS MORE
* [Primary Article IT Support Ticket Classification](https://towardsdatascience.com/it-support-ticket-classification-and-deployment-using-machine-learning-and-aws-lambda-8ef8b82643b6) #one of primary articles to help with research

* [Automation all the way - Machine Learning for IT Service Management](https://medium.com/datadriveninvestor/automation-all-the-way-machine-learning-for-it-service-management-9de99882a33) # if want to fully automate your classification model

* [Report on Text Classification using CNN, RNN & HAN](https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f) # good article to go further into text classification

* [Understanding AUC - ROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) #supplemental reading

* [Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21) #have note finished reading but another good article

* [VADER](https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f) #Alexander has used library for sentiment analysis


### Other Info

* [Encoding Method Label vs One Hot Encoder](https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b) # I was to lazy to post in ML week


**Helpful Links**
* [Recommended Book: Python Natural Language Processing Techniques](https://www.amazon.com/Python-Natural-Language-Processing-techniques/dp/1787121429)

* [Natural Language Toolkit](https://www.nltk.org/)

* [NLP for Big Data:  What Everyone Should Know](http://www.expertsystem.com/nlp-big-data-everyone-know/)

* [What Is Natural Language Processing?](https://machinelearningmastery.com/natural-language-processing/)

* [7 Applications of Deep Learning for Natural Language Processing](https://machinelearningmastery.com/applications-of-deep-learning-for-natural-language-processing/)

* [The Art of Tokenization](https://www.ibm.com/developerworks/community/blogs/nlp/entry/tokenization?lang=en)

* [What does tf-idf mean?](http://www.tfidf.com/)

* [TD-IDF Explained](https://www.elephate.com/blog/what-is-tf-idf/)

* [TD-IDF in Apache Spark](https://mingchen0919.github.io/learning-apache-spark/tf-idf.html)

* [6 Easy Steps to Learn Naive Bayes Algorithm (with codes in Python and R)](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)

* [Feature Extraction and Transformation in Adobe Spark - Documentation](https://spark.apache.org/docs/2.2.0/mllib-feature-extraction.html#tf-idf)



## Lession 1: Introduction to Big Data
* Identify the pieces of the Hadoop ecosystem.
* Identify the differences and similarities between Hadoop and Spark.
* Write MapReduce jobs locally with mrjob.
* Manipulate data using PySpark DataFrames.


### Additional Resources: Lesson 1
[Four V's of Big Data](https://www.dummies.com/careers/find-a-job/the-4-vs-of-big-data/)  
[MrJob](https://pythonhosted.org/mrjob/)  
[MapReduce Explained](https://searchcloudcomputing.techtarget.com/definition/MapReduce)  
[MapReduce vs Spark](https://searchcloudcomputing.techtarget.com/answer/MapReduce-vs-Spark-Which-should-I-choose-for-big-data-in-the-cloud)  
[PySpark Tutorial](https://www.datacamp.com/community/tutorials/apache-spark-python)  
[PySpark Documentation](http://spark.apache.org/docs/latest/api/python/index.html)  


* Additional Notes
    * Four V's of Big Data:   
        1) Volume: Size of Data  
        2) Velocity: How quickly the data is coming in  
        3) Variety: Diversity of Data  
        4) Veracity: Uncertainty of Data  
* __01-Ins_MapReduce__  
    * first create a job that is defined by a clss that inherits from `MRJob`.  The class contains methods that define the steps of your job.
    * Steps consist of `mapper(), combiner(), reducer()` All are optional but you have to have at least one. 
    * `mapper()` takes a key and a value as args and yields as many key-value pairs as it likes. (in first example the key is ignored)
    * `reducer()` method takes a key and an iterator of values and also yields as many key-value pairs as it likes.  
    * FINAL REQUIREMENT EVERY TIME MUST END THE FILE WITH 
    ```python
    if __name__ == '__main__':
        yourfilename.run() #where yourfilename is the job class.
    ```
* __02-Evr_Word_Count__  
* __03-Evr_MrJob_CSV__  
* __04-Stu_Austin_Snow__  
* __05-Ins_Pyspark_DataFrames_Basics__
  * Spark uses the .`show()` method to display the data from DataFrames.
  * Columns can be manipulated using the `withColumn()` method.
  * Columns can be renamed using `withColumnRenamed()`.
  * Use `collect()` to get a list of values from a column.
  * Use `toPandas()` to convert a PySpark DataFrame to a Pandas DataFrame. This should only be done for summarized or aggregated subsets of the original Spark DataFrame.
* __06-Stu_Pyspark_DataFrames_Basics__  
* __07-Ins_Pyspark_DataFrames_Filtering__  
    * `filter()` method allows more data manipulation, similar to SQL's `WHERE` clause.
* __08-Stu_Pyspark_DataFrames_Filtering__  
* __09-Ins_Pyspark_DataFrames_Dates__  
    *  void errors in reading the data `inferSchema=True, timestampFormat="yyyy/MM/dd HH:mm:ss"` is used to tell Spark to infer the schema and use this format for handling timestamps
* __10-Stu_Pyspark_DataFrames_Dates__ 


## Lession 2: Big Data in the Cloud
* Explain why NLP is necessary for a big data tool kit.
    * Natural language processing is a field focused on the goal of having computers interact with (understand and generate) natural (human) language.
* Apply transformations resulting from NLP data processing to PySpark DataFrames.
* Explain and utilize PySpark text-processing methods like tokenization, stop words, n-grams, and term and document frequency.
* Describe example steps in an NLP data processing pipeline.


### Additional Resources: Lesson 2
[Spark Machine Learning Library](https://spark.apache.org/docs/latest/ml-guide.html)  
[Intro to NLP](https://medium.com/analytics-vidhya/introduction-to-natural-language-processing-part-1-777f972cc7b3)  
[Spark UDF's](https://jaceklaskowski.gitbooks.io/mastering-spark-sql/spark-sql-udfs.html)  
[StopWordsRemover](https://spark.apache.org/docs/latest/ml-features.html#stopwordsremover)  
[Spark TF-IDF](https://spark.apache.org/docs/2.1.0/ml-features.html#tf-idf)  
[What does tf-idf mean?](http://www.tfidf.com/)  
[The TD-IDF Algorithm Explained](https://www.onely.com/blog/what-is-tf-idf/)  
[TD-IDF Weighting](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html)  
[Into Naive Bayes](https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf)  
[Spark MLib Naive Bayes](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3741049972324885/3783546674231736/4413065072037724/latest.html)  
[Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/)  
**Great Article Showing Comparison of Models**  
[Multi-Class Text Classification with PySpark](https://towardsdatascience.com/multi-class-text-classification-with-pyspark-7d78d022ed35)  


* __01-Ins_Pyspark_NLP_Tokens__  
    * Tokenization segments the text into a group of characters that have meaning. The text is separated into tokens, which is similar to the `.split()` method in Python
    * Words are tokenized by applying the `Tokenizer()` function.  Then apply the `transform` method from tokenizer to turn DataFrame into tokenized data set.
    * `tuncate = False` parameter in the `show()` method returns whole column
* __02-Ins_UDF__
    * Spark UDFs (user-defined functions) allow Python functions to be passed into SQL.  Allowing to add custom output columns.
* __03-Stu_Pyspark_NLP_Tokens__  
    * Practice using `Tokenizer()` function, `udf()` 
* __04-Ins_Pyspark_NLP_Stopwords__  
    * `StopWordsRemover` words that should be excluded from the input, typically because they do not carry much meaning (ex "and", "the", "a", etc) to sentence.
    * Removing stop words from the data can imporve the accuracy of language model because it removes words that aren't important to the text.
    * Using `StopWordsRemover` is like `Tokenizer` function in it takes an input column and returns a output column with stop words removed.
        * `stopWords =` parameter in `StopWordsRemover` allows to customize words to remove.
* __05-Stu_Pyspark_NLP_Stopwords__  
* __06-Ins_Pyspark_NLP_HashingTF__  
    * NLP TF-IDF with HashingTF. 
        * `TD-IDF` term frequency-inverse document frequency, a vectorization method for showing how important a word is in a text.
        * `TF` Term Frequency the value of a word increases based on how often it appears in a document
        * `IDF` Inverse Document Frequency is a measure of how significant a word is with respect to the entire corpus.
        * `TF` can be calculated using:
            * `HashingTF` is a `Transformer` taking sets of terms and converting those sets into fixed-length feature vectores, map to an index, and return a vector of term counts. `numFeatures` parameter specificies number of buckets which words will be split.  Must be higher than the number of unique words. By default, this value is `2^18`, or `262,144`. A `power of 2` should be used so that indexes are evenly mapped
            * `CountVectorizer` converts text documents to vectors of term counts.
        * `IDF` is an `Estimator` which is fit on a dataset and produces an `IDFModel`.  
* __07-Stu_Pyspark_NLP_HashingTF__  
* __08-Evr_Naive_Bayes_Reviews__  
    * from Medium Olli Huang `Naive Bayes` family of algorithms based on applying Bayes theorem with strong(naive) assumption, that every feature is independent of the others, in order to predict the category of given sample.  Naive Bayes classifiers used extensively in NLP.  
    $P(A|B) = P(B|A) x P(A)/P(B)$
    * `MulticlassClassificationEvaluator` to evaluate the model on the testing set.


## Lession 3: Cloud ETL
* Define and create ETL pipelines with SQL and Python.
* Create and use databases in the cloud.
* Define and create ETL pipelines in the cloud.
* **Amazon Elastic MapReduce (EMR)** is an Amazon Web Services (AWS) tool for big data processing and analysis. **Amazon EMR** offers the expandable low-configuration service as an easier alternative to running in-house cluster computing.


### Additional Resources: Lesson 3
[pgAdmin Download](https://www.pgadmin.org/download/)  
[pgAdmin User Interface](https://www.pgadmin.org/docs/pgadmin4/4.x/user_interface.html)  
[pgAdmin Connect to Server](http://127.0.0.1:50384/help/help/server_dialog.html)  
[AWS Free Tier](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc)  
[AWS RDS GUIDE](https://smu.bootcampcontent.com/SMU-Coding-Bootcamp/SMDA201902DATA2/blob/master/01-Lesson-Plans/22-Big-Data/3/Supplemental/AWS_RDS_guide.pdf)  
[RDS pgAdmin GUIDE](https://smu.bootcampcontent.com/SMU-Coding-Bootcamp/SMDA201902DATA2/blob/master/01-Lesson-Plans/22-Big-Data/3/Supplemental/RDS_pgAdmin_guide.pdf)  
[S3 GUIDE](https://smu.bootcampcontent.com/SMU-Coding-Bootcamp/SMDA201902DATA2/blob/master/01-Lesson-Plans/22-Big-Data/3/Supplemental/S3_guide.pdf)  
[S3 Security Breaches](https://securityboulevard.com/2018/01/leaky-buckets-10-worst-amazon-s3-breaches/)  
**for fun**
[Set up Jupyter Notebook in AWS EMR](https://aws.amazon.com/blogs/big-data/running-jupyter-notebook-and-jupyterhub-on-amazon-emr/)  
[PySpark in Jupyter Notebook - Working with Dataframe & JDBC Data Source](https://medium.com/@thucnc/pyspark-in-jupyter-notebook-working-with-dataframe-jdbc-data-sources-6f3d39300bf6)  



* __00-AWS_Free_Tier__  
    * Create a PostrgreSQL Database in Amazon Web Services RDS (Relational Database Service)
* __01-Evr_S3__  
    * AWS S3 stands for `Simple Storage Service`.  It is Amazon's cloud file storage service that uses key-value pairs.  Files are stored on multiple servers and have high rate of availability.
    * S3 uses buckets to store files (similar to computer folders, directories, or GitHub repository). Buckets can contain additional files and folders but **not other buckets**.  Each bucket must have a unique name (a URL that is unique across AWS so get creative but not overly)
    * S3 buckets can have individual access or total public access.
* __02-Evr_RDS_CRUD__  
    * I would highly suggest if anyone is looking to use an AWS RDS to read through security parameters and how to properly set up.  IE not made available to the world.  
    * `Endpoint & port` are used to connect to the database.
    * After database is created you can go to `Security Group Tab`.  
        * `Security groups` tell the RDS instance what traffic is allowed into and out of the database.
        * Security can range from restrictive to open
        * For our activities the database is open **THIS IS NOT RECOMMENDED FOR PRODUCTION CODE**
* __03-Ins_ETL_S3_RDS__  
    * Typically you need a UI to interact or do queries on AWS RDS.  For this class we are using PostgreSQL.  You can connect a local database to your RDS instance by pointing to the correct `Endpoint`
    * The ETL process will need to `extract` the necessary data from the CSVs, `transform` it, and then `load` the data into these tables.
    * Example loading S3 bucket to Spark Dataframe.  This constitutes the `extract` phase:
    ```python
    from pyspark import SparkFiles
    # Load in user_data.csv from S3 into a DataFrame
    url = "https://s3.amazonaws.com//<bucket name>/user_data.csv"
    spark.sparkContext.addFile(url)

    user_data_df = spark.read.option('header', 'true').csv(SparkFiles.get("user_data.csv"), inferSchema=True, sep=',')
    user_data_df.show(10)
    ```
    * You will then do some transformation to the data using Pyspark DataFrame `transform` phase
    * After finished with transformation you will `load` to RDS.
    * Example Load
    ```python
        mode = "append"
        jdbc_url="jdbc:postgresql://<endpoint>:5432/my_data_class_db"
        config = {"user":"root", "password": "<password>", "driver":"org.postgresql.Driver"}
        clean_user_df.write.jdbc(url=jdbc_url, table='active_user', mode=mode, properties=config)
    ```
     * Supported Writing Modes:
         * `append`: Append contents of this DataFrame to existing data.
         * `overwrite`: Overwrite existing data.
         * `ignore`: Silently ignore this operation if data already exists.
         * `error (default case)`: Throw an exception if data already exists.
* __04-Stu_ETL_S3_ZEPL__  
* __05-Evr_ZEPL_RDS__  
    * WE DID NOT REVIEW THIS DO TO OUR WORKAROUND AND NOT HAVING TO USE ZEPL.  You are welcome :)
* __06-Stu_Big_Data_Review__  
* __07-Stu_Cloud_ETL_Project__ 
    * If have time stick or stick around after class we will review:
        * cloud_etl_analysis
        * cloud_etl_nlp
    * Both of these are just reviews of earlier material but good to go through.


