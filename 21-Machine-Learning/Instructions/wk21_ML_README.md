# Week 21 Lesson to Machine Learning

## Objectives
* Calculate and apply regression analysis to datasets.
* Understand the difference between linear and non-linear data.
* Understand how to quantify and validate linear models.
* Understand how to apply scaling and normalization as part of the data preprocessing step in machine learning.
* Understand how to calculate and apply the fundamental classification algorithms: logistic regression, SVM, KNN, decision trees, and random forests.
* Understand how to quantify and validate classification models including calculating a classification report.
* Understand how to apply `GridSearchCV` to hyper tune model parameters.
* Understand unsupervised learning and how to apply the kmeans algorithm.
* Articulate specific problems on which neural nets perform well.
* Use sklearn's to build and train a deep neural network.
* Use Keras to build and a train a deep neural network.

**What is the relevance of machine learning?**
Machine Learning is one kind of data analysis that automates model building leading us into the creation of systems that can identify patterns, make decisions, and learn without much human intervention. It's an exciting realization of the growth of artificial intelligence as a computer learns from previous computations and produces reliable and repeatable results from their decisions.

While Machine Learning has been around for a while, there's been a spike in its popularity in more recent years. You can hear about it used more by marketing professionals through Twitter, in the recommendation system that Netflix, Hulu, and Amazon use, and even within self-driving cars. The purpose of Machine Learning here is to analyze more data in less time, delivering accurate results every time.

This branch of data analytics is going to continue growing in various areas including the financial sector to identify investment opportunities, government work to detect fraud, and the further advancement of wearable healthcare devices.

**How much Machine Learning knowledge will I walk away with?**
Machine learning is a very deep well of knowledge. We could have an entire course dedicated to Machine Learning and still walk away wanting to know more. This course is meant to provide you with an introduction into Machine Learning and give you the tools and familiarity to continue this education on your own. Take advantage of in-class time, office hours, and your network of peers to continue learning more about Machine Learning.


## Lesson 1: Introduction to Machine Learning
* Understand how to calculate and apply regression analysis to datasets.
* Understand the difference between linear and non-linear data.
* Understand how to quantify and validate linear models.
* Understand how to apply scaling and normalization as part of the data preprocessing step in machine learning.


### **Additional Resources: Lesson 1**  
[SciKit Learn Machine Learning Tutorial](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)  
[Top 20 Machine Learning Software and Tools](https://www.ubuntupit.com/top-20-best-machine-learning-software-and-tools-to-learn/)  
[Top 8 Open Source](https://opensource.com/article/18/5/top-8-open-source-ai-technologies-machine-learning)  
[DataCamp Ridge, Lasso and Elastic Net](https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net)  
[Sci Kit Learn Common Terms](https://scikit-learn.org/stable/glossary.html)  
[random state](https://www.google.com/search?q=the+answer+to+life+the+universe+and+everything&rlz=1C1AVFC_enUS833US833&oq=the+answer&aqs=chrome.0.0j69i57j0l4.2848j0j7&sourceid=chrome&ie=UTF-8)  
[Distill](https://distill.pub/about/) - good learning resource  
[Open AI](https://openai.com/)   
[Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)  
[Interpreting Residual Plots](http://docs.statwing.com/interpreting-residual-plots-to-improve-your-regression/#nonlinear-header)  
[Supervised Learning](http://cs229.stanford.edu/notes/cs229-notes1.pdf)  
[Machine Learning with Sci-Kit Video](https://www.youtube.com/watch?v=4PXAztQtoTg)  




* __01-Ins_Univariate_Linear_Regression_Sklearn__  
* Linear Regression used to find best a line that best fits through the data. Example is making prediction of home selling price given an input of number of bathrooms.   
$y = \theta_0 + \theta_1 x$

    * $y$ is the output response
    * $x$ is the input feature
    * $\theta_0$ is the y-axis intercept
    * $\theta_1$ is weight coefficient (slope)
* Friendly advice before you dive into your model, you should check out your .head(), take a chance at .describe() what you see, look for additional .info() on your data and always always .scatter() to see some finer details
* OLS two critical characteristics of estimators to be condidered:
    * bias - is difference between true population parameter and the expected estimator (ie accuracy of measure)
    * variance - measures the spread or uncertainty in the estimates. where unknown error can be estimated from the residuals
    * From Medium Devin Soni: In any model, there is a balance between bias, which is the constant error term, and variance, which is the amount by which the error may vary between different training sets. So, high bias and low variance would be a model that is consistently wrong 20% of the time, whereas a low bias and high variance model would be a model that can be wrong anywhere from 5%-50% of the time, depending on the data used to train it. Note that bias and variance typically move in opposite directions of each other; increasing bias will usually lead to lower variance, and vice versa.
* __02-Stu_LSD__  
* __03-Ins_Quantifying_Regression__  
    $r^2$ will be close to 1.  How much variance has been explained or captured by your linear regression equation  
`MSE` will be close to 0  
`train_test_split()` to split data into training and testing data  
[Documentation Make Regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)  
* __04-Stu_Brains__  
    * machine learning algorithms work with numerical data.  if you have categorical data will need to transform to int using one hot or binary encoding `label_enconder.transform()`.  Pandas also provides` get_dummies()`
* Scaling and normalization - two most common are minmax and StandardScaler. use StandardScaler when you do not know anything about your data. StandardScaler applies a Guassian Distribution to our data where the mean is 0 and the standardard deviation is 1.  
* __05-Ins_Multiple_Linear_Regression_Sklearn__  
* __06-Stu_Beer_Foam__  
* __07-Ins_Data_Preprocessing__  
* __08-Stu_Respiratory_Disease__  
    `See additional resources for more on Lasso, Ridge, and ElasticNet` or this article
    [SciKit Learn Linear Model](https://scikit-learn.org/stable/modules/linear_model.html)  
    * OLS difference of squares like MSE
    * Lasso Model uses absolute value - absolute difference
    * Ridge Model uses squared difference
    * ElasticNet Model use a combination of both.  


## Lesson 2: Classification
* Understand how to calculate and apply the fundamental classification algorithms: logistic regression, SVM, KNN, decision trees, and random forests.
* Understand how to quantify and validate classification models including calculating a classification report.
* Understand how to apply GridSearchCV to hyper tune model parameters.


### **Additional Resources: Lesson 2**
[Supervised vs Unsupervised Learning](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)  
[Machine Learning](https://vas3k.com/blog/machine_learning/)  
[Difference between Classification and Regression](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)  
[Stratify Parameter Explained](https://michaeljsanders.com/2017/03/24/stratify-continuous-variable.html)  
[Decision Trees and Random Forest](https://towardsdatascience.com/decision-trees-and-random-forests-df0c3123f991)  
[Feature Importance in Random Forest Model](https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e)  
[K Nearest Neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)  
[F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)  
[F1 Score wiki](https://en.wikipedia.org/wiki/F1_score)  
[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)  
[Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)  
[Classification Report Explained](https://stackoverflow.com/questions/30746460/how-to-interpret-scikits-learn-confusion-matrix-and-classification-report)  
[Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)  
[Gamma and C](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)  


* __01-Ins_Logistic_Regression__  
    * `Logistic Regression` is a statistical method for predicting binary outcomes from data, where we apply an activation function in the final step of our linear model. This transforms numerical range to a bounded probability of classifying a discrete output between 0 and 1.  IE finding the numerical value for age and labeling as "young" vs "old"  
    * `make_blobs` function is used to generate two different groups (classes) of data.  Allowing for us to apply logistic regression to determine if a new data points belong to one of the groups.
* __02-Stu_Voice_Recognition__  
    * from Michael J Sanders: `stratify parameter` in `train_test_split` is a way to insure ostensibly random sample is a representative sample.  Usually comes into play for categorical variables.  
* __03-Ins_Trees__  
    * From Neil Liberman Medium: `Decision Trees` Trees answer sequential questions which send us down a certain route of the tree given the answer. The model behaves with “if this than that” conditions ultimately yielding a specific result.
    * From Neil Liberman Medium:`Random Forest` is simply a collection of decision trees whose results are aggregated into one final result. Their ability to limit overfitting without substantially increasing error due to bias is why they are such powerful models.
    * `feature_importances_` attribute, which returns an array of each feature’s importance in determining the splits.  
# ``` python
    import pandas as pd  
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index = X_train.columns,columns=['importance'])
    .sort_values('importance',ascending=False)
    ```
    
* __04-Stu_Trees__  
* __05-Ins_KNN__  
    * `KNN - K Nearest Neighbor` can be used for regression and classification.  Typically though is used for classification.
        * From wiki: `k-NN classification`, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
        * In `k-NN regression`, the output is the property value for the object. This value is the average of the values of k nearest neighbors.
* __06-Stu_KNN__  
* __07-Ins_SVM__  
    * From sklearn The `F1 score` can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:  
    `F1 = 2 * (precision * recall) / (precision + recall)`
    * In the multi-class and multi-label case, this is the average of the F1 score of each class with weighting depending on the `average` parameter.
* __08-Stu_SVM__  
* __09-Ins_GridSearch__  
    * Hyperparameter tuning with `GridSearchCV` a brute force approach trying different combinations of values to see which has the best performance. `GridSearchCV` this class is known as a `meta-estimator`, taking a model and a dictionary of parameter settings and tests all combinations of parameter settings to see which settings have the best performance. 
    * Once model has been trained can use the attribute `best_params_` to find best paramaters.  
    `print(grid.best_params_)`
    * You can also find the best score by using `best_score_` attribute  
    `print(grid.best_score_)`  
    * From wiki: `precision` (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, while `recall` (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. Both precision and recall are therefore based on an understanding and measure of `relevance`. `Precision` can be seen as a measure of exactness or quality, whereas `recall` is a measure of completeness or quantity. 
* __10-Stu_GridSearch__


## Lesson 3: Neural Networks and Deep Learning!
* Be able to articulate specific problems on which neural networks perform well.
* Be able to use Keras to build and train neural networks.
* Be able to use Keras to build and a train a deep neural network.
* Gain understanding of unsupervised learning and how to apply the Kmeans algorithm.


### **Additional Resources: Lesson 3**
[Tensor Flow Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.67480&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)  
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)  
[Keras Documentation](https://keras.io/)  
[Keras Models](https://keras.io/models/about-keras-models/)  
[Curse of Dimensionality Wiki](https://en.wikipedia.org/wiki/Curse_of_dimensionality)  
[Activation Function Wiki](https://en.wikipedia.org/wiki/Activation_function)  
[More on Activation Functions](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)  
[Sigmoid, tanh, Softmax, ReLU, Leaky ReLU EXPLAINED EVEN MORE](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)  
[ReLu Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))  
[L1 and L2 Regularization Methods](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)  
[One-Hot Encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)  
[NOIR](https://www.usablestats.com/lessons/noir)  
[Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy)  
[Cross Entropy Loss Function](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)  
[Dense Layer](https://www.quora.com/In-Keras-what-is-a-dense-and-a-dropout-layer)  
[Softmax Function](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d)  
[Optimizer Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)  
[Verbose Parameter](https://stackoverflow.com/questions/47902295/what-is-the-use-of-verbose-in-keras-while-validating-the-model)  
[HDF5](https://portal.hdfgroup.org/display/support)  #HDF5 is a data model, library, and file format for storing and managing data.  
[to_categorical](https://keras.io/utils/#to_categorical)  
[Kmeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)  
[Bias variance Tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229) #good explanation and aritcle  
[Deep Dream Generator](https://deepdreamgenerator.com/) # Cool site which contains set of tools which make it possible to explore different AI algorithms.  
[Feature Selection with sklearn and Pandas](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b) #good article when deciding what features to include in your model  
[Pandas Profiling](https://pandas-profiling.github.io/pandas-profiling/docs/#pandas-profiling) #cool basic tool for quick profile report of dataframe.


**Additional Notes**
* Artificial Neural Network is simply an architecture of interconnected neurons.
* Neural Network processes information by inputs flowing through network and are trained to detect higher level features as the progress through network unit an output or decision can be made.
* Neuron is similar to logistic regression.  Most common output of neural network is an output probability.  
* Neural nets are particularly powerful modeling nonlinearities with high dimensionality

__Tensor Flow Playground__  
* Epoch, Learning Rate, Activation Function, Regularization, Regularization Rate, Problem Type
    * All affect how quickly network learns and influence the goodness of it predictions
    * Epochs counts the number of training cycles, over which the network has been trained.
    * Learning Rate are constants in the learning equation.
    * Parameters are important but more advances would suggest you do additional research into how to tune.
* Data; Features; Hidden Layers and Output
    * Data is the dataset to train the model
    * Features allow you to specify properties to look for in the input data
    * Hidden Layers identify "higher-order" patterns and correlations amongst input features.
        * Common rule of thumb for 3-layer networks is to use 3 times as many nodes in hidden layer as input layer.
        * Hidden layers increases the "level of nonlinearity" a network can detect and fit.
    * Output image, plots the network's decision boundaries.
* Keras The Python Deep Learning Library
    * Capable of running on top of:
        * TensorFlow - Developed by Google
        * CNTK "Microsoft Cognitive Toolkit" - Developed by Microsoft
        * Theano - Developed by Montreal Institute for Learning Algorithms 

__Key Terms for Data Types__
* Two basic types of structured data: numeric and categorical. Numeric data comes in two forms:
    * `Continous` data that can take on any value in an interval
        * Synonyms: interval, float, numeric, ratio
    * `Discrete` data that can take only interger values such as counts
        * Synonyms: interger, count
* Categorical types only take a fixed set such as:
    * `Categorical` data that can take only a specific set of values representing a set of possible categories
        * Synonyms: enums, enumerated, factors, nominal, polychotomous
    * `Binary` a special case of categorical data with just two categories (0/1, true/ false)
        * Synonyms: dichotomous, logical, indicator, boolean
    * `Ordinal` Categorical data that has an explicit ordering
        * Synonyms: ordered factor


* __01-Ins_One_Hot_Encoding__  
    * From Jason Brownlee article: One-Hot Encoding is used for categorical variables where no such ordinal relationship exists, interger encoding is not enough
        * Categorical data is defined as variables with a finite set of label values.
        * Machine learning algorithms require numerical input and output variables.
        * An integer and one hot encoding is used to convert categorical data to integer data
    * One-Hot Encoding converts numeric value to one-hot encoded array.  Avoiding biasing the model by applying numeric classes.
* __02-Evr_First_Neural_Network__  
* Create sequential model  
# ```python
from keras.models import Sequential
model = Sequential()
# ```
    * `Sequential Model` is a linear stack of layers. Meaning the data flows from layer to the next layer.
* Add layer to function using `Dense` for densley connected layer
# ```python
from keras.layers import Dense
number_inputs = 3
number_hidden_nodes = 4
model.add(Dense(units=number_hidden_nodes,activation='relu',input_dim=number_inputs))
# ```
    * `input_dim` parameter equa to the input dimensions (number of features)
    * `units` parameter is the number of desired hidden nodes in the layer
    * `activation function ReLu` see addition resources links on activation functions.
* Add out put layer
# ```python
number_classes = 2
model.add(Dense(units=number_classes, activation='softmax')
# ```
    * Number of nodes for output layer equal possible outcomes or classes.
    * From Medium: Activation of `softmax` function turns numbers aka logits into probabilities that sum to one.  Outputs a vector that represents the probability distribution of a list of potential outcomes.  
* Compile model using a loss function and optimizer.
    * Use Categorical Crossentropy for classification models and MSE for linear regression.
* Fit (train) the model
    * `verbose` parameter used to see training progress for each epoch 
        * verbose = 0 : will show nothing
        * verbose = 1 : will show you animated progress bar
        * verbose = 2 : will show you one line per epoch
* Evaluate model using testing data
* Then make predictions with your model!
* __03-Evr_Deep_Learning__  
    * Many classification models designed simply draw a line between two linearly spearable regions of space.  This dataset is non-linear and no such line can be used to designate between two classes. 
    * Deep learning network encompasses a second hidden layer which the extra nodes help the network to adapt to non-linear data.
* __04-Stu_Moons__ - **homework example**  
* __05-Stu_Deep_Voice__  
    * `predicted_classes` method makes a prediction and returns the original categorical enconding.
    * `inverse_transfor` method is applied to convert encoded prediction to the original string value
* __06-Ins_Saving_Models__  
    * `model.save("modelname.h5")` can be called to save a trained model
    * To reuse or share trained models use
# ```python
from keras.models import load_model
voice_model = load_model("voice_model_trained.h5")
# ```
* __07-Stu_Smartphone__  
* __08-Ins_Kmeans__  
    * Unsupervised machine learning algorithms draw inferences directly from data without any previously labeled output(no `y` labels)
    * clustering analysis attempt to group data into clusters based on relationships and features in data.
    * Kmeans clustering groups the data into `k` groups.  The cluster center is the mean of all the points belonging to that cluster. A small `k` (ie `k=2`) will create larger clusters where as a larger `k` (ie `k=6`) will create smaller clusters.
    * Predicting new values with trained Kmeans model is looking at cluster centers to see which cluster is closest to the new data.
* __09-Stu_Kmeans__  



