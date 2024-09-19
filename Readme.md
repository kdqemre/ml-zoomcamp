Midterm Project  
Predicting Water Pumps Status

Emre Ozturk 
 
## Description of the problem:  
Using data (WP_fulldataset.csv) provided from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional, which are functional but need some repairs (functional needs repair), and which don't work at all (non functional)? Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania. 
Prelude to Study: 
First of all, I have performed the domain research shortly; try to understand data frame, data structure and feature types. Based on this, our findings as following; 
- The given features have object type dominantly 
- Null values with the features public meeting, scheme management and permit 
- 0 values with the features construction year and population 
- Duplicate values - Some feature values are meaningless - Found “unknown” and  “other” as feature type - Realize the attributes water quality and quality group have same values 
 
Additionally, detailed exploratory analysis has been performed. Distribution of target and feature values and univariate analysis have been shown. With Univariate analysis, the relations between the attributes is examined as detailed. 
 
## Preprocessing:
IMPORTANT NOTE: Remember Input data also needs preprocessing for to use model and get correct predictions. 
In this step, I have take care the findings as the result of above analysis. 
- To dealing with the null values, I have replaced them as “various”, then separate these data from the main data.  
- After that, separated data as “Various” type, have been filled with mode of main, meaningful data. 
- For the 0 values, it has been filled with the mean of clean data. - Dropped the duplicate values and keep the last duplicate one. 
- The smoothing operation has been performed as replace the “unknown” type with the “other” type. Meaning, I add the “unknown” type to the “other” type. 
- The feature “quality group” has been dropped due to the same values with the water quality. For dropping, I choose this feature because quality group has less data than water quality. Ultimately, more data is desired. 
- For categorical types, dummy variables are created. Based on this, the feature “extraction type other - mkulima/shinyanga” is seemed noise and would be a trouble. Therefore it is dropped. 
- For the feature selection, observed that life of a pump depends on the following features; construction year , population, quantity and waterpoint types. Therefore, I choose these mentioned features as important independents. 
Now, whole data is cleaned and is ready for the prediction.  
 
## Predictions: 
My prediction process is explained step by step as below. 
- Firstly the data is separated as train and test. 
- The scaling is performed to avoid feature dominancy and outliers. I trained this algorithms: Logistic Regression, Decision Tree, Random Forest and XGBOOST. As a result, I choose the Random Forest Model which gives the best accuracy value with 0.79 and this model is chosen also for the reasons that our data have many categorical values and RF algorithm needs less cost than other models. 
- I have tuned my algorithm to optimizing with below points; 

• Criterion:  Function to measure the quality (purity) of a split. Choices are 'gini' (default) for the Gini impurity and 'entropy' for the information gain. 
• Max features: Represents the number of features to consider when looking for the best split (default = None = n features) 
• Max depth: The maximum depth of the tree. This indicates how deep the tree can be. The deeper the tree, the more splits it has and it captures more information about the data. If none (default), then nodes are expanded until all leaves are pure or until all leaves contain less than min samples split samples. 
• Min samples split: Minimum number of samples required to split an internal node (default=2). A split will not happen if there are less records than min samples split. When I increase this parameter, the tree becomes more constrained as it has to be considered more samples at each node. 
• Min samples leaf: The minimum number of samples required to be at a leaf node (default=1). This parameter is similar with min samples splits. However, this describes the minimum number of samples at the leafs, the base of the tree. 
• Cross Validation (stratified k-fold): Used while algorithm being trained. I used 5 folds; it means 4 train and 1 validation. 
 
## Instructions (How to run the project):
First of all you need Python in your computer.
In the project folder you will find train.py , predict.py , RFmodel.bin and project’s notebook.
The trained algorithm needs a water pump’s features as an input. Remember Input data also needs preprocessing .
-	If you want to use the project locally, load and use the pre-trained model file :RFmodel.bin
The trained model needs a water pump’s features as an input
-	If you want to tune the model and use different algorithms you can use train.py script
-	In the project's notebook you can see pre-processing and data cleaning process.
-	If you want to use the model as a web service, you can use predict.py script. You should send a post request as an input and have to be json file.  
-   You can find predic-test.py script in project's folder for test the model
-   You will find a Docker file in the folder.  

## Dependencies   (How to run the project)
For use the project you need some dependencies:
- First of all you need Python in your computer.
- For notebook you need an anaconda distribution and jupyter notebook.
- sklearn library: You need to install this library for load and use the model. It is essential for use the pre trained RFmodel
- If you want to use the project’s notebook  you need to install some libraries:
Numpy , pandas, sklearn, xgboost

## How to install dependencies (for windows):

First in cmd you can create a virtual environment:

conda create -n my-env
Than activate the environment:

conda activate my-env
Installing Anaconda distribution: You can download anaconda from here https://www.anaconda.com/products/individual

Installing Numpy:

You can use 

conda install numpy
or  
pip install numpy

Installing Pandas:

In virtual environment:

conda install pandas

or

pip install pandas

Installing Sklearn:

In your virtual environment:

pip install -U scikit-learn


### Installing XGBOOST:

If you use anaconda or miniconda, try installing xgboost with conda.
First, uninstall xgboost with pip 
(if you already installed it previously with pip):

pip uninstall xgboost

Then re-install it with conda:

conda install -c conda-forge xgboost
It will also install the required version of libomp.

#Using Docker File:
First install the python 3.9, the slim version have less size
FROM python:3.9-slim

We have created a directory in Docker named app and we're using it as wor directory 
WORKDIR /app                                                                

Copy the requirements.txt into our working derectory 
COPY ["requirements.txt","./"] 

install the requirements.txt  dependecies we had from the project and deploy them 
RUN pip install -r requirements.txt

Copy any python files and the model we had to the working directory of Docker 
COPY ["predict.py","RFmodel.bin", "./"] 

We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

If we run the Docker image, we want our pump status prediction app to be running
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]









