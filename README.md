# Artificial-Neural-Network_tabular-dataset_PyTorch

Artificial Neural Network with tabular dataset_PyTorch. In this exercise I'll combine continuous and categorical data 

to perform a regression. The goal is to estimate the cost of a New York City cab ride from several inputs. 

The inspiration behind this code along is a recent Kaggle competition.

#Tabular Dataset

Here I avae done with tabular data (spreadsheets, SQL tables, etc.) with 

columns of values that may or may not be relevant. As it happens, neural networks can learn to make connections we probably wouldn't have developed on our own. However, to do this we have to 

handle categorical values separately from continuous ones,

The Kaggle competition provides a dataset with about 55 million records. 

The data contains only the pickup date & time, the latitude & longitude 

(GPS coordinates) of the pickup and dropoff locations, and the number of passengers.

It is up to the contest participant to extract any further information

WorkFlow>

   -Calculte the distance traveled
    
   -Add a datetime column and derive useful statistics

   -Separate categorical from continuous columns
   
   -Categorify
   
   -Convert numpy arrays to tensors
   
   -Convert continuous variables to a tensor
   
   -Convert labels to a tensor
   
   -Set an embedding size
   
   -Define a TabularModel
   
   -Define loss function & optimizer
   
   -Perform train/test splits
   
   -Train the model
   
   -Plot the loss function
   
   -Validate the model
   
   -Finally observed the first 50 predicted values.

   
