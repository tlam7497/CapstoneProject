
# Stats_170AB

Our project is on Predicting the Difference between Goodreads Ratings and Amazon Ratings

This folder is the subset of the complete_project file. It contains small sample datasets and only Jupyter Notebooks. To see the full code, look for the same filenames in the complete_project file.

---

## Datasets

There are 4 datasets for our project: Goodreads metadata, Goodreads reviews, Amazon metadata, Amazon reviews. We also uploaded the LDA dataset extracted from the reviews datasets. Since the files are too large and take too long to run, we have created a sample dataset of them and uploaded under the following names:
- ```gr_metadata_sample.csv```: contains a subset of the Goodreads metadata dataset (5k rows)
   - rows = books uniquely identified by ASIN
   - columns = metadata attributes such as number of ratings, number of reviews, genres, etc.
- ```am_metadata_sample.csv```: contains a subset of the Amazon metadata dataset (5k rows)
    - rows = books uniquely identified by ASIN
   - columns = metadata attributes such as number of ratings, number of reviews, genres, etc.
- ```am_gr_LDA_sample.csv```: contains ASIN feature for each book along with 35 proportion topics features which were extracted from LDA model (5k rows)
   - rows = books uniquely identified by ASIN
   - columns = proportions topics (each book's proportion of reviews under a certain topic)
- ```reviews.tsv```: contains a subset of text reviews
----

## Code

Like the datasets, the code files are also shortened to only demonstrate code snippets with important findings in order to reduce grading runtime. To see the full code, look for the same filenames in the complete_project file. The folder contains the following code snippets:
- Data collection:
   - ```official_amazon.ipynb```: contains the code for reading Amazon reviews and metadata from  Julian McAuley's dataset (http://jmcauley.ucsd.edu/data/amazon/)
   - ```official_goodreads.ipynb```: contains the code for crawling for reviews and metadata from Goodreads API
- Data cleaning/processing:
   - ```cleaning_text.ipynb```: contains the code for cleaning reviews ( This file is modified from am_text_cleaning.py in complete project folder. It only has a demo of 30 cleaned text reviews)
   - ```lda_wmallet_tune.py```: contains the LDA model features extraction
- Exploratory data analysis (EDA)
   - ```eda.ipynb```: contains the plots of the distribution of average ratings of Goodreads and Amazon and relationships between various attributes and rating differences
- Prediction modeling:
   - ```models_with_lda.ipynb```: contains the best Random Forests, Extreme Gradient Boosting (Xgboost), Neural Networks, and ensemble models we tested
   - ```am_predict_gr.ipynb```: contains the best RF + XGB + NN ensemble we got using Amazon features to predict rating differences
   - ```gr_predict_am.ipynb```: contains the best RF + XGB + NN ensemble we got using Goodread features to predict rating differences
   
 ----

## Instruction to run code
### DEMO NOTEBOOK:  models_with_lda.ipynb 
      Step 1: Download 3 csv files : gr_metadata_sample.csv, am_metadata_sample.csv, am_gr_LDA_sample.csv
      Step 2: Run models_with_lda.ipynb  ( It may take 59 secs to complete running this file)

### If you want to run the LDA model : lda_wmallet_tune.py  
NOTE : It take times to run this file, so only 30 text reivews were used.  
```
      Step 1 : Download reviews.tsv  
      Step 2 : Run cleaning_text.ipynb which produces reviews_cleaned.tsv  
      Step 3 : Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip and unzip  
      Step 3 : run lda_wmallet_tune.py ( remmeber to update the path in the below code at cell 9: 
                     mallet_path = 'mallet-2.0.8/bin/mallet'  
  ```

   
