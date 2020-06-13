
# Stats_170AB

Our project is on Predicting the Difference between Goodreads Ratings and Amazon Ratings

This folder is the subset of the complete_project file. It contains small sample datasets and only Jupyter Notebooks.

---

## Datasets

There are 4 datasets for our project: Goodreads metadata, Goodreads reviews, Amazon metadata, Amazon reviews. We did not choose to upload the reviews datasets but instead chose to upload the LDA dataset extracted from the reviews datasets since we did not use the reviews datasets for anything else. Since the files are too large and take too long to run, we have created a sample dataset of them and uploaded under the following names:
- ```gr_metadata_sample.csv```: contains a subset of the Goodreads metadata dataset (5k rows)
- ```am_metadata_sample.csv```: contains a subset of the Amazon metadata dataset (5k rows)
- ```am_gr_LDA_sample.csv```: contains ASIN feature along with 35 proportion topics features which were extracted from LDA model (5k rows)
   
----

## Code

Like the datasets, the code files are also shortened to only demonstrate important findings in order to reduce grading runtime. The code files in the folder are:
- ```eda.ipynb```: contains the code for plots of the distribution of average ratings of Goodreads and Amazon and relationships between various attributes and rating differences
- ```models_with_lda.ipynb```: contains the best Random Forests, Extreme Gradient Boosting (Xgboost), Neural Networks, and ensemble models we tested
