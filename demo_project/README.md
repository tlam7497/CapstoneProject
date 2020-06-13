
# Stats_170AB

Our project is on Predicting the Difference between Goodreads Ratings and Amazon Ratings

This folder is the subset of the complete_project file. It contains small sample datasets and only Jupyter Notebooks.

---

## Datasets

There are 4 datasets for our project: Goodreads metadata, Goodreads reviews, Amazon metadata, Amazon reviews. We did not choose to upload the reviews datasets but instead chose to upload the LDA dataset extracted from the reviews datasets since we did not use the reviews datasets for anything else. Since the files are too large and take too long to run, we have created a sample dataset of them and uploaded under the following names:
- ```gr_metadata_sample.csv```: contains a subset of the Goodreads metadata dataset (5k rows)
   - rows = books uniquely identified by ASIN
   - columns = metadata attributes such as number of ratings, number of reviews, genres, etc.
- ```am_metadata_sample.csv```: contains a subset of the Amazon metadata dataset (5k rows)
    - rows = books uniquely identified by ASIN
   - columns = metadata attributes such as number of ratings, number of reviews, genres, etc.
- ```am_gr_LDA_sample.csv```: contains ASIN feature for each book along with 35 proportion topics features which were extracted from LDA model (5k rows)
   - rows = books uniquely identified by ASIN
   - columns = proportions topics (each book's proportion of reviews under a certain topic)
   
----

## Code

Like the datasets, the code files are also shortened to only demonstrate code snippets with important findings in order to reduce grading runtime. To see the full code, look for the same filenames in the complete_project file.
Em ghi thêm kiểu như cái demo folder chỉ là lấy sample thôi nên eda hay mấy cái models hay clean sẽ ko thể hiện hết, muốn xem hết thì qua complete folder xem ( name và description của từng code file sẽ như nhau)
The folder contains the following code snippets.
- Data collection:
   - ```official_amazon.ipynb```:
   - ```official_goodreads.ipynb```:
- Data cleaning/processing:
   - ```amazon_text_cleaning.ipynb```:
   - ```goodreads_text_cleaning.ipynb```:
   - ```lda_wmallet_tune.py```:
- Exploratory data analysis (EDA)
   - ```eda.ipynb```: contains the code for plots of the distribution of average ratings of Goodreads and Amazon and relationships between various attributes and rating differences
- Prediction modeling:
   - ```linear_regressionl.ipynb```:
   - ```random_forest.ipynb```:
   - ```xgboost.ipynb```:
   - ```neural_network.ipynb```:
   - ```models_with_lda.ipynb```: contains the best Random Forests, Extreme Gradient Boosting (Xgboost), Neural Networks, and ensemble models we tested
