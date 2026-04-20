# Part B - Business Case Analysis

## B1. Problem Formulation

### (a) ML Problem Formulation

**Target variable:** items_sold

**Input features:**
- store_id, store_size, location_type, competition_density, monthly_footfall
- promotion_type
- month, is_weekend, is_festival

**Problem type:**
I think this is a regression problem because we are predicting 
a number which is items_sold. We are not predicting a category 
so it is not classification.

The way I would solve this is — train the model to predict 
items_sold for each of the 5 promotion types separately. 
Then whichever promotion gives the highest predicted items_sold 
for that store in that month, we recommend that one.

---

### (b) Why items_sold is better than revenue

Revenue = price x quantity. When we give a flat discount the 
price goes down. So even if we sell more items the revenue 
might not increase much.

For example if a store sells 200 items at Rs 400 each that is 
Rs 80,000 revenue. But another store sells 150 items at Rs 500 
that is Rs 75,000. If we use revenue as target the model thinks 
200 items sold is worse which does not make sense.

items_sold directly shows how well the promotion worked. 
It does not depend on price so it is a cleaner target variable.

The main lesson here is that we should always pick a target 
variable that directly measures what we actually want. 
If we pick a wrong target the model will optimise for 
the wrong thing and give bad recommendations. 
This is called metric misalignment.

---

### (c) Problem with one global model

If we train one model for all 50 stores it will not work well. 
A store in a big city like Mumbai behaves very differently from 
a small rural store. One model cannot learn both at the same time.

My suggestion is to first group the stores into segments. 
For example urban large stores in one group and rural small 
stores in another group. Then train a separate model for 
each group.

This way each model only learns from similar stores and 
gives better predictions. It is also easier to update later 
if one group's behaviour changes we only retrain that model.

---

## B2. Data and EDA Strategy

### (a) Joining the four tables

We have four tables - transactions, store_attributes, 
promotion_details and calendar.

I would join them like this:

First join transactions with store_attributes on store_id. 
This gives us store size and location for each transaction.

Then join with promotion_details on promotion_type and month. 
This adds what promotion was running at that time.

Finally join with calendar on transaction_date to get 
is_weekend and is_festival for each date.

**Final table grain:**
One row = one store, one month, one promotion type

Before modelling I would aggregate:
- Sum of items_sold per store per month per promotion
- Average competition_density per store per month
- Total footfall per store per month

---

### (b) EDA I would do

**1. Bar chart - promotion type vs average items sold**

I would plot average items_sold for each promotion type. 
This tells me which promotion works best on average. 
If one promotion is always the winner the problem becomes simpler.

**2. Heatmap - location type vs promotion type**

I would make a heatmap of average items_sold for each 
combination of location and promotion type.
This shows me if BOGO works in cities but not in villages. 
If yes then I need separate models per location type.

**3. Line chart - monthly sales over 3 years**

I would plot total items_sold every month over 3 years. 
I want to check if sales go up in December or during festivals. 
If yes then month and is_festival are very important features.

**4. Histogram - distribution of items_sold**

I would check if items_sold values are normally distributed 
or very skewed. If heavily skewed I might apply log 
transformation before training a linear model.

---

### (c) 80% transactions have no promotion

This is a big problem. The model will mostly train on 
no-promotion data so it will not learn promotion patterns well. 
It might just recommend no promotion for every store.

What I would do:

First I would remove no-promotion rows when training the model 
that decides which promotion to use. Those rows are not helpful 
for comparing promotions.

Second if some promotion types are very rare I would oversample 
them so the model sees enough examples of each type.

Third I would use a two stage approach. Stage 1 decides should 
we run a promotion this month or not. Stage 2 decides which 
promotion to run. This way each model focuses on one task only.

---

## B3. Model Evaluation and Deployment

### (a) Train test split and metrics

**How I would split:**
I would sort data by date and use first 30 months for training 
and last 6 months for testing.

**Why not random split:**
If I randomly split the data some future months will go into 
training. The model will learn from future data which it should 
not know about. This is called data leakage. In real life the 
model only knows past data so we must test it on future data only.

**Metrics I would use:**

MAE - tells me on average how many items my prediction is wrong by. 
Easy to explain to business team. For example MAE of 20 means 
prediction is off by 20 items per store per month.

RMSE - penalises big errors more. Useful when a very wrong 
prediction causes serious problems like ordering too much stock.

R2 score - tells me what percentage of variation the model explains. 
R2 of 0.80 means model explains 80% of variation which is decent.

For this problem I would mainly focus on MAE because it is 
easy to understand and directly useful for the business team.

---

### (b) Explaining different recommendations using feature importance

Model said Loyalty Points for Store 12 in December and 
Flat Discount for Store 12 in March.

I would look at feature importances from the Random Forest model 
and check what was different between December and March for Store 12.

For December I would expect:
- is_festival = 1 because of Christmas and New Year
- month = 12
Customers are already buying a lot in festive months. 
Past data shows Loyalty Points work best then because 
people want to collect reward points when spending more.

For March I would expect:
- is_festival = 0
- Regular month with average footfall
In slow months customers need a direct price reason to shop. 
Flat Discount gives immediate savings so it works better.

To explain this to the marketing team I would make a simple table 
showing top 3 reasons for each recommendation in plain words. 
Something like - in December festive season drives loyalty program 
usage, in March direct discount brings more footfall.

---

### (c) Deployment process

**Step 1 - Save the model**

After training I would save the full pipeline including 
preprocessing and model together using joblib.
This makes sure new data is processed exactly like training data.
The saved file goes to cloud storage so it is always available.

**Step 2 - Prepare data every month**

At start of each month I would pull latest store data, 
add calendar flags for that month and make sure columns 
match what the model was trained on.

**Step 3 - Generate recommendations**

Load saved model and run predictions for each store.
For each store predict items_sold for all 5 promotions.
Recommend the one with highest predicted value.
Send this table to marketing team before month starts.

**Step 4 - Monitor the model**

Every month after actual results come in I would compare 
predicted vs actual items_sold and track MAE over time.

If MAE suddenly goes up by more than 20% I would investigate 
and possibly retrain the model.

I would also check if input data is changing a lot over time. 
For example if competition_density values shift a lot the 
model predictions may become unreliable.

I would retrain the model every 6 months using the most 
recent 2 years of data to keep it updated with latest trends.