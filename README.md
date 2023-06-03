# livestream_sentimentAnalysis
 
# I. Introduction
The classification of positive and negative emotions is a well-known problem that has received numerous proposed solutions. In this presentation, I will outline a straightforward approach to classify user comments from chat channels in live streaming platforms.\
Here is the program flow:\
<img width="771" alt="Screenshot 2023-06-03 at 2 44 30 PM" src="https://github.com/justt1n/livestream_sentimentAnalysis/assets/66020389/c816fefe-8d26-4d8f-9da3-2f03b04daeea">
# II. Fine-tuning RoBERTa
<img width="666" alt="Screenshot 2023-06-03 at 2 47 58 PM" src="https://github.com/justt1n/livestream_sentimentAnalysis/assets/66020389/accb0bd2-b58e-448a-8d13-5795a2fbb1cf">

## The process of fine-tuning RoBERTa for emotion classification involves the following steps:

### 1. Data preparation: 
#### Prepare labeled training data consisting of sentences or text samples for emotion classification.
Usage dataset: Coronavirus tweets NLP - Text Classification
This is a dataset containing tweets related to the COVID-19 pandemic, collected from January 26, 2020 to March 26, 2020. This train dataset was used to classify emotions in tweets related to the COVID-19 pandemic, including 5 emotions: exceptionally positive, positive, exceptionally negative, negative, and neutral . This dataset includes 41157 tweets related to COVID-19 and is manually labeled. The dataset includes 6 columns: UserName, ScreenName, Location, Tweet At, Original Tweet, Label.
### 2. Data preprocessing: 
First, use the strip_emoji function to remove unnecessary emojis in tweets. Next, apply the strip_all_entities function to remove the links and usernames in the string. Then use the clean_hashtags function to remove the '#' character in the hashtag. By using filter_chars function we can remove any word containing unwanted special characters like '$' or '&'. Finally, use the remove_mult_spaces function to remove extra spaces in the string, remove non-english data.\

Next merge the particularly positive and particularly negative labels.\
<img width="344" alt="image" src="https://github.com/justt1n/livestream_sentimentAnalysis/assets/66020389/706bdb8f-ee67-464d-a792-9ea66ebfc770"> \
Finally use RandomOverSampler to balance the labels and then use one hot encoding to encode.\
<img width="326" alt="image" src="https://github.com/justt1n/livestream_sentimentAnalysis/assets/66020389/9a9db428-62d4-40e2-b95c-811f410aa268">

### 3. Fine-tuning:  Load the pre-trained RoBERTa model. 
I will use it as a feature extractor for emotion classification. Fine-tune the RoBERTa model on our training data by retraining a portion of the model on our labeled emotion classification dataset. \
<img width="745" alt="image" src="https://github.com/justt1n/livestream_sentimentAnalysis/assets/66020389/b385af71-fca1-41af-ae45-545f5c2a49b2">

### 4. Model evaluation:
<img width="400" alt="image" src="https://github.com/justt1n/livestream_sentimentAnalysis/assets/66020389/e0776ff0-d2c2-4642-8d60-38b5867354b0"> <img width="184" alt="image" src="https://github.com/justt1n/livestream_sentimentAnalysis/assets/66020389/619335dd-c7ad-45ac-bcfd-f61f0729306a">

# III. Usage

- Clone this repository 
- Install requirement by this command: \
`pip install -r requirements.txt `
- Paste Youtube/Twitch url into the program
- Press `Ctrl + C` to exit program. The csv file will be save after exit program.

