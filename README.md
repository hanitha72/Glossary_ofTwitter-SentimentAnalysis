# Glossary_ofTwitter-SentimentAnalysis


<p align="center">
<img src = "images/tw3.png" width = 400 height=300>
</p>

This repository has a complete guide of Sentiment Analysis on twitter extract. It statrs from data extraction from twitter upto deployment.
twetter

**Data extraction:**
   Data extracted from twitter by using "tweepy" library and handled emoji's on web scrapping itself
   
 **Predictive Modelling**
 * Data cleanining done on  extrated twitter data
 * Text preprocessing like word_embeeding, tokenizing and lemmatization done. And vectorization of words done through countVectorization, tf-idf and through word2vec and    compared the **F1 score**  after building models.
 * Models built are,
   1. NaiveBayes
   2. MultiNominal Naivebayes
   3. SVM
   4. Random Forest
   5. LSTM
   6. BERT
   
      On the above all models BERT gave good results
      
  * Deployment done through Flask
  
  <p align="right">
<img src = "images/tw4.jpg" width = 200 height=200>
</p>

 
