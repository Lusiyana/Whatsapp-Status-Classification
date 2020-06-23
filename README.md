# Whatsapp-Status-Classification

This project aims is to provide an automatic category classification to filter the collection of WhatsApp statuses. The algorithm used to build this program is Multinomial Naïve Bayes. The dataset containts 80 WhatsApp statuses that classified into 3 categories which are personal (30 datas), promotion (30 datas), and news (20 datas).

In the testing phase, 4 types of testing are carried out:
1. K-Fold Cross Validation Testing:
To evaluate the performance of the method that has been used by dividing the data into two parts, which are training data and validation data. In this test the data will be folded by K-value and it will be iterated by K-value. The K values used in this test are 2, 3, 5, 7, and 9.
2. Normalization Testing:
The accuracy of the data that has been normalized with the data that hasn't been normalized will be compared.
3. Term Weighting Testing:
The accuracy of data results that are only using the TF term weighting will be compared with data that use the TF-IDF term weighting.
4. Computation Time Testing: Comparing the execution time for K-Fold Cross Validation Testing, Normalization Testing, and Term Weighting Testing.

Tests result:
1. K-Fold Cross Validation Testing:
From the results of tests that have been done, obtained an average accuracy of 60.15% for K = 2, 75.27% for K = 3, 65% for K = 5, 69.52% for K = 7, and 71 .85% for K = 9.
2. Normalization Testing: 
By using the same random data, the accuracy obtained for data that has been normalized is 70% as well as for data that hasn't been normalized.
3. Term Weighting Testing: 
By using the same random data, the accuracy of the results obtained for data with TF term weighting is 70% as well as for data using TF-IDF term weighting.
4. Computation Time Testing:
The computational time required for K with values 2, 3, 5, 7, and 9 respectively are 9.8 seconds, 8.2 seconds, 5.7 seconds, 4.4 seconds, and 3.1 seconds.
The computational time required for data using only TF term weighting is 6.6 seconds, for data that hasn't been normalized and using TF-IDF term weighting is 6.7 seconds, while for normalized data is 6.9 seconds.

Based on several tests that have been done, the results show that the K-Fold Cross Validation test produces the best accuracy value of 75.27% at K = 3. Followed by an accuracy of 71.85% generated at K = 9. For TF term weighting, TF-IDF term weighting, normalized data and non-normalized data testing got the same accuracy that is equal to 70%. In computation time testing it was found that testing with K-Fold Cross Validation with K = 9 takes the least time which is 3 seconds. The greater the K used, the smaller the computation time. Meanwhile, for the normalization test computation time the results state that the non-normalized data had the least time which is 6.7 seconds. For the computation time of the term weighting test, the result for TF term weighting got the least time which is 6.5 seconds.
