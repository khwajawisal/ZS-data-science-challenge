# ZS-data-science-challenge
3 solution files, 1st solution LB-0.847, 2nd solution LB-0.921, 3rd solution (using stacking)  LB-0.956. Although 2nd and 3rd solution i created after the competition got over hope you can find something useful through this.
zsstack.py contains the code for solution , I use target mean encoding to encode my categorical features , then produce a soltion using stacking.
the base models i used are:
MLP based classifier,catboost,adaboost,random forest,Lightgbm and XGboost.
At 2nd level i.e. i tried various models as meta learners the best results were given by MLP based classifier with an accuarcay of 78% and LB score of 0.956 the LB rank at this score was 9th all india
although i was not able to submit this solution as i realised later that my handling of missing values was not proper plus targget mean encoding really helped me in improving my results significantly
You can try other modifications i neglected the adoption of time series analysis and i personally feel using time series will definetly help you 
break past 0.97 mark.Good Luck .Have fun and don't forget to give credit if you happen to use this code.
