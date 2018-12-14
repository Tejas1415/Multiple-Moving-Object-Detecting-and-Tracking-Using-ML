#### Multiple-Moving-Object-Detecting-and-Tracking-Using-ML
#### Coded by Tejas Krishna Reddy, November 2018.

Multiple Moving objects in a surveillance video were detected and tracked using ML models such as AdaBoosting. 
The obtained results were compared with the results from Kalman Filter.

Read through the power point ppt, for a detailed explanation over what we have done.
Youtube video, shows the results obtained. (Link in PPT).
The paper attached explains the different approaches in the past for implementing Moving Object Detection in a survailance Video. 
Run the files as shown in the youtube video to get the results. 

Download MV20011 database, which is an annotated database. The annotation is in the form of a XML file that can be converted to an excell file
to read and further build ML model based on that data. The images in this annotated data can be directly used for ML model, but we have implemented
a unique feature extraction method where Hu's Invariant moments, HOG features and Color Histogram values were  extracted as features from the database
and Adaboosting Technique was used to build the model since the features extracted were too weak, it was optimal to build 100's of classifiers and 
then converge the results to a strong answer.

For any further doubts, feel free to contact me at tejastk.reddy@gmail.com
