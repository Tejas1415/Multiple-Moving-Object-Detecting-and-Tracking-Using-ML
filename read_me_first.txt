Main M Files:

1)Run Ada_boost_learning.m - to generate model for learning based on extracted features.
And tests performance of learned model for known tracks .

2)Run Tracking_using_ML.m -to test the performance of tracking online.

Support files:
1)blob_state_reorder - support file for reordering detected blobs/objects
2)kalman_state_reorder - support file for reassigning kalman filter tracks
3) kalman filtering - kalman update and predict steps
3)hus_invariance - support file for hus variance computation

Database:
Training on file MVI_20011
Testing for confusion matrix on MV_20012


Testing of moving object detection and tracking on 
three video signals.

1st video is a vehicular traffic signal.
2nd video is an indoor video signal.
3rd video is a pedestrian traffic signal.
