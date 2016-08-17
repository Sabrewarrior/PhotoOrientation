# PhotoOrientation
Examining machine learning techniques to detect photo orientation

Flickr images were downloaded using https://github.com/Sabrewarrior/FlickrImages

Using data with the following characteristics: https://github.com/Sabrewarrior/PhotoOrientation/blob/master/data_info.txt
Data was hand examined to determine how many were wrongly classified to begin with. Most pictures seem to be oriented correctly and the first 10000 pictures has error rate or less than .05%.

Using Alexnet was able to get 80% accuracy. However just using hog descriptors is able to get similiar levels of accuracy. Adapting data for using with VGG16.

It seems that because of abundance of people in the data set it is learning to orient it properly more than the other data. Need to do further investigation.
