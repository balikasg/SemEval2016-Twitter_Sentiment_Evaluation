# Subtask D
This includes the code used for my participation in subtask D. It includes the feature extraction step and the learning steps.
To run it first clone the project and move to the current directory. In a terminal:

git clone https://github.com/balikasg/SemEval2016-Twitter_Sentiment_Evaluation.git
cd SemEval2016-Twitter_Sentiment_Evaluation/src/subtaskD
python phaseB.py 0.8

The numeric argument 0.8 is used for the a-power transformation which is applied to the n-gram and character-gram features.


### Requirements
scikit-learn 0.17
