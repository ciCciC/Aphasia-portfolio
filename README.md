<h1>Data Science portfolio Koray Poyraz</h1>
<p>Semester: Data Science</p>
<p>Project: Aphasia</p>

**Description**

Patients that suffer from Aphasia have difficulty comprehending and/or formulating language. 
The cause is usually brain damage in the language center. Recovery from Aphasia is usually never 100%, and rehab can take years but does help the patients. 
Regardless, having Aphasia is usually very stressful for the patients even during rehabilitation sessions. Specialists from the Rijndam rehabilitation institute in Rotterdam treat patients that suffer from Aphasia. 
Their impression is that the stress experienced by patients may be amplified by human-human interaction in which the patients experience the 'embarrassment' of not being able to communicate correctly. 
Possibly, the rehabilition stress can be reduced by having patients do exercises on a computer rather than to talk to a person. For this project the first goal is to see if we can properly translate what Aphasia patients say to text and identify where they likely make mistakes in their language.

<br />
<h2>Reading guide</h2>

- Communication
- Domain Knowledge
- Data Collection
- Data Preperation
- Predictive Models
- Data Visualization
- Oversampling
- Model Selection
- Evaluation
- Diagnostics
- Extra

<br />
<h2>Communication</h2>

- Summaries
  - [DTW algorithm](https://drive.google.com/open?id=1LXNcv708e6wNzxt1yUf-5IvGmn7w8j28)
  - [Aphasia](https://drive.google.com/open?id=1XC5KO49hhVlRnTzpUgk5_EsWqkBjdQA_)
  - [Phonology](https://drive.google.com/open?id=1eQMhui_E9tXWjDe0CW03YpHo1Rr4H6cb)
  - [Phonetics](https://drive.google.com/open?id=1NetEeGGN6kJM-wjqDAOdOYDvPhFIOtFv)
- Paper 
  - [A Review on Speech Recognition Technique](https://pdfs.semanticscholar.org/1062/8132a34301f66a0af4bc485f05e3988cdc44.pdf)
  - [Speech Processing for Machine Learning MFCCs](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
  - [Python For Audio Signal Processing](http://eprints.maynoothuniversity.ie/4115/1/40.pdf)
  - [librosa: Audio and Music Signal Analysis in Python](http://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf)

<br />

<h2>Domain knowledge</h2>
Below is described per subject the studies performed, techniques used, references to literature and results.

<h3>Jargon</h3>

- API = stands for "Application Programming Interface". Is a program that communicates depending on the protocols with another program. E.g. a program I developed communicates with the Google Speech to Text.
- Notebook = A web application that allows you to create and share documents that contain live code, comparisons, visualisations and narrative text.
- MFCCs = (Mel Frequency Cepstral Coefficient) a feature extraction method that is widely used in automatic speech and speaker recognition.  
- Scraper = also called web scraping, is for extracting data from websites.
- Library = a library of functions you can get to develop a program or script.
- SPHINX = a ready-made Speech to Text tool / engine with which you can develop your own Speech to Text.
- Phoneme boundary generator = a generator that generates phoneme boundaries.
- STT = Speech to Text (Google Service)

<h3>- Important articles for Speech Recognition systems</h3>
These articles have helped me get a picture to tackle this project. E.g. the feature extraction techniques that are applied or a pipeline that is used for Speech Recognition systems. This information also gave me the opportunity to submit an idea to my project group to go in the right direction.

- Literatuur
  - Provides a picture of the techniques that are applied
  - [A Review on Speech Recognition Technique](https://pdfs.semanticscholar.org/1062/8132a34301f66a0af4bc485f05e3988cdc44.pdf)
  - Provides the steps taken to extract the features from the audio signals
  - [Speech Processing for Machine Learning MFCCs](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
  - A GitHub repository with useful information about existing repositories for Speech Recognition systems
  - [awesome-python-scientific-audio](https://github.com/faroit/awesome-python-scientific-audio#feature-extraction)
  - DTW (Dynamic Time Warping) is used to compare word signals. In time series, dynamic time warping (DTW) is one of the algorithms for measuring the similarity between two temporal sequences, which can vary in speed.
  - [Understanding Dynamic Time Warping](https://databricks.com/blog/2019/04/30/understanding-dynamic-time-warping.html)

<br />
<br />
<h2>Data Collection</h2>
<p><b>"Data collection"</b> is important for the stage <b>"Data Preperation"</b>. Below are topics of tasks that have been performed to collect and structurally store the data for the project. Each topic can include desk research and notebooks to perform a specific task or tasks that are relevant to the project. Data has been collected from different sources. The sources are <b>"VoxForge", "Uva" and "CORPUS"</b>.</p>

<h3>- Uva data</h3>
<p>fon.hum.uva is a website where a free database is offered with spoken audio files and accompanying texts, <a href="http://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFAcorpus/">link to the website</a>. For retrieving data from Uva I wrote a scraper to get the data from their website. The reason for the scraper is because the database is not downloadable so one has to download from their website per click and that takes a lot of time.</p>

- Notebook
  - [Scraper Uva Data notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/ScrapingDataUva/ScrapingDataUva.md)

<h3>- VoxForge data</h3>
<p>VoxForge is a website where spoken audio files with accompanying texts are offered for free. For VoxForge I also wrote a scraper to get the data from their website, because like Uva, downloading per click takes a lot of time. Hence the scraper also for VoxForge. <a href="http://www.voxforge.org/home/downloads/speech/dutch">link to the website</a>.</p>

- Notebook
  - [Scraper VoxForge Data notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/ScrapingDataVoxforge/ScrapingDataVoxforge.md)

<h3>- CORPUS data</h3>
<p>CORPUS is a large data consisting of Dutch spoken audio with related words. This data is available free of charge at <a href="https://ivdnt.org/nieuws">this link</a>. The data is downloaded and stored on the server by our project manager. I wrote a transformer to transform the data into the desired structure to get started. It is described under "Data Preperation". This data collection is important for the <b> Phoneme Boundary Classifier </b>. </p>
  
<h3>Extra - folder structure for collected data</h3>
<p>As an extra I have written a method that creates a folder structure on the server for the collected data. This gives the possibility to easily store, process and create data. This makes the paths to the data clear, so a better overview of what is on the server.</p>

- Notebook
  - [Initialize folder structure notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/initialize_directory_structure/initialize_directory_structure.md)

<br />
<br />
<h2>Data Preperation</h2>
<p>Below are topics of tasks performed to prepare data for the project. Each topic can include desk research and notebooks to perform a specific task or tasks that are relevant to the project. </p>

<h3>- Develop API Aphasia by using Speech To Text Google Service</h3>
<p>I developed this API to quickly convert the process of audio files to text. Otherwise, that process had to be done manually which takes a lot of time. In addition, this API also has the function to get the timestamps of per word in an audio signal. This was important to be able to create a data set for future use, e.g. for a neural network.</p>

To realize this I created a project on GitHub called "Aphasia project". I also prepared an installation guide for my project colleagues so that they can use the API.
- Aphasia-project Github
  - [Aphasia-project Repository](https://github.com/ciCciC/Aphasia-project)

To get an overview of the existing Speech to Text service, I did a desk research. I have come to the conclusion that there are services from large companies that do not support the Dutch language except Google. In order to link Google's Speech to Text to my API, I consulted the following literature.
- Literature
  - [Google Speech to Text documentation](https://cloud.google.com/speech-to-text/docs/)

Google has a number of rules when it comes to transforming audio signal into text. One (without using Cloud Storage) may not pass on audio for more than 1 minute. Since we have audio files that are longer than a minute, another solution had to be found.

<b> First solution </b>was implementing a function that cuts an audio in minutes taking into account not cutting by word signal. I implemented this function to cut audio files within 1 minute and transform them to text.
Cut functie:
- [Method slicing audio PNG](https://drive.google.com/open?id=16UPK4XQozjz5NT5cblhc8xMPL29ilMy9)
- [Method slicing audio SCRIPT](https://github.com/ciCciC/Aphasia-project/blob/master/AudioTranscribe.py)

<b> Second solution </b> was to enable a Cloud Storage service and link it to the Aphasia API. This gives the freedom to transform audio into text for more than a minute.

The architecture of Aphasia API:
- [Aphasia API architecture PNG](https://drive.google.com/open?id=1G1ckCQ-MElPZKq9lQn3mzqChB0-xHwtU)

<h3>- Development of STT (Speech to Text) timestamps generator in notebooks</h3>
<p>On this topic I converted the Aphasia API into a notebook with additional functions to batch run the <b> data collection "Voxforge" </b> to transform a folder full of audio files into word timestamps and creating CSV files as a dataset. This datasets consists of columns "begin", "end", "word" and "audiopath" which will eventually be used with the "Phoneme boundary generator". See notebook for more information.</p>

- A desk-research into existing tools that can extract the word timestamps from an audio signal.
  - [existing word alignment tools for timestamps](https://drive.google.com/open?id=1-HS5edq61a1NErzNLFjARwHTNYibYAg6)

- Notebook
  - [stt timestamps generator notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/stt_timestamps_generator%20/stt_timestamps_generator.md)

<h3>- Development of Alignment</h3>
<p>An aligner script has been developed for this project. The aligner was important to be able to generate data as training set for SPHINX (a ready-to-use Speech to Text tool). The aligner is mainly intended for the data collection from "UVA" because the sentences are not aligned. The Aeneas library was used to realize this.</p>

- Aeneas documentation
  - [Aeneas documentation](https://www.readbeyond.it/aeneas/docs/)

- Aeneas library
  - [Aeneas library](https://www.readbeyond.it/aeneas/)

- Notebook
  - [aligner uva data notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/aligner_uva_data/aligner_uva_data.md)

<h3>- Development of a transformer for the CORPUS data</h3>
<p>For the data of CORPUS, a transformer has been written that transforms the data of CORPUS to the desired structure consisting of columns "begin", "end", "word" and "audiopath" and save as CSV file. This data is used with the <b> Phoneme Boundary Generator </b> which then generates a new data set for the <b> Phoneme Boundary Classifier </b>. See notebook for further information. This data is also been used by my project colleagues.</p>

- Notebook
  - [transforming CORPUS data](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/transforming_corpus_data/transforming_corpus_data.md)

<h3>- Development of a Phoneme boundary generator</h3>
<p>After "Data Collection" and "Data Preperation" topics, which have put the data in a desired structure, a Phoneme Boundary Generator has been developed. What this generator does is generate phoneme boundaries as data by concatenating the last N milliseconds of a word and beginning N milliseconds of the next word. This dataset is for training a <b> Phoneme Boundary Classifier </b>.</p>

<p>Two types of generators have been developed. The "V2" stores the aggregated N milliseconds as described above, and "V3" only stores the difference between the last N milliseconds of a word and the beginning N milliseconds of the next word. With this I want to see which approach produces a better validation accuracy and recall score.</p>

For feature extraction of the audio signals and obtaining the MFCCs, the following library and source were used:
- Source
  - [Speech Processing for Machine Learning MFCCs](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)

- Library
  - [python speech features library](https://github.com/jameslyons/python_speech_features)

- Notebook
  - [Phoneme boundary generator v2](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_generator_v2/phoneme_boundary_generator_v2.md)
  - [Phoneme boundary generator v3](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_generator_v3/phoneme_boundary_generator_v3.md)

<br />
<h2>Predictive Models</h2>
In this project, not only data collection or data preparation was important, but also the development and training of a Phoneme Boundary Classifier. In order to train a Phoneme Boundary Classifier model with the collected Dutch-language CORPUS data, a number of machine and deep learning models have been tested. The models are:

- Machine learning
  - Random Forest Classifier

- Deep learning
  - MLP (Multi Layer Perceptron)
  - Bi-LSTM (bi-directional Long Short-Term Memory)
  
For some of the above models, Scikit-Learn and Tensorflow Core library have been used.

One reason for using the Tensorflow Core is more customization options such as selection of the GPU cores, application of activation function per neural network layer and it is more suitable for developing deep learning networks.

These models have been trained with the data generated by the <b> Phoneme Boundary Generator (CORPUS NL) </b> to develop a Phoneme Boundary Classifier.

The goal of trying out these models is to ultimately choose a model to train it with the dataset.

- Random Forest Classifier
  - [phoneme boundary random forest classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_random_forest_classifier/phoneme_boundary_random_forest_classifier.md)
  
- MLP classifier (Multi Layer Perceptron)
  - [phoneme boundary MLP Classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_scikit_MLP/phoneme_boundary_scikit_MLP.md)
  
- Bi-LSTM classifier
  - [phoneme boundary Bi-LSTM Classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_classifier_LSTM/phoneme_boundary_classifier_LSTM.md)

<p> The dataset generated with the V2 phoneme boundary generator is used in the following topics: Oversampling, Model Selection, Evaluation and Diagnosis. The reason for this is because it provides a better result in validation accuracy and recall score. </p>

<br />
<h2>Data Visualization</h2>
<p> Below is a visualization of the datasets generated by V2 and V3 generators. Each dataset consists of the columns "region", "label", "sample_rate", "begin", "end" and "audiopath". </p>

<h3>- Data V2</h3>
<p>Overview dataset</p>
<img src="/notebooks_data/data_visualization/1nondifference.png" width="1000" height="400"/>
<p>Info datatypes</p>
<img src="/notebooks_data/data_visualization/2nondifference.png" width="300" height="200"/>
<p>Visualization N milliseconden audio signal and MFCCs</p>
<img src="/notebooks_data/data_visualization/3nondifference.png" width="640" height="300"/>

<h3>- Data V3</h3>
<p>Overview dataset</p>
<img src="/notebooks_data/data_visualization/1difference.png" width="1000" height="400"/>
<p>Info datatypes</p>
<img src="/notebooks_data/data_visualization/2difference.png" width="300" height="200"/>
<p>Visualization N milliseconden audio signal and MFCCs</p>
<img src="/notebooks_data/data_visualization/3difference.png" width="640" height="400"/>

<br />
<h2>Oversampling</h2>
<p>To counter <b> skewed classes </b>, a "generateMoreData ()" function has been written to improve the ratio between the label 0 and 1, making them balanced. This function is implemented in the notebooks of the models. </p>

<p>Before oversampling - here you can clearly see that the dataset suffers from <b> skewed classes </b> because the ratio between label 0 and 1 is not balanced. </p>
<img src="/notebooks_data/oversampling/before.png" width="450" height="350"/>

<p>After oversampling - here it can be clearly seen that the dataset is well balanced.</p>
<img src="/notebooks_data/oversampling/after.png" width="650" height="500"/>

<br />
<h2>Model Selection</h2>
<p> In this section it is important to select the most interesting values for the hyper parameters. For each model it is indicated which parameters it concerns, along with a plotted result. For further information see the notebook per model. Before the model selection is carried out, an <b> OVERSAMPLING </b> takes place to improve the ratio between the labels 0 and 1 so that they are balanced. Otherwise the dataset will of course have to contend with <b> Skewed Classes </b> and we don't want that. </p>

<h3>Random Forest Classifier</h3>

- notebook
  - [phoneme boundary random forest classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_random_forest_classifier/phoneme_boundary_random_forest_classifier.md)

<p>This section shows which value is interesting to use for the hyper parameters "max depth" and "estimators"</p>
-  Max depth
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/max_depth.png" width="450" height="350"/>
The plot above shows that the model becomes more complex, which leads to overfitting. This happens when the value for "Max of depth" is higher than 5. This information gives the possibility to choose a max depth to use a larger data set for training the model.
<br />
<br />
- Estimator

Below, the model is retrained, but with 1 million dataset. In order to realize a desired number of dataset, a function has been written that returns a balanced desired number of dataset called "getBatchData ()", see notebook. In this selection, the focus is on the "estimators" value.

<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/1_10_est.png" width="500" height="350"/>
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/20_40_est.png" width="500" height="350"/>
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/98_100_est.png" width="500" height="350"/>
<p>The plots from above show that there is very little difference after 8 estimators. Even with 100 estimators. The line of train and validation accuracy are not far apart. This indicates that there is no under- or overfitting. The estimator 32 gives the highest validation accuracy score.</p>

<h3>MLP Classifier</h3>

- notebook
  - [phoneme boundary scikit MLP classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_scikit_MLP/phoneme_boundary_scikit_MLP.md)
  
<p>Here we look at which value is interesting to use with the hyperparameters "number of neurons", "learning rate" and "number of layers"</p>
-  Number of neurons
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/num_neurons.png" width="450" height="350"/>
In the plot of the number of neurons we can see that he suffers from overfitting after 60 neurons.
<br/>
<br/>
-  Learning rate
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/learningrate.png" width="450" height="350"/>
In the learning rate plot we can see that the validation accuracy decreases with a higher learning rate.
<br/>
<br/>
-  Number of layers
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/layers.png" width="450" height="350"/>
In the plot of number of layers we can see that the validation accuracy decreases and training accuracy increases with more layers, so he overfit.
<br />
<br />
<p>Here we look at what classification report score the different values of the hypermeter "num_neurons" and "num layers" give.</p>

- number of neurons
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/recallneurons2.png" style="width:100%" height="300"/>
<p>In these plots we can see that the Recall at 70 neurons is highest at class 1 and lowest at class 0.
Since the focus is on class 1, 70 neurons is interesting.</p>

- number of layers
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/recalllayers2.png" style="width:100%" height="300"/>
<p>In these plots we can see that the Recall scores worse with more than 1 layer at class 1.
Since the focus is on class 1, 1 layer is interesting.</p>

<p>From the results above we see that 70 neurons with 1 layer gives the highest Recall score on class 1. We will use these values to create an MLP classifier.</p>

<h3>Bi-LSTM</h3>

- notebook
  - [phoneme boundary Bi-LSTM classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_classifier_LSTM/phoneme_boundary_classifier_LSTM.md)

<p>Here we look at which value can best be used with the hyper parameters "num neurons", "learning rate" and "learningsteps".</p>
-  Number of neurons
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/output_38_1.png" width="650" height="350"/>
In these plots we can see that the Recall at 70 neurons is highest at class 1 and lowest at class 0.
In the left plot we see that we are dealing with overfitting.
<br/>
<br/>
-  Learning rate
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/output_40_1.png" width="650" height="350"/>
In the plot of learning rate, we can see that the validation accuracy and Recall score at class 1 decreases with a higher learning rate.
<br/>
<br/>
-  number of training steps
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/output_42_1.png" width="650" height="350"/>
In the plot of learning steps we can see that the Recall score at class 1 is highest at approximately 8200 learning steps. However, we are struggling with an overfitting in the left plot.

<br />
<br />
<h2>Evaluation</h2>

<p>In this section, "oversampling" has been performed for each model first to improve the ratio between the label 0 and 1 so that they are balanced. For this I wrote the function "generateMoreData ()", see the topic <b> Oversampling </b>.</p>

<p>In this part, an evaluation of the results of the models that have been trained is performed. Finally, one model is chosen (model selection) and then proceeds to the last stage "Diagnosis".</p>

<h3>Random Forest Classifier</h3>

- notebook
  - [phoneme boundary random forest classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_random_forest_classifier/phoneme_boundary_random_forest_classifier.md)

<p>After model selection of the value for "max depth" and "estimators", the model was trained with the full datasets.</p>

- Train, validation acc., Recall and Precision score
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/nondiff.png" width="600" height="300"/>

<h3>MLP Classifier</h3>

- notebook
  - [phoneme boundary scikit MLP classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_scikit_MLP/phoneme_boundary_scikit_MLP.md)

<p>After selecting the values for "num neurons", "learning rate" and "num layers", the model was trained with the complete datasets.</p>

- Train, validation acc., Recall and Precision score
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/nondiff.png" width="600" height="300"/>

<h3>Bi-LSTM Classifier</h3>

- notebook
  - [phoneme boundary Bi-LSTM classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_classifier_LSTM/phoneme_boundary_classifier_LSTM.md)

<p>After selecting the values for "num neurons", "learning rate" and "number of training steps", the model was trained with the complete datasets.</p>

- Train and validation accuracy %
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/val.png" width="300" height="100"/>

- Recall and Precision score
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/recall.png" width="500" height="200"/>

- A score table

[0] = class 0,  [1] = class 1

| Score %        | RFC    | MLP   | Bi-LSTM |
| -------------  |:------:| -----:| -------:|
| Training acc.  | 0.56   | 0.62  |   0.76  |
| Validation acc.| 0.56   | 0.62  |   0.58  |
| Recall [0]     | 0.54   | 0.67  |   0.61  |
| Recall [1]     | 0.59   | 0.57  |   0.51  |
| Precision [0]  | 0.57   | 0.61  |   0.64  |
| Precision [1]  | 0.56   | 0.63  |   0.49  |
| F1 score  [0]  | 0.55   | 0.63  |   0.62  |
| F1 score  [1]  | 0.57   | 0.60  |   0.50  |

<p>From the score table we can see that the model MLP has the best results. He scores highest in validation accuracy, Precision class 1 and second best on recall class 1. This means that we proceed with MLP to the Diagnosis stage.</p>

<br />
<h2>Diagnostics of the learning process</h2>

- notebook
  - [phoneme boundary MLP classifier diagnostics](/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/phoneme_boundary_scikit_MLP_diagnos.md)

<p>In this section we continue with the chosen model MLP. Here we look at the problems the model faces, eg High Bias or High Variance. For further information is above the link to the notebook "phoneme boundary MLP classifier diagnostics".</p>

<p>In the plot of iterations we can see that we are dealing with HIGH VARIANCE (Overfitting). So the model is too complex.</p>
<img src="/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/iterations1.png" width="500" height="200"/>

<p>A zoomed-in plot on the learning process</p>
<img src="/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/iterations2.png" width="500" height="200"/>

<h3>Solution</h3>
<p>As a solution we will use <b> Regularization </b>.</p>

- Regularization
<p>By first plotting lambda values regularization we can see which value gives the best result.</p>
<p>Regularization values: [0, 1e-05, 0.001, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]</p>
<img src="/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/regularization.png" width="550" height="350"/>
<p>Above we see at a low lambda value HiGH VARIANCE and at a high lambda value HIGH BIAS. The lambda values: (0.64, 1.28, 2.56) give better generalization.</p>

<p>After regularization, the best value was chosen to reduce overfitting and underfitting.</p>

<p>Below the plot of the final model with the selected lambda value.</p>
<img src="/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/finalmodel.png" width="550" height="350"/>
<p>In this plot we can see that the model is not overfit or underfit but generalized.</p>


<br />
<h2>Extra</h2>

<h3>Generating a dataset for my colleagues</h3>

- notebook
  - [transformer mfcc, word, wordtranscription and phonemes](/extra/transformer_audio_word_to_mfcc.md)
  - [transformer word, sounds and phonemes](/extra/transforming_corpus_data.md)

<p>- Dataset MFCC and word of which only the words starting with "st".</p>
<img src="/extra/stwordlist.png" width="600" height="350"/>
<p>- Dataset MFCC and word.</p>
<img src="/extra/wordlist.png" width="600" height="350"/>
<p>- Dataset MFCC, word with sounds and phonemes code list.</p>
<img src="/extra/fondecoder.png" width="650" height="350"/>
