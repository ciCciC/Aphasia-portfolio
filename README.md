<h1>Datascience portfolio Koray Poyraz</h1>
<p>Minor: Datascience kb-74</p>
<p>Project: Aphasia</p>

**Description**

Patients that suffer from Aphasia have difficulty comprehending and/or formulating language. 
The cause is usually brain damage in the language center. Recovery from Aphasia is usually never 100%, and rehab can take years but does help the patients. 
Regardless, having Aphasia is usually very stressful for the patients even during rehabilitation sessions. Specialists from the Rijndam rehabilitation institute in Rotterdam treat patients that suffer from Aphasia. 
Their impression is that the stress experienced by patients may be amplified by human-human interaction in which the patients experience the 'embarrassment' of not being able to communicate correctly. 
Possibly, the rehabilition stress can be reduced by having patients do exercises on a computer rather than to talk to a person. For this project the first goal is to see if we can properly translate what Aphasia patients say to text and identify where they likely make mistakes in their language.

<br />
<h2>Reading guide</h2>

- Courses
- Scrumwise taskboards
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
<h2>Courses</h2>

- DataCamp
  - [DataCamp certifications](https://github.com/ciCciC/Aphasia-portfolio/blob/master/datacamp/datacamp_certifications.md)
- Coursera
  - [Coursera](https://github.com/ciCciC/Aphasia-portfolio/blob/master/coursera/results.md)

<br />
<h2>Communication</h2>

- Summaries
  - [DTW algoritme](https://drive.google.com/open?id=1LXNcv708e6wNzxt1yUf-5IvGmn7w8j28)
  - [Afasie](https://drive.google.com/open?id=1XC5KO49hhVlRnTzpUgk5_EsWqkBjdQA_)
  - [Fonologie](https://drive.google.com/open?id=1eQMhui_E9tXWjDe0CW03YpHo1Rr4H6cb)
  - [Fonetiek](https://drive.google.com/open?id=1NetEeGGN6kJM-wjqDAOdOYDvPhFIOtFv)
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
  
<b>Interview <br /></b>
During the interviews my task was not only to ask questions but also to record the interviews. I did this by using the voice recorder application on my phone. So that we can later listen to the recordings again for clarification of the conversations.
In addition, it was my task to hold a second interview at Rijndam Institute with Ms Ineke and the security manager (AVG), about getting the necessary audio data and the security of the data. The conversation about AVG was important for the use of the Google Services. Mainly the Google Text to Speech and Cloud Storage services. This was initially important for converting the aphasia audio files to text as soon as possible.

- Literature
  - [Security and Privacy Considerations](https://cloud.google.com/storage/docs/gsutil/addlhelp/SecurityandPrivacyConsiderations)

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
In dit project was mijn taak niet alleen data verzamelen of data voorbereiding maar ook het ontwikkelen en trainen van een Phoneme Boundary Classifier. Om een Phoneme Boundary Classifier model te kunnen trainen met de verzamelde Nederlands gesproken CORPUS data heb ik aantal machine en deep learning modellen uitgeprobeerd. Daarvoor heb ik de volgende modellen gebruikt:

- Machine learning
  - Random Forest Classifier

- Deep learning
  - MLP (Multi Layer Perceptron)
  - Bi-LSTM (bi-directional Long Short-Term Memory)
  
Voor sommige van de bovenstaande modellen heb ik gebruik gemaakt van Scikit-Learn en Tensorflow Core bibliotheek.

Een reden voor het gebruik maken van de Tensorflow Core is meer aanpas mogelijkheden zoals selectie van de GPU cores, toepassing van activation function per neurale netwerk laag en hij is meer geschikt voor het ontwikkelen van deep learning netwerken.

Deze modellen zijn getraind met de data die is gegenereerd door de <b>Phoneme Boundary Generator (CORPUS NL)</b> voor het ontwikkelen van een Phoneme Boundary Classifier.

Het doel van het uitproberen van deze modellen is om uiteindelijk een model te kiezen om hem te trainen met de dataset.

- Random Forest Classifier
  - [phoneme boundary random forest classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_random_forest_classifier/phoneme_boundary_random_forest_classifier.md)
  
- MLP classifier (Multi Layer Perceptron)
  - [phoneme boundary MLP Classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_scikit_MLP/phoneme_boundary_scikit_MLP.md)
  
- Bi-LSTM classifier
  - [phoneme boundary Bi-LSTM Classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_classifier_LSTM/phoneme_boundary_classifier_LSTM.md)

<p>De dataset die is gegenereerd met de V2 phoneme boundary generator wordt gebruikt bij de volgende onderwerpen: Oversampling, Model Selection, Evaluation en Diagnosis. De reden hiervoor is omdat hij een betere resultaat levert in validation accuracy en recall score.</p>

<br />
<h2>Data Visualization</h2>
<p>Hieronder heb ik een visualisatie van de datasets die zijn gegenereert door V2 en V3 generators. Elke dataset bestaat uit de kolommen "region", "label", "sample_rate", "begin", "end" en "audiopath".</p>

<h3>- Data V2</h3>
<p>Overzicht dataset</p>
<img src="/notebooks_data/data_visualization/1nondifference.png" width="1000" height="400"/>
<p>Info datatypes</p>
<img src="/notebooks_data/data_visualization/2nondifference.png" width="300" height="200"/>
<p>Visualizatie N milliseconden audio signaal en MFCCs</p>
<img src="/notebooks_data/data_visualization/3nondifference.png" width="640" height="300"/>

<h3>- Data V3</h3>
<p>Overzicht dataset</p>
<img src="/notebooks_data/data_visualization/1difference.png" width="1000" height="400"/>
<p>Info datatypes</p>
<img src="/notebooks_data/data_visualization/2difference.png" width="300" height="200"/>
<p>Visualizatie N milliseconden audio signaal en MFCCs</p>
<img src="/notebooks_data/data_visualization/3difference.png" width="640" height="400"/>

<br />
<h2>Oversampling</h2>
<p>Om <b>skewed classes</b> tegen te gaan heb ik een "generateMoreData()" functie geschreven om de verhouding tussen de label 0 en 1 te verbeteren waardoor ze gebalanceerd zijn. Deze functie is geimplementeerd in de notebooks van de modellen.</p>

<p>Before oversampling - hier is overduidelijk te zien dat de dataset kampt met <b>skewed classes</b> want de verhouding tussen label 0 en 1 is niet gebalanceerd.</p>
<img src="/notebooks_data/oversampling/before.png" width="450" height="350"/>

<p>After oversampling - hier is duidelijk te zien dat de dataset goed is gebalanceerd.</p>
<img src="/notebooks_data/oversampling/after.png" width="650" height="500"/>

<br />
<h2>Model Selection</h2>
<p>Bij dit onderdeel is het van belang om de meest interessante waardes te selecteren voor de hyperparameters. Per model wordt aangegeven om welke parameters het gaat met daarbij een geplotte resultaat. Voor verdere info zie de notebook per model. Voordat de model selection wordt uitgevoerd vindt eerst een <b>OVERSAMPLING</b> plaats om de verhouding tussen de label 0 en 1 te verbeteren waardoor ze gebalanceerd zijn. Anders kampt de dataset natuurlijk met <b>Skewed Classes</b> en dat willen we niet.</p>

<h3>Random Forest Classifier</h3>

- notebook
  - [phoneme boundary random forest classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_random_forest_classifier/phoneme_boundary_random_forest_classifier.md)

<p>Hier kijk ik naar welke waarde het beste kan worden gebruikt bij de hyperparameters "max depth" en "estimators"</p>
-  Max depth
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/max_depth.png" width="450" height="350"/>
Uit de plot van hierboven zien we dat het model complexer wordt dus overfit wanneer de waarde voor "Max of depth" hoger is dan 5. Dit geeft mij nu de mogelijkheid om een max depth te kiezen om een grotere dataset te gebruiken voor het trainen van het model.
<br />
<br />
- Estimator

Hieronder train ik opnieuw een model maar dan met 1 miljoen dataset. Om een gewenste aantal dataset te kunnen realizeren heb ik een functie geschreven die een gebalanceerde gewenste aantal dataset teruggeeeft genaamd "getBatchData()", zie notebook. Bij deze selectie ligt de focus op de "estimators" waarde.

<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/1_10_est.png" width="500" height="350"/>
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/20_40_est.png" width="500" height="350"/>
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/98_100_est.png" width="500" height="350"/>
<p>Bij de plots van hierboven zien we dat er vrij weinig verschil is na 8 estimators. Zelfs bij 100 estimators. De lijn van train en validation accuracy liggen niet ver van elkaar af. Dit geeft aan dat er geen sprake is van under- of overfitting. De estimator 32 geeft de hoogste validation accuracy score.</p>

<h3>MLP Classifier</h3>

- notebook
  - [phoneme boundary scikit MLP classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_scikit_MLP/phoneme_boundary_scikit_MLP.md)
  
<p>Hier kijk ik naar welke waarde het beste kan worden gebruikt bij de hyperparameters "num neurons", "learning rate" en "num layers"</p>
-  Num neurons
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/num_neurons.png" width="450" height="350"/>
In de plot van aantal neurons kunnen we zien dat hij kampt met overfitting na 60 neurons.
<br/>
<br/>
-  Learning rate
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/learningrate.png" width="450" height="350"/>
In de plot van learning rate kunnen we zien dat de validation accuracy omlaag gaat bij hoger learning rate.
<br/>
<br/>
-  Num layers
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/layers.png" width="450" height="350"/>
In de plot van aantal layers kunnen we zien dat de validation accuracy omlaag gaat en training accuracy hoger bij meer layers dus hij overfit.
<br />
<br />
<p>Hier kijk ik naar wat voor classification report score de verschillende waardes van de hypermeter "num_neurons" en "num layers" geven.</p>

- num neurons
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/recallneurons2.png" style="width:100%" height="300"/>
<p>In deze plots kunnen we zien dat de Recall bij 70 neurons het hoogst is bij class 1 en laagst bij class 0.
Aangezien de focus op class 1 ligt is 70 neurons interessant.</p>

- num layers
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/recalllayers2.png" style="width:100%" height="300"/>
<p>In deze plots kunnen we zien dat de Recall slechter scoort bij meer dan 1 layer bij class 1.
Aangezien de focus op class 1 ligt is 1 layer interessant.</p>

<p>Van de resultaten hierboven zien we dat 70 neurons met 1 laag hoogste Recall score geeft op class 1. Deze waardes gaan we gebruiken om een MLP classifier.</p>

<h3>Bi-LSTM</h3>

- notebook
  - [phoneme boundary Bi-LSTM classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_classifier_LSTM/phoneme_boundary_classifier_LSTM.md)

<p>Hier kijk ik naar welke waarde het beste kan worden gebruikt bij de hyperparameters "num neurons", "learning rate" en "learningsteps"</p>
-  Num neurons
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/output_38_1.png" width="650" height="350"/>
In deze plots kunnen we zien dat de Recall bij 70 neurons het hoogst is bij class 1 en laagst bij class 0.
In de linker plot zien we dat we echter te maken hebben met overfitting.
<br/>
<br/>
-  Learning rate
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/output_40_1.png" width="650" height="350"/>
In de plot van learning rate kunnen we zien dat de validation accuracy en Recall score bij class 1 omlaag gaat bij hoger learning rate.
<br/>
<br/>
-  aantal trainingsteps
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/output_42_1.png" width="650" height="350"/>
In de plot van learning steps kunnen we zien dat de Recall score bij class 1 het hoogst is bij circa 8200 learning steps. Echter kampen we bij de linker plot met een overfitting.

<br />
<br />
<h2>Evaluation</h2>

<p>Bij dit onderdeel heb ik voor elk model eerst "oversampling" uitgevoerd om de verhouding tussen de label 0 en 1 te verbeteren waardoor ze gebalanceerd zijn. Hiervoor heb ik de functie "generateMoreData()" geschreven, zie de onderwerp <b>Oversampling</b>.</p>

<p>Bij dit onderdeel voer ik een evaluatie uit van de resultaten van de modellen die ik heb getraind. Uiteindelijk wordt één model gekozen (model selection) om vervolgens naar de laatste etappe "Diagnosis" te gaan.</p>

<h3>Random Forest Classifier</h3>

- notebook
  - [phoneme boundary random forest classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_random_forest_classifier/phoneme_boundary_random_forest_classifier.md)

<p>Na model selectie van de waarde voor "max depth" en "estimators" heb ik het model getraind met de volledige datasets.</p>

- Train, validation acc., Recall en Precision score
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/nondiff.png" width="600" height="300"/>

<h3>MLP Classifier</h3>

- notebook
  - [phoneme boundary scikit MLP classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_scikit_MLP/phoneme_boundary_scikit_MLP.md)

<p>Na deze selectie van de waarde voor "num neurons", "learning rate" en "num layers" heb ik het model getraind met de volledige datasets.</p>

- Train, validation acc., Recall en Precision score
<img src="/notebooks_data/phoneme_boundary_scikit_MLP/nondiff.png" width="600" height="300"/>

<h3>Bi-LSTM Classifier</h3>

- notebook
  - [phoneme boundary Bi-LSTM classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_classifier_LSTM/phoneme_boundary_classifier_LSTM.md)

<p>Na deze selectie van de waarde voor "num neurons", "learning rate" en "aantal trainingsteps" heb ik het model getraind met de volledige datasets.</p>

- Train en validation accuracy %
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/val.png" width="300" height="100"/>

- Recall en Precision score
<img src="/notebooks_data/phoneme_boundary_classifier_LSTM/recall.png" width="500" height="200"/>

- Een score tabel

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

<p>Uit de score tabel kunnen we zien dat het model MLP het beste resultaten heeft. Hij scoort het hoogst in de validation accuracy, Precision class 1 en een na beste op recall class 1. Dit betekent dat ik verder ga met MLP naar de etappe Diagnosis.</p>

<br />
<h2>Diagnostics of the learning process</h2>

- notebook
  - [phoneme boundary MLP classifier diagnostics](/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/phoneme_boundary_scikit_MLP_diagnos.md)

<p>Bij dit onderdeel ga ik verder met het gekozen model MLP. Hier ga ik kijken met welke problemen het model kampt, bijv. High Bias of High Variance. Voor verdere info staat hierboven de link naar de notebook "phoneme boundary MLP classifier diagnostics".</p>

<p>In de plot van iterations kunnen we zien dat we kampen met HIGH VARIANCE (Overfitting). Dus het model is te complex.</p>
<img src="/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/iterations1.png" width="500" height="200"/>

<p>Een ingezoomde plot op het leerproces</p>
<img src="/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/iterations2.png" width="500" height="200"/>

<h3>Oplossing</h3>
<p>Als oplossing ga ik gebruik maken van <b>Regularization</b>.</p>

- Regularization
<p>Door lambda waardes regularization eerst te plotten kunnen we zien welke waarde de beste resultaat geeft.</p>
<p>Regularization values: [0, 1e-05, 0.001, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]</p>
<img src="/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/regularization.png" width="550" height="350"/>
<p>Hierboven zien we bij lage lambda waarde HiGH VARIANCE en bij hoge lambda waarde HIGH BIAS. De lambda waardes: (0.64, 1.28, 2.56) geven betere generalization.</p>

<p>Na regularization heb ik de beste waarde gekozen om overfitting en underfitting verminderen.</p>

<p>Hieronder de plot van het finale model met de geselecteerde lambda waarde.</p>
<img src="/notebooks_data/phoneme_boundary_scikit_MLP_diagnos/finalmodel.png" width="550" height="350"/>
<p>In deze plot kunnen we zien dat het model niet overfit of underfit maar generalized is.</p>


<br />
<h2>Extra</h2>

<h3>Dataset genereren voor Jeroen en Erik</h3>

- notebook
  - [transformer mfcc, word, wordtranscription en fonemen](/extra/transformer_audio_word_to_mfcc.md)
  - [transformer word, klanken en fonemen](/extra/transforming_corpus_data.md)

<p>- Dataset MFCC en woord waarvan alleen de woorden die beginnen met "st".</p>
<img src="/extra/stwordlist.png" width="600" height="350"/>
<p>- Dataset MFCC en woord.</p>
<img src="/extra/wordlist.png" width="600" height="350"/>
<p>- Dataset MFCC, woord met klanken en fonemen code lijst.</p>
<img src="/extra/fondecoder.png" width="650" height="350"/>

<h3>Script als onderdeel genereren dataset voor Jeroen en Erik</h3>
<p>- Script fonemen code decoder methode <- (Methode decodeert phonemen naar woord en creert phonemen lijst.)</p>
<p>- Script woorden scraper methode</p>
