<h1>Datascience portfolio Koray Poyraz</h1>
<p>Minor: Datascience kb-74</p>
<p>Project: Aphasia</p>
<br />
<h2>Courses</h2>

- DataCamp
  - [DataCamp certifications](https://github.com/ciCciC/Aphasia-portfolio/blob/master/datacamp/datacamp_certifications.md)
- Coursera
  - [Coursera](https://github.com/ciCciC/Aphasia-portfolio/blob/master/coursera/results.md)

<br />
<h2>Communication</h2>

- Presentations
- Summaries
- Paper
<br />

<h2>Domain knowledge</h2>
Hieronder staat per onderwerp beschreven de uitgevoerde onderzoeken, gebruikte technieken, verwijzingen naar literatuur en resultaten.

<h3>- Vooronderzoek Aphasia (afasie)</h3>
<p>Hier wordt een vooronderzoek gedaan naar afasie. Dit is van belang voor het opbouwen van kennis over afasie. Voor het opbouwen van kennis over Afasie heb ik gebruik gemaakt van de <b>techniek desk-research en interview</b>. Daarbij komen de <b>onderzoeksstrategie BIEB en VELD</b> bij kijken. Bij de literatuur wordt verwezen naar samenvattingen die ik van mijn desk-research heb gemaakt met daarin de referenties naar de bronnen:</p>

<b>Desk-research (BIEB)</b>
- Literatuur
  - [Desk-research naar Afasie](https://drive.google.com/open?id=1XC5KO49hhVlRnTzpUgk5_EsWqkBjdQA_)
  
Om de opgestelde onderzoekvragen te kunnen beantwoorden heb ik  aantal documenten over fonologie op het internet geraadpleegd. Onderzoek  gaat over de definitie van fonologie, het proces, hoe een spraak nauwkeurig gemaakt kan worden en welke programma’s gaan om met fonologie.

- Literatuur
  - [Desk-research naar Fonologie](https://drive.google.com/open?id=1eQMhui_E9tXWjDe0CW03YpHo1Rr4H6cb)
  
Na het lezen van gebruik van fonetiek in wetenschappelijke artikelen over speech to text systemen vond ik het handig om een desk-research naar te doen om te kunnen begrijpen wat fonetiek betekent.
- Literatuur
  - [Desk-research naar Fonetiek](https://drive.google.com/open?id=1NetEeGGN6kJM-wjqDAOdOYDvPhFIOtFv)
  
  
<b>Interview (VELD) <br /></b>
Bij de interviews was mijn taak niet alleen het stellen van vragen maar ook het opnemen van de interviews. Dit heb ik gedaan door gebruik te maken van de voicerecorder applicatie op mijn telefoon. Zodat we later de opnames nogmaals kunnen naluisteren voor verduidelijking van de gesprekken.
Daarnaast was mijn taak om met Doortje samen een tweede interview te houden bij Rijndam Instituut met mevrouw Ineke (opdrachtgever) en de security manager, die gaat over AVG, over het krijgen van benodigde audio data en de veiligheid van de data. Het gesprek over AVG was van belang voor het gebruik kunnen maken van de Google Services. Voornamelijk de Google Text to Speech en Cloud Storage services. Dit was in eerste instantie van belang voor het z.s.m. kunnen omzetten van de afasie audiobestanden naar tekst.

- Literatuur
  - [Security and Privacy Considerations](https://cloud.google.com/storage/docs/gsutil/addlhelp/SecurityandPrivacyConsiderations)

<h3>- Belangrijke artikelen voor Speech Recognition systemen</h3>
Deze artikelen hebben mij geholpen om een beeld te krijgen om dit project te kunnen aanpakken. Bijv. de feature extraction technieken die worden toegepast of een pipeline die wordt gehanteerd voor Speech Recognition systemen. Ook gaf deze informatie mij de gelegenheid een idee voor te leggen aan mijn projectgroep om de juiste richting op te gaan.

- Literatuur
  - Geeft een beeld over de technieken die worden toegepast (op minder technisch niveau)
  - [A Review on Speech Recognition Technique (wetenschappelijke)](https://pdfs.semanticscholar.org/1062/8132a34301f66a0af4bc485f05e3988cdc44.pdf)
  - Geeft de stappen weer die worden genomen om de features te kunnen extracten van de audiosignalen (op technisch niveau)
  - [Speech Processing for Machine Learning MFCCs (wetenschappelijke)](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
  - Een GitHub repository met handige informatie over bestaande repositories voor een onderdeel van Speech Recognition systemen
  - [awesome-python-scientific-audio](https://github.com/faroit/awesome-python-scientific-audio#feature-extraction)
  - DTW (Dynamic Time Warping) wordt toegepast om woord signalen met elkaar te kunnen vergelijken. Hiervoor heb ik een desk-research gedaan en een samenvatting over geschreven.
  - [Onderzoek DTW algoritme](https://drive.google.com/open?id=1LXNcv708e6wNzxt1yUf-5IvGmn7w8j28)

<br />
<br />
<h2>Data Collection</h2>
<p><b>"Data collection"</b> is van belang voor de etappe <b>"Data Preperation"</b>. Hieronder staan onderwerpen van taken die zijn verricht voor het verzamelen en structureel opslaan van de data voor het project. Elk onderwerp kan bestaan uit een desk-research en notebooks voor het uitvoeren van een bepaalde taak of taken die relevant zijn voor het project. Ik heb data verzamelt uit verschillende bronnen. De bronnen zijn <b>"VoxForge" en "Uva"</b>.</p>

<h3>- Uva data</h3>
<p>fon.hum.uva is een website waar een gratis database wordt aangeboden met daarin gesproken audio bestanden met daarbij horende teksten, <a href="http://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFAcorpus/">link naar de website</a>. Voor Uva heb ik een scraper geschreven om de data van hun website af te halen. De reden voor de scraper is omdat de database niet te downloaden is dus moet men op hun website per klik downloaden en dat kost veel tijd.</p>

- Notebook
  - [Scraper Uva Data notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/ScrapingDataUva/ScrapingDataUva.md)

<h3>- VoxForge data</h3>
<p>VoxForge is een website waar gesproken audio bestanden met daarbij horende teksten gratis worden aangeboden. Voor VoxForge heb ik ook een scraper geschreven om de data van hun website af te halen, want zoals bij Uva gebeurt downloaden per klik en dat kost veel tijd. Vandaar de scraper ook voor VoxForge. <a href="http://www.voxforge.org/home/downloads/speech/dutch">link naar de website</a>.</p>

- Notebook
  - [Scraper VoxForge Data notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/ScrapingDataVoxforge/ScrapingDataVoxforge.md)

<h3>- CORPUS data</h3>
<p>CORPUS is een grote data bestaande uit Nederlands gesproken audio met gerelateerd woorden. Deze data is gratis verkrijgbaar op <a href="https://ivdnt.org/nieuws">deze link</a>. De data is gedownload en opgeslagen op de server door onze projectbegeleider. Voor het transformeren van de data naar de gewenste structuur om er mee aan de slag te kunnen heb ik een transformer geschreven. Die staat beschreven onder "Data Preperation".</p>
  
<h3>Extra - mappenstructuur voor verzamelde data</h3>
<p>Als extra heb ik een methode geschreven die een mappenstructuur op de server creert voor de verzamelde data. Dit geeft de mogelijkheid om makkelijk data op te slaan, verwerken en aanmaken. Dit maakt de paden naar de data overzichtelijk dus een beter overzicht waar wat staat op de server.</p>

- Notebook
  - [Initialiseren mappenstructuur notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/initialize_directory_structure/initialize_directory_structure.md)

<br />
<br />
<h2>Data Preperation</h2>
<p>Hieronder staan onderwerpen van taken die zijn verricht voor het voorbereiden van data voor het project. Elk onderwerp kan bestaan uit een desk-research en notebooks voor het uitvoeren van een bepaalde taak of taken die relevant zijn voor het project. </p>

<h3>- Ontwikkelen API Aphasia met Google Services</h3>
<p>Deze API heb ik ontwikkeld om het proces van audio bestanden op een snelle manier te kunnen omzetten naar tekst. Anders moest dat proces handmatig moeten worden gedaan wat veel tijd kost. Daarnaast heeft deze API ook als functie om de timestamps van per woord in een audio signaal te kunnen krijgen. Dit was van belang om een dataset te kunnen creëren voor toekomstig gebruik bijv. voor een neurale netwerk.</p>

Om dit te kunnen realiseren heb ik een project aangemaakt in GitHub genaamd "Aphasia-project". Daarnaast heb ik dit project gekoppeld aan de Google Services met mijn eigen Credentials. Ook heb ik een installatie guide opgesteld voor mijn projectgenoten zodat zij gebruik konden maken van de API.
- Aphasia-project Github
  - [Aphasia-project Repository](https://github.com/ciCciC/Aphasia-project)

Om een overzicht te kunnen krijgen over de bestaande Speech to Text services heb ik een desk-research naar gedaan. Ik ben tot conclusie gekomen dat er services bestaan van grote bedrijven die de Nederlandse taal niet ondersteunen behalve Google.
- Literatuur
  - [Bestaande Speech to Text services](https://drive.google.com/open?id=1odo6bqDnnt94Juf_-Ih-7UYFU-NXov1VFIhr93GRMb8)
  
Om de Speech to Text van Google te kunnen koppelen aan mijn API heb ik de volgende literatuur geraadpleegd.
- Literatuur
  - [Google Speech to Text documentatie](https://cloud.google.com/speech-to-text/docs/)

Google kent aantal regels als het komt tot transformeren van audio signaal naar tekst. Men (zonder gebruik van Cloud Storage) mag niet audio langer dan 1 minuut meegeven. Aangezien wij audio bestanden hebben die langer dan een minuut zijn moest er een ander oplossing voor komen. 

<b>Eerste oplossing</b> was, als we niet Cloud Storage mochten gebruiken vanwege AVG, een functie implementeren die een audio snijdt in minuten rekening houdend met niet het snijden door een woordsignaal. Deze funtie heb ik geimplementeerd om audio bestanden te kunnen snijden binnen 1 minuut en transformeren naar tekst.
Snij functie:
- [Method slicing audio PNG](https://drive.google.com/open?id=16UPK4XQozjz5NT5cblhc8xMPL29ilMy9)
- [Method slicing audio SCRIPT](https://github.com/ciCciC/Aphasia-project/blob/master/AudioTranscribe.py)

<b>Tweede oplossing</b> was een Cloud Storage service aanzetten en die koppelen aan de Aphasia API. Dit geeft de vrijheid van audio langer dan een minuut te kunnen transformeren naar tekst.

Om de Cloud Storage van Google te kunnen koppelen aan mijn API heb ik de volgende literatuur geraadpleegd.
- Literatuur
  - [Google Cloud Storage documentatie](https://cloud.google.com/storage/docs/)
  
De architectuur van Aphasia API:
- [Aphasia API architectuur PNG](https://drive.google.com/open?id=1G1ckCQ-MElPZKq9lQn3mzqChB0-xHwtU)

<h3>- Ontwikkelen STT (Speech to Text) timestamps generator in notebooks</h3>
<p>Bij dit onderwerp heb ik de Aphasia API omgezet in een notebook met extra functies om een batch te kunnen uitvoeren op de <b>data collectie "Voxforge"</b> om een map vol met audio bestanden te kunnen transformeren naar woord timestamps en het creëren van CSV bestanden als dataset. Deze datasets bestaat uit kolommen "begin", "end", "word" en "audiopath" die uiteindelijk gebruikt zal worden bij de "Phoneme boundary generator". Zie notebook voor verdere informatie.</p>

Ik heb eerst een desk-research gedaan naar bestaande tools die uit een audio signaal de woord timestamps kunnen uithalen, zie "desk-research".
- Desk-research
  - [bestaande word alignment tools voor timestamps](https://drive.google.com/open?id=1-HS5edq61a1NErzNLFjARwHTNYibYAg6)

- Notebook
  - [stt timestamps generator notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/stt_timestamps_generator%20/stt_timestamps_generator.md)

<h3>- Ontwikkelen Alignment in notebooks</h3>
<p>Voor dit project heb ik een aligner script in notebook ontwikkeld. Ontwikkeling van de aligner was van belang voornamelijk voor het kunnen trainen van SPHINX (een kant en klare Speech to Text tool). Deze aligner is voornamelijk bestemd voor de data collectie van de "UVA" omdat de zinnen niet zijn alignt.</p>

<p>Voor het kiezen van een goede bibliotheek voor het alignen van zinnen heb ik eerst een desk-research gedaan. Ik ben toen de bibliotheek "Aeneas" tegengekomen welke vaak wordt gebruikt voor alignen van zinnen.</p>

- Aeneas documentatie
  - [Aeneas documentation](https://www.readbeyond.it/aeneas/docs/)

- Aeneas bibliotheek
  - [Aeneas library](https://www.readbeyond.it/aeneas/)

- Notebook
  - [aligner uva data notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/aligner_uva_data/aligner_uva_data.md)

<h3>- Ontwikkelen transformer CORPUS data in notebooks</h3>
<p>Voor de data van CORPUS heb ik een transformer geschreven die de data van CORPUS naar de gewenste structuur transformeert bestaande uit kolommen "begin", "end", "word" en "audiopath" en als CSV bestand opslaat. Deze data wordt gebruikt bij de Phoneme Boundary Generator. Zie notebook voor verdere info. Deze uiteindelijke data wordt overigens ook gebruikt door mijn projectgenoten. Dus ik heb het niet alleen voor mezelf gedaan maar ook voor mijn projectgenoten.</p>

- Notebook
  - [transforming CORPUS data notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/transforming_corpus_data/transforming_corpus_data.md)

<h3>- Ontwikkelen Phoneme boundary generator in notebooks</h3>
<p>Na "Data Collection" en "Data Preperation" onderwerpen, die de data in een gewenste structuur hebben gezet, heb ik een Phoneme Boundary Generator ontwikkeld. Wat deze generator doet is het genereren van foneem grenzen als data door de laatste N milliseconden van een woord en begin N milliseconden van het volgende woord samen te voegen. Deze dataset is om een <b>Phoneme Boundary Classifier</b> te kunnen trainen.</p>

<p>Ik heb twee soorten generators ontwikkeld. De "V2" slaat de samengevoegde N milliseconden op, zoals hierboven beschreven, en "V3" slaat alleen het verschillen tussen de laatste N milliseconden van een woord en begin N milliseconden van het volgende woord op. Hiermee wil ik dus kijken welke een betere validation acc. en recall score levert. Voor verdere informatie over deze versies zie de notebooks.</p>

Voor feature extraction van de audio signalen dus het verkrijgen van de MFCCs heb ik gebruik gemaakt van de bibliotheek en bron:
- Bron
  - [Speech Processing for Machine Learning MFCCs (wetenschappelijke)](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)

- Bibliotheek
  - [python speech features library](https://github.com/jameslyons/python_speech_features)

- Notebook
  - [Phoneme boundary generator v2](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_generator_v2/phoneme_boundary_generator_v2.md)
  - [Phoneme boundary generator v3](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_generator_v3/phoneme_boundary_generator_v3.md)

<br />
<h2>Predictive Models</h2>
Om een Phoneme Boundary Classifier model te kunnen trainen met de verzamelde Nederlands gesproken CORPUS data heb ik aantal machine en deep learning modellen uitgeprobeerd. Daarvoor heb ik de volgende modellen gebruikt:

- Machine learning
  - Random Forest Classifier

- Deep learning
  - MLP (Multi Layer Perceptron)
  - Bi-LSTM (bi-directional Long Short-Term Memory)
  
Voor sommige van de bovenstaande modellen heb ik gebruik gemaakt van Scikit-Learn en Tensorflow Core bibliotheek.

Een reden voor het gebruik maken van de Tensorflow Core is meer aanpas mogelijkheden zoals selectie van de GPU cores, toepassing van activation function per neurale netwerk laag en hij is meer geschikt voor het ontwikkelen van deep learning netwerken.

Deze modellen zijn getraind met de data die is gegenereerd door de <b>Phoneme Boundary Generator (CORPUS NL)</b> voor het ontwikkelen van een Phoneme Boundary Classifier.

- Random Forest Classifier
  - [phoneme boundary random forest classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_random_forest_classifier/phoneme_boundary_random_forest_classifier.md)
  
- MLP classifier (Multi Layer Perceptron)
  - [MLP notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_generator_v3/phoneme_boundary_generator_v3.md)
  
- Bi-LSTM classifier
  - [Bi-LSTM notebook](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_generator_v3/phoneme_boundary_generator_v3.md)

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
<h2>Evaluation</h2>

<p>Bij dit onderdeel heb ik voor elk model eerst "oversampling" uitgevoerd om de verhouding tussen de label 0 en 1 te verbeteren waardoor ze gebalanceerd zijn. Hiervoor heb ik de functie "generateMoreData()" geschreven, zie de notebook per model.</p>

<h3>Random Forest Classifier</h3>

- notebook
  - [phoneme boundary random forest classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_random_forest_classifier/phoneme_boundary_random_forest_classifier.md)
  
<p>Een "model selection". Hier kijk ik naar welke waarde het beste kan worden gebruikt bij de hyperparameters "max depth" en "estimators"</p>
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
<p>Bij de plots van hierboven zien we dat er vrij weinig verschil is na 8 estimators. Zelfs bij 100 estimators. De lijn van train en test accuracy liggen niet ver van elkaar af. Dit geeft aan dat er geen sprake is van under- of overfitting. De estimator 32 geeft de hoogste test accuracy score.</p>

<p>Na deze selectie van de waarde voor "max depth" en "estimators" heb ik het model getraind met de volledige datasets.</p>

- dataset: df_nondifference
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/nondiff.png" width="600" height="300"/>

- dataset: df_difference
<img src="/notebooks_data/phoneme_boundary_random_forest_classifier/diff.png" width="600" height="300"/>

<p>Uit deze resultaten zien we dat de dataset zonder verschil dus "df_nondifference" met Random Forest Classifier een betere validation acc en RECALL geeft op class 1 dan bij de dataset met verschil.</p>

<h3>MLP Classifier</h3>

- notebook
  - [phoneme boundary scikit MLP classifier](https://github.com/ciCciC/Aphasia-portfolio/blob/master/notebooks_data/phoneme_boundary_random_forest_classifier/phoneme_boundary_random_forest_classifier.md)

<p>Een "model selection". Hier kijk ik naar welke waarde het beste kan worden gebruikt bij de hyperparameters "num neurons", "learning rate" en "num layers"</p>
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
<h2>Diagnostics of the learning process</h2>

