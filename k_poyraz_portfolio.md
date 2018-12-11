<h1>Datascience kb-74: project Aphasia 2018</h1>

<h2>Courses</h2>

- DataCamp
  - [DataCamp certifications](https://github.com/ciCciC/Aphasia-portfolio/blob/master/datacamp/datacamp_certifications.md)
- Coursera
  - [Coursera](https://github.com/ciCciC/Aphasia-portfolio/blob/master/coursera/results.md)

<h2>Domain knowledge</h2>
Hieronder staat per onderwerp beschreven de uitgevoerde onderzoeken, gebruikte technieken, verwijzingen naar literatuur en resultaten.

<h3>- Aphasia (afasie)</h3>
<p>Hier wordt een vooronderzoek gedaan naar afasie. Dit is van belang voor het opbouwen van kennis over afasie. Voor het opbouwen van kennis over Afasie heb ik gebruik gemaakt van de <b>techniek desk-research en interview</b>. Daarbij komen de <b>onderzoeksstrategie BIEB en VELD</b> bij kijken. Bij de literatuur wordt verwezen naar samenvattingen die ik van mijn desk-research heb gemaakt met daarin de referenties naar de bronnen:</p>

- Literatuur
  - [Desk-research naar Afasie](https://drive.google.com/open?id=1XC5KO49hhVlRnTzpUgk5_EsWqkBjdQA_)
  
Om de opgestelde onderzoekvragen te kunnen beantwoorden heb ik  aantal documenten over fonologie op het internet geraadpleegd. Onderzoek  gaat over de definitie van fonologie, het proces, hoe een spraak nauwkeurig gemaakt kan worden en welke programma’s gaan om met fonologie.

- Literatuur
  - [Desk-research naar Fonologie](https://drive.google.com/open?id=1eQMhui_E9tXWjDe0CW03YpHo1Rr4H6cb)
  
Na het lezen van gebruik van fonetiek in wetenschappelijke artikelen over speech to text systemen vond ik het handig om een desk-research naar te doen om te kunnen begrijpen wat fonetiek betekent.
- Literatuur
  - [Desk-research naar Fonetiek](https://drive.google.com/open?id=1NetEeGGN6kJM-wjqDAOdOYDvPhFIOtFv)
  
  
<p>Interview</p>
Bij de interviews was mijn taak niet alleen het stellen van vragen maar ook het opnemen van de interviews. Dit heb ik gedaan door gebruik te maken van de voicerecorder applicatie op mijn telefoon. Zodat we later de opnames nogmaals kunnen naluisteren voor verduidelijking van de gesprekken.
Daarnaast was mijn taak om met Doortje samen een tweede interview te houden bij Rijndam Instituut met mevrouw Ineke (opdrachtgever) en de security manager, die gaat over AVG, over het krijgen van benodigde audio data en de veiligheid van de data. Het gesprek over AVG was van belang voor het gebruik kunnen maken van de Google Services. Voornamelijk de Google Text to Speech en Cloud Storage services. Dit was in eerste instantie van belang voor het z.s.m. kunnen omzetten van de afasie audiobestanden naar tekst.

- Literatuur
  - [Security and Privacy Considerations](https://cloud.google.com/storage/docs/gsutil/addlhelp/SecurityandPrivacyConsiderations)

<h3>- API Aphasia met Google Services</h3>
<p>Deze API heb ik ontwikkeld om het proces van audio bestanden op een snelle manier te kunnen omzetten naar tekst. Anders moest dat proces handmatig moeten worden gedaan wat veel tijd kost. Daarnaast heeft deze API ook als functie om de timestamps van per woord in een audio signaal te kunnen krijgen. Dit was van belang om een dataset te kunnen creëren voor toekomstig gebruik bijv. voor een neurale netwerk.</p>

Om dit te kunnen realiseren heb ik een project aangemaakt in GitHub genaamd "Aphasia-project". Daarnaast heb ik dit project gekoppeld aan de Google Services met mijn eigen Credentials. Ook heb ik een installatie guide opgesteld voor mijn projectgenoten zodat zij gebruik konden maken van de API.
  - [Github - Aphasia-project Repository](https://github.com/ciCciC/Aphasia-project)

Om een overzicht te kunnen krijgen over de bestaande Speech to Text services heb ik een desk-research naar gedaan. Ik ben tot conclusie gekomen dat er services bestaan van grote bedrijven die de Nederlandse taal niet ondersteunen behalve Google.
- Literatuur
  - [Bestaande Speech to Text services](https://drive.google.com/open?id=1odo6bqDnnt94Juf_-Ih-7UYFU-NXov1VFIhr93GRMb8)
  
Om de Speech to Text van Google te kunnen koppelen aan mijn API heb ik de volgende literatuur geraadpleegd.
- Literatuur
  - [Google Speech to Text documentatie](https://cloud.google.com/speech-to-text/docs/)

Google kent aantal regels als het komt tot transformeren van audio signaal naar tekst. Men (zonder gebruik van Cloud Storage) mag niet audio langer dan 1 minuut meegeven. Aangezien wij audio bestanden hebben die langer dan een minuut zijn moest er een ander oplossing voor komen. De oplossing was een Cloud Storage service aanzetten en die koppelen aan de Aphasia API. Dit geeft de vrijheid van audio langer dan een minuut te kunnen transformeren naar tekst.

Om de Cloud Storage van Google te kunnen koppelen aan mijn API heb ik de volgende literatuur geraadpleegd.
- Literatuur
  - [Google Cloud Storage documentatie](https://cloud.google.com/storage/docs/)

<p>Aphasia API architectuur</p>
<img src="https://lh4.googleusercontent.com/4t4Zh6dmZxZIALZan5IRzNyZfJYVA3vqyFOTShG-2KzUt_TmyVznqyNQ44A0xbQESXQnyOq0kvZT599hkM5o=w1920-h969"
alt="drawing" width="400" height="500"/>
