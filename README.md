## JobMatcher _ Capstone Project

![streamlit](https://user-images.githubusercontent.com/117051182/219459638-6119911c-23e6-4500-88ca-7826be2c19a5.jpeg)


## PRODUCT OVERVIEW

JobMatcher is a tool that utilizes advanced algoriths and data analysis techniques for matching jobs with candidates based on the information entered in the input. 

## PURPOSE AND BUSINESS USE CASE

#### Talent Acquisition: 
The model used behind JobMatcher tool helps streamline the recruitment process by shortening candidate sourcing time and reducing recruitment costs. It matches job seekers with appropriate job openings through aligning their skills, experiences, and preferences with the requirements and qualifications of available positions. This not only saves time and money for employers but also helps job seekers find the right job opportunities more easily and efficiently.

Possible use cases: 
* Recruiting agency - assist headhunters to locate matched jobs for candidates based on their skillset. 
* Job application process - asks candidates to identify their skill and the system returns recommended jobs based on skillset. 

#### Job Seekers:

JobMatcher provides a platform for exploring new job opportunities and making a career change. It is especially useful for new graduates who are looking to start their careers and need help finding suitable job openings that match their qualifications and interests.

Possible use cases: 
* Career fair - job seekers may enter their skillset to find feasible opportunities at a company within a few secs. 
* School career counseling services - student advisors may use this tool to help students identify the possible careerpath.

#### Workforce Planning:

JobMatcher helps companies retain talent and enhance workforce planning by aligning their existing employees with internal job openings that match their skills, experiences, and preferences and enhancing employee engagement and retention. JobMatcher can help employers identify and promote these internal job opportunities to their existing employees, thereby increasing employee retention and reducing turnover.

Possible use cases: 
* Training and development: Identifying skills gaps - analyze the skills and experiences of existing employees and compare them to the skills and experiences required for internal job openings. This can help employers identify skills gaps and determine where to focus training and development programs to build their workforce's skills and capabilities.
* Restructuring and streamlining workforce planning: identifying the right employees for internal job openings quickly and efficiently. This can save time and resources, reduce hiring costs, and help employers manage their workforce more effectively.

## Inspirations
Inspiration for creating this tool came from my personal experience with job applications, previous employers, and discussions with classmates.

* Marsh McLennan - Ambiguity with Job Titles

![Screen Shot 2023-02-13 at 9 57 50 AM](https://user-images.githubusercontent.com/117051182/219463640-edcdc83f-990f-4125-a53e-4832d23d39e1.jpg)

One experience that inspired me was a job application I submitted to Marsh McLennan for a HR role. Two days later, I found out from a friend that there was another opportunity that could also be suitable for me, yet I had omitted it as the job title was ambiguous. This experience inspired me to create a tool for matching an individual's skills based on the job description rather than just the job title.


* Past Employer 


* New graduates


## STAKEHOLDERS
There are several parties that could benefit from this model:
* Talent Acquisition
* Workforce Planning
* Workday/Oracle/ATS 
* Employees
* Job searchers
* Recruiting Agencies/headhunters

## DATASET EXPLANATION:

#### Data Source

Dataset was download from Kaggle.com <https://www.kaggle.com/datasets/PromptCloudHQ/us-technology-jobs-on-dicecom>
This dataset provides a dataset consisting of more than 20,000 job listings from Dice.com, a well-known US-based technology job board. 
Contains: url, company, type of employment, job description, job location, job title, job posting time, shift, job skills

#### Data Cleaning

The following steps were taken to clean the data:
* Nans were dropped.
* Job skills were concatenated with job descriptions.
* For NLP modeling, lemmatization, removal of stopwords and repetitive words, and converting to lowercase were performed.
* Gensim was utilized to create bigrams from the job descriptions."

#### Challenges with the dataset:

* Unable to get data from glassdoor + linkedin due to cyber security
* Dice is a technology job board - the majority of jobs are tech jobs

## Exploratory Data Analysis

The chart below shows the number of jobs by state:
CA, NY, GA, TX, NJ have the most tech jobs.

![jobs by state](https://user-images.githubusercontent.com/117051182/219465212-ff790915-8d98-4865-b535-c2c9b562c012.png)


Word cloud for jobtitles 

![workcloude title](https://user-images.githubusercontent.com/117051182/219465262-9e1bacbc-59f8-427a-981d-b8c9687c1a6b.png)


Word cloud for job skills

![workdcloud skillset](https://user-images.githubusercontent.com/117051182/219465303-e8b0b5cb-e56d-4ff4-847a-5fb899f6e926.png)

## MODELING

Data was preprocessed using TFIDF to vectorize into a bag of words. 

There were four models used for this project: NLTK NMF, NLTK LDA, GENSIM LDA, GENSIM NMF.

NLTK - NMF was selected as the final model for proceeding to the recommendation stage
The data was converted into a W and H matrix of topics and documents.
Visualization of the 10 topics generated from the NMF algorithm is shown below.

![10 topcs distribution](https://user-images.githubusercontent.com/117051182/219465623-a3747142-4990-48fe-81bf-c4a4ebc51345.png)


As you can see here that the java web developer have the highest weights amongst all the topics.

Visualization using TSNE to examine the unsupervised clustering of the datapoints. 

![scatterplot of job classes](https://user-images.githubusercontent.com/117051182/219465860-054d9c04-3a48-4dc9-b216-f83e93871503.png)


## RECOMMENDATION SYSTEM

Cosine similarity recommendation system was used to produce the final result. 

For testing, the keywords entered in the system are : 
human resource, data analytics, NLP, machine learning, data science, tableau, data visualization

The system generates both HR and data anayltics jobs:

![Screen Shot 2023-02-13 at 9 57 50 AM](https://user-images.githubusercontent.com/117051182/219466811-8a452c30-c219-44f3-b823-95f57554555c.jpg)


## NEXT STEPS

#### DATASET
Find better data source to update the dataset return with up-to-date results. Depending on the usage of the model, the dataset could be updated to all the jobs within a company and build the model for internal use.

#### EXPLORATION
Exploring projects for possible applications of the model - example: build the recommendation system in Workday.

#### Sponsor
Connect with designated sponsors to deploy the model - Workday system, workforce planning.

JobMatcher is deployed as a software application through Streamlit. Link to the JobMatcher tool: https://t36yang-jobmatcher-capstone-stream-lit-nlp-j9a064.streamlit.app/
