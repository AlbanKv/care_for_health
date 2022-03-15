# Why Care for Health?
- In France, there is an **alarming trend** regarding what is called **'medical desertification'**. 
- One of the three key elements of it is the **lack of General Practitioners** (GPs) in the countryside. 
- As training new GPs takes about 10 years...
- ... We investigate another option: **what would be the impact of 'reallocating' GPs throughout the territory?**

_We are a team of 3 in the final sprint of the amazing Le Wagon Data Science bootcamp_

# Our Data Science value proposition
- Looking for an off-the-shelf product within the huge SKlearn library, we could not find something that suited our needs.
- We've thus **developed an algorithm** that would realize the following: 
  => Within a reasonable radius, reallocate surplus of GPs in some cities to 'neighbor' municipalities in deficit
  => Play with parameters such as: 
    How to select GPs | 
    Target and 'rank' municipalities in deficit options | 
    Set a max radius for local reallocation | 
    Minimum number of neighbor cities to target for each reallocation | 
    ...
- Developed in Python using Jupyter Notebooks & VS Code
- Concept of 'Neighbor municipalities' computed using SKLearn library NearestNeihbors
- Algorithm available on the cloud using Uvicorn & FastAPI > stored on a Docker image > uploaded to Google Cloud Platform (Container Registry + Cloud Run)
- Data visualization & online render: using Streamlit > hosted on Heroku

# Bootcamp final project has been presented to public
during 'Le Wagon batch 789 Demo Day' on March, 11th.

# What's next? 
[Work in Progress :)]
