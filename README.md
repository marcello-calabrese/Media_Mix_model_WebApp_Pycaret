# Media_Mix_model_WebApp_Pycaret

** Web Link of the APP online: https://share.streamlit.io/marcello-calabrese/media_mix_model_webapp_pycaret/main/app.py

- This app predicts sales on media mix spend for different media channels. 
- Before the prediction, we apply a saturated spending function to the marketing spend vector.
- The machine learning model used is the ExtraTreeRegression model.

## The dataset media spend variables are: 

- tv_sponsorships 
- tv_cricket
- tv_RON	
- radio
- NPP
- Magazines
- OOH
- Social
- Programmatic
- Display_Rest
- Search	
- Native


**Dataset file of unseen data in the repository:** unseen_data.csv

Machine Learning Package Used: Pycaret, link: https://pycaret.org/

## What is Saturated Spending:

### Let's start with a practical example: 

We assume that the more money you spend on advertising, the higher your sales get. 
However, the increase gets weaker the more we spend. For example, increasing the TV spends from 0 € to 100,000 € 
increases our sales a lot, but increasing it from 100,000,000 € to 100,100,000 € 
does not do that much anymore. This is called a saturation effector theeffect of diminishing returns.

Increasing the amount of advertising increases the percent of the audience reached by the 
advertising, hence increases demand, but a linear increase in the advertising exposure doesn’t have 
a similar linear effect on demand.

**Typically** each incremental amount of 
advertising causes a progressively 
lesser effect on demand increase.
This is advertising saturation. 
**Usually Digital display ads and digital advertising in general have a high saturation effect, 
meanwhile TV, Radio have a low saturation effect.**
