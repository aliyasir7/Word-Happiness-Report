## WORLD HAPPINESS REPORT ##
PROJE = "World Happiness Report"

""" 1- Importing Libraries and Packages """
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statistics as st
import os
import functions as f

path="C:\\Users\\hseym\\OneDrive\\Masaüstü\\Yeni klasör\\sample data and codes\\World_Happiness_Report"
os.chdir(path)
f.display()
data_2015 = pd.read_csv("2015.csv")
data_2016 = pd.read_csv("2016.csv")
data_2017 = pd.read_csv("2017.csv")
data_2018 = pd.read_csv("2018.csv")
data_2019 = pd.read_csv("2019.csv")
data_2015.head()
data_2016.head()
data_2017.head()
data_2018.head()
data_2019.head()

#because of they have same columns, data was split up as 2015-2016-207 and 2018-2019.
""" 2015-2016-2017 """
data_2015.columns = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score', 'Standard_Error',
                     'Economy_GDP_per_Capita', 'Family', 'Health_Life_Expectancy','Freedom',
                     'Trust_Government_Corruption', 'Generosity', 'Dystopia_Residual']
data_2016.columns = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score', 'Lower_Confidence _Interval',
                     'Upper Confidence Interval','Economy_GDP_per_Capita', 'Family', 'Health_Life_Expectancy',
                     'Freedom', 'Trust_Government_Corruption', 'Generosity','Dystopia_Residual']
data_2017.columns = ["Country", 'Happiness_Rank', 'Happiness_Score', 'Whisker_high', 'Whisker_low',
                     'Economy_GDP_per_Capita', 'Family','Health_Life_Expectancy', 'Freedom', 'Generosity',
                     'Trust_Government_Corruption', 'Dystopia_Residual']
data_2018.columns = ['Happiness_Rank', 'Country', 'Happiness_Score','GDP_per_capita','Social_support',
                     'Healthy_life_expectancy','Freedom_to_make_life_choices','Generosity','Perceptions_of_corruption']
data_2019.columns = ['Happiness_Rank', 'Country', 'Happiness_Score','GDP_per_capita','Social_support',
                     'Healthy_life_expectancy','Freedom_to_make_life_choices','Generosity','Perceptions_of_corruption']

## According to six important figure top 50 country for every year
def most_happy_country(data,year):
    if year in (2015,2016,2017):
        col=['Economy_GDP_per_Capita','Health_Life_Expectancy','Freedom','Trust_Government_Corruption','Generosity','Family']
    else:
        col=['GDP_per_capita','Healthy_life_expectancy','Freedom_to_make_life_choices','Perceptions_of_corruption','Generosity','Social_support']
    rich_set = set(data[["Country", col[0]]].sort_values(col[0],ascending = False).reset_index(
        drop = True).head(50)["Country"])
    healty_set = set(data[["Country", col[1]]].sort_values(col[1],ascending = False).reset_index(
        drop = True).head(50)["Country"])
    free_set = set(data[["Country", col[2]]].sort_values(col[2], ascending = False).reset_index(
        drop = True).head(50)["Country"])
    trusty_set = set(data[["Country", col[3]]].sort_values(col[3],ascending = False).reset_index(
        drop = True).head(50)["Country"])
    generosity_set = set(data[["Country", col[4]]].sort_values(col[4], ascending = False).reset_index(
        drop = True).head(50)["Country"])
    family_set = set(data[["Country", col[5]]].sort_values(col[5], ascending = False).reset_index(
            drop = True).head(50)["Country"])
    top_set = set.intersection(rich_set, healty_set, free_set, trusty_set, generosity_set, family_set)
    top_country = pd.DataFrame({"Country": list(top_set)})
    vars()[f"top_country_{year}"] = pd.merge(data[["Country", "Happiness_Rank", "Happiness_Score"]], top_country, how = "inner",
                               on = "Country")
    return vars()[f"top_country_{year}"]

top_country_2015 = most_happy_country(data_2015,2015)
top_country_2016 = most_happy_country(data_2016,2016)
top_country_2017 = most_happy_country(data_2017,2017)
top_country_2018 = most_happy_country(data_2018,2018)
top_country_2019 = most_happy_country(data_2019,2019)

plt.figure(figsize = (10,8))
sns.barplot(x=top_country_2015["Happiness_Score"], y=top_country_2015.Country)
plt.ylabel("Countries")
plt.title("Most Happy Country in 2015")

plt.figure(figsize = (10,8))
sns.barplot(x=top_country_2016["Happiness_Score"], y=top_country_2016.Country)
plt.ylabel("Countries")
plt.title("Most Happy Country in 2016")

plt.figure(figsize = (10,8))
sns.barplot(x=top_country_2017["Happiness_Score"], y=top_country_2017.Country)
plt.ylabel("Countries")
plt.title("Most Happy Country in 2017")

plt.figure(figsize = (10,8))
sns.barplot(x=top_country_2018["Happiness_Score"], y=top_country_2018.Country)
plt.ylabel("Countries")
plt.title("Most Happy Country in 2018")

plt.figure(figsize = (10,8))
sns.barplot(x=top_country_2019["Happiness_Score"], y=top_country_2019.Country)
plt.ylabel("Countries")
plt.title("Most Happy Country in 2019")

## Region analyse
all_top = pd.concat([top_country_2015,top_country_2016,top_country_2017,top_country_2018,top_country_2019], ignore_index = True)
all_top = all_top[["Country"]].drop_duplicates().reset_index(drop = True)
all_top.columns=["Country"]
all_top = pd.merge(all_top, data_2015[["Country","Region"]], how = "inner", on="Country")

plt.figure(figsize = (8,5.5))
cmap = plt.get_cmap("tab20c")
colors = cmap(np.array([1, 7, 14]))
plt.pie(all_top["Region"].value_counts(),labels =all_top["Region"].value_counts().index ,autopct='%1.2f%%',colors=colors)
plt.title("Regions of Top Happy Country in 2015-2019")

plt.figure(figsize=(10,7.5))
sns.stripplot(data=all_top,x='Region',y='Country')
plt.xticks(rotation=90)
plt.tight_layout()

## Corelations between six important figure for every year
plt.figure(figsize=(12,10))
sns.heatmap(data_2015.corr(),annot=True,fmt=".2f",linewidth=1.5)
plt.figure(figsize=(12,10))
sns.heatmap(data_2016.corr(),annot=True,fmt=".2f",linewidth=1.5)
plt.figure(figsize=(12,10))
sns.heatmap(data_2017.corr(),annot=True,fmt=".2f",linewidth=1.5)
plt.figure(figsize=(12,10))
sns.heatmap(data_2018.corr(),annot=True,fmt=".2f",linewidth=1.5)
plt.figure(figsize=(12,10))
sns.heatmap(data_2019.corr(),annot=True,fmt=".2f",linewidth=1.5)


## yıllara göre mutluluk skoru ortalması değişimi
sc_mean15=data_2015['Happiness_Score'].mean()
sc_mean16=data_2016['Happiness_Score'].mean()
sc_mean17=data_2017['Happiness_Score'].mean()
sc_mean18=data_2018['Happiness_Score'].mean()
sc_mean19=data_2019['Happiness_Score'].mean()
scores =np.array([sc_mean15,sc_mean16,sc_mean17,sc_mean18,sc_mean19])
xx=np.arange(2015,2020,1)
yy=scores
plt.figure(figsize=(10,5))
plt.grid(True)
plt.title("Hapiness Score")
plt.xlabel("Year")
plt.ylabel("Hapiness")
plt.xticks(xx)
plt.plot(xx,yy)

## yıllara göre GDP ortalması değişimi
GDP15=data_2015["Economy_GDP_per_Capita"].mean()
GDP16=data_2016["Economy_GDP_per_Capita"].mean()
GDP17=data_2017["Economy_GDP_per_Capita"].mean()
GDP18=data_2018["GDP_per_capita"].mean()
GDP19=data_2019["GDP_per_capita"].mean()
GDP=np.array([GDP15,GDP16,GDP17,GDP18,GDP19])
x=np.arange(2015,2020,1)
y=GDP
plt.plot(x,y)
plt.xticks(x)
plt.grid(True)
plt.title('Year GDP')
plt.ylabel('GDP')
plt.xlabel('Year')

## yıllara göre family ortalması değişimi
fam15=data_2015['Family'].mean()
fam16=data_2016['Family'].mean()
fam17=data_2017['Family'].mean()
fam18=data_2018['Social_support'].mean()
fam19=data_2019['Social_support'].mean()
famscore=np.array([fam15,fam16,fam17,fam18,fam19])
famy=famscore
famx=np.arange(2015,2020,1)
plt.plot(famx,famy)
plt.title('Year Family')
plt.ylabel('Family support')
plt.xlabel('Year')
plt.xticks(famx)
plt.grid(True)

## yıllara göre healty ortalması değişimi
healty15=data_2015['Health_Life_Expectancy'].mean()
healty16=data_2016['Health_Life_Expectancy'].mean()
healty17=data_2017['Health_Life_Expectancy'].mean()
healty18=data_2018['Healthy_life_expectancy'].mean()
healty19=data_2019['Healthy_life_expectancy'].mean()
helscore=np.array([healty15,healty16,healty17,healty18,healty19])
healty=helscore
helx=np.arange(2015,2020,1)
plt.plot(helx,healty)
plt.title('Year health')
plt.ylabel('health expectancy')
plt.xlabel('Year')
plt.xticks(helx)
plt.grid(True)

""" dramatic changes between the years"""
differ_2015_2016 = data_2015[data_2015["Happiness_Score"].notna()][["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2015"})
differ_2015_2016 = pd.merge(differ_2015_2016, data_2016[["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2016"}), how = "outer", on="Country")
differ_2015_2016.info()
differ_2015_2016.dropna(inplace = True)
differ_2015_2016["2015-2016"] = differ_2015_2016["Happiness_Score_2015"]-differ_2015_2016["Happiness_Score_2016"]
differ_2015_2016 = differ_2015_2016.sort_values("2015-2016", ascending = False).reset_index(drop=True)
differ_2015_2016.head()
differ_2015_2016.tail()

differ_2016_2017 = data_2016[data_2016["Happiness_Score"].notna()][["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2016"})
differ_2016_2017 = pd.merge(differ_2016_2017, data_2017[["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2017"}), how = "outer", on="Country")
differ_2016_2017.info()
differ_2016_2017.dropna(inplace = True)
differ_2016_2017["2016-2017"] = differ_2016_2017["Happiness_Score_2016"]-differ_2016_2017["Happiness_Score_2017"]
differ_2016_2017 = differ_2016_2017.sort_values("2016-2017", ascending = False).reset_index(drop=True)
differ_2016_2017.head()
differ_2016_2017.tail()

differ_2017_2018 = data_2017[data_2017["Happiness_Score"].notna()][["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2017"})
differ_2017_2018 = pd.merge(differ_2017_2018, data_2018[["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2018"}), how = "outer", on="Country")
differ_2017_2018.info()
differ_2017_2018.dropna(inplace = True)
differ_2017_2018["2017-2018"] = differ_2017_2018["Happiness_Score_2017"]-differ_2017_2018["Happiness_Score_2018"]
differ_2017_2018 = differ_2017_2018.sort_values("2017-2018", ascending = False).reset_index(drop=True)
differ_2017_2018.head()
differ_2017_2018.tail()

differ_2018_2019 = data_2018[data_2018["Happiness_Score"].notna()][["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2018"})
differ_2018_2019 = pd.merge(differ_2018_2019, data_2019[["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2019"}), how = "outer", on="Country")
differ_2018_2019.info()
differ_2018_2019.dropna(inplace = True)
differ_2018_2019["2018-2019"] = differ_2018_2019["Happiness_Score_2018"]-differ_2018_2019["Happiness_Score_2019"]
differ_2018_2019 = differ_2018_2019.sort_values("2018-2019", ascending = False).reset_index(drop=True)
differ_2018_2019.head()
differ_2018_2019.tail()

differ_2015_2019 = data_2015[data_2015["Happiness_Score"].notna()][["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2015"})
differ_2015_2019 = pd.merge(differ_2015_2019, data_2019[["Country","Happiness_Score"]].rename(columns = {"Happiness_Score":"Happiness_Score_2019"}), how = "outer", on="Country")
differ_2015_2019.info()
differ_2015_2019.dropna(inplace = True)
differ_2015_2019["2015-2019"] = differ_2015_2019["Happiness_Score_2015"]-differ_2015_2019["Happiness_Score_2019"]
differ_2015_2019 = differ_2015_2019.sort_values("2015-2019", ascending = False).reset_index(drop=True)
differ_2015_2019.head()
differ_2015_2019.tail()

""" for Venezuela """
vene_2015 = data_2015[data_2015["Country"]=="Venezuela"].iloc[:,[0,2,3,5,6,7,8,9,10]]
vene_2016 = data_2016[data_2016["Country"]=="Venezuela"].iloc[:,[0,2,3,6,7,8,9,10,11]]
vene_2017 = data_2017[data_2017["Country"]=="Venezuela"].iloc[:,[0,1,2,5,6,7,8,9,10]]
vene_2017 = vene_2017[['Country', 'Happiness_Rank', 'Happiness_Score', 'Economy_GDP_per_Capita', 'Family', 'Health_Life_Expectancy', 'Freedom',
       'Trust_Government_Corruption', 'Generosity']]
vene_2018 = data_2018[data_2018["Country"]=="Venezuela"]
vene_2018 = vene_2018[["Country","Happiness_Rank","Happiness_Score","GDP_per_capita","Social_support","Healthy_life_expectancy","Freedom_to_make_life_choices","Perceptions_of_corruption","Generosity"]]
vene_2018.rename(columns = {"GDP_per_capita":"Economy_GDP_per_Capita","Social_support":"Family","Healthy_life_expectancy":"Health_Life_Expectancy",
                  "Freedom_to_make_life_choices":"Freedom","Perceptions_of_corruption":"Trust_Government_Corruption"}, inplace = True)
vene_2019 = data_2019[data_2019["Country"]=="Venezuela"]
vene_2019 = vene_2019[["Country","Happiness_Rank","Happiness_Score","GDP_per_capita","Social_support","Healthy_life_expectancy","Freedom_to_make_life_choices","Perceptions_of_corruption","Generosity"]]
vene_2019.rename(columns = {"GDP_per_capita":"Economy_GDP_per_Capita","Social_support":"Family","Healthy_life_expectancy":"Health_Life_Expectancy",
                  "Freedom_to_make_life_choices":"Freedom","Perceptions_of_corruption":"Trust_Government_Corruption"}, inplace = True)

venezuela = pd.concat([vene_2015,vene_2016,vene_2017, vene_2018,vene_2019], ignore_index = True)
print(venezuela)

eco=venezuela["Economy_GDP_per_Capita"]
ecox=np.arange(2015,2020,1)
plt.plot(ecox,eco)
plt.title('Year~GDP Venezuela')
plt.xlabel("Year")
plt.ylabel("GDP per Capita")
plt.xticks(ecox)
plt.grid(True)

healt=venezuela["Health_Life_Expectancy"]
healtx=np.arange(2015,2020,1)
plt.plot(healtx,healt, color="orange")
plt.title('Year~Health Life Expectancy Venezuela')
plt.xlabel("Year")
plt.ylabel("Health Life Expectancy")
plt.xticks(healtx)
plt.grid(True)

fam=venezuela["Family"]
fx=np.arange(2015,2020,1)
plt.plot(fx,fam, color="red")
plt.title('Year~Social Support Venezuela')
plt.xlabel("Year")
plt.ylabel("Social Support")
plt.xticks(fx)
plt.grid(True)

fre=venezuela["Freedom"]
frex=np.arange(2015,2020,1)
plt.plot(frex,fre, color="green")
plt.title('Year~Freedom to make life choices Venezuela')
plt.xlabel("Year")
plt.ylabel("Freedom to make life choices")
plt.xticks(frex)
plt.grid(True)

trust=venezuela["Trust_Government_Corruption"]
trustx=np.arange(2015,2020,1)
plt.plot(trustx,trust, color="black")
plt.title('Year~Perceptions of corruption Venezuela')
plt.xlabel("Year")
plt.ylabel("Freedom to make life choices")
plt.xticks(trustx)
plt.grid(True)

