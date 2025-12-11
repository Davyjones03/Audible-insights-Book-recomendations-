import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#loading the saved files 
df_cleaned = pd.read_csv("D:/Guvi/Projects/Book recommendations/cleaned_data.csv")
df_clustered = pd.read_csv("D:/Guvi/Projects/Book recommendations/clustered_data.csv")
final_emb = np.load("final_embeding.npy")

#laoding the similarity:
sim_matrix = cosine_similarity(final_emb)

#Homepage
def home():
    st.title("Welcome To Audible Book Recommendations")
    welcome_image = Image.open("D:/Guvi/Projects/Book recommendations/Images/Gemini_Generated_Image_643d4q643d4q643d.png")
    st.image(welcome_image, use_container_width=True)

#EDA of the Data:
def EDA():
    st.title("Exploring the Data: Insights and Distributions")
    c1, c2 = st.columns([1,1])
    c3, c4 = st.columns([1,1])

    #Price
    with c1:
        st.subheader("Price Distribution")
        eda_image1 = Image.open("D:/Guvi/Projects/Book recommendations/EDA images/price.png")
        st.image(eda_image1, use_container_width=True)

    #Review:
    with c2:
        st.subheader("Reviews Distribution")
        eda_image2 = Image.open("D:/Guvi/Projects/Book recommendations/EDA images/Reviewa.png")
        st.image(eda_image2, use_container_width=True)

    #Rating:
    with c3:
        st.subheader("Ratings Distribution")
        eda_image3 = Image.open("D:/Guvi/Projects/Book recommendations/EDA images/rating.png")
        st.image(eda_image3, use_container_width=True)

    #Best rank:
    with c4:
        st.subheader("Ratings Distribution")
        eda_image4= Image.open("D:/Guvi/Projects/Book recommendations/EDA images/best rank.png")
        st.image(eda_image4, use_container_width=True)

    #Words: 
    st.subheader("Words Distribution")
    eda_image5 = Image.open("D:/Guvi/Projects/Book recommendations/EDA images/wrod.png")
    st.image(eda_image5, use_container_width=True)

    c5, c6 = st.columns([1,1])
    #Authors:
    with c5:
        st.subheader("Top Authors")
        eda_image5 = Image.open("D:/Guvi/Projects/Book recommendations/EDA images/top author.png")
        st.image(eda_image5, use_container_width=True)
    with c6:
        st.subheader("Top Genres")
        eda_image5 = Image.open("D:/Guvi/Projects/Book recommendations/EDA images/genre.png")
        st.image(eda_image5, use_container_width=True)
#Recommendation Functions:
# Model 1 (Content-Bases)

def recommend_books_name(title, n=5):
    matches = df_clustered[df_clustered["Book Name"] == title.lower()]
    if len(matches) ==0:
        return "Book Not Found"
    idx=matches.index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key= lambda x: x[1], reverse=True)
    top = scores[1:n+1]
    #using cleaned data for recommendations output
    return df_cleaned.iloc[[i[0] for i in top]][["Book Name","Author","Language","Rating","Price"]]

def Recommend_cluster_based(title, n=5):
    matches = df_clustered[df_clustered["Book Name"]==title.lower()]
    if len(matches)==0:
        return "Book Not Found"
    idx = matches.index[0]
    cluster_Id = df_clustered.loc[idx, "Clusters"]
    clustered_book_idx = df_clustered[df_clustered["Clusters"]==cluster_Id].index
    return df_cleaned.loc[clustered_book_idx].sample(n)[["Book Name","Author","Language","Rating","Price"]]

def Hybrid_recommendation(title, n=5):
    matches = df_clustered[df_clustered["Book Name"] == title.lower()]
    if len(matches) ==0:
        return "Book Not found"
    idx = matches.index[0]
    text_sim = sim_matrix[idx]

    score =(
        0.6*text_sim+
        0.2*df_clustered["Rating"].values+
        0.1*df_clustered["Number of Reviews"].values
    )
    
    top_idx = score.argsort()[::-1][1:n+1]
    ##using cleaned data for recommendations output
    return df_cleaned.iloc[top_idx][['Book Name','Author','Language','Rating','Price']]



def Recommendation():
    st.title("Explore Similar Books of Your Like")
    title = st.selectbox("Select a Book:", df_cleaned["Book Name"].tolist())
    rec_type = st.radio("Choose Recommendation Type :",
                            ["Content-Based", "Cluster-Based", "Hybrid"])
    
    if st.button("Recommend Books"):
        st.subheader(f"Recommendation for : **{title}** as follows")

        if rec_type == "Content-Based":
            result = recommend_books_name(title)
        elif rec_type == "Cluster-Based":
            result = Recommend_cluster_based(title)
        elif rec_type == "Hybrid":
            result = Hybrid_recommendation(title)  

        if result is None:
            st.error("Book not Found in the database")
        else:
            st.dataframe(result)      
            


pages = {
    "Home" : home,
    "Data Viz" : EDA,
    "Find Book" : Recommendation
}

#Navigation:
selection  = st.sidebar.radio("Choose a Page :", list(pages.keys()))
if selection:
    pages[selection]()