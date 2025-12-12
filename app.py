import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#loading the saved files 
df_cleaned = pd.read_csv("D:/Guvi/Projects/Book recommendations/cleaned_data.csv")
df_clustered = pd.read_csv("D:/Guvi/Projects/Book recommendations/clustered_data.csv")
final_emb = np.load("final_embeding.npy")
df_genre = pd.read_csv("D:/Guvi/Projects/Book recommendations/genre_data.csv")

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

    st.subheader("Ratings To Number of Reviews")
    eda_image6 = Image.open("D:/Guvi/Projects/Book recommendations/EDA images/revVsrating.png")
    st.image(eda_image6, use_container_width=True)
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
#model 1 (Author based)

def recommend_books_Author(name, n=5):
    matches = df_clustered[df_clustered["Author"]==name.lower()]
    if len(matches)==0:
        return "Author not in list"
    author_books = df_cleaned[df_cleaned['Author']==name]

    if len(author_books)>=n:
        return author_books.sample(n)[['Book Name','Author','Language','Rating','Price']]
    
    #combine books if there is no enough book for the same author:
    needed = n-len(author_books) 
    other_books = df_cleaned[df_cleaned['Author']!=name]
    filler_books = other_books.sample(needed)[['Book Name','Author','Language','Rating','Price']]

    final_result = pd.concat([
        author_books[['Book Name','Author','Language','Rating','Price']], 
        filler_books
    ], ignore_index=True)
    
    return final_result

#Genre recommendation:
#genre extraction for recommendation:
def clean_genre_list(x):
    if pd.isna(x):
        return []
    return [g.strip().lower() for g in x.split(",")]
    
df_genre["genre_list"] = df_genre["Genres"].apply(clean_genre_list)

all_genres = sorted(
    {g for sublist in df_genre["genre_list"] for g in sublist}
)
def recommend_by_genre(genre, n=5):
    genre = genre.lower()

    
    genre_matches = df_genre[df_genre["genre_list"].apply(lambda x: genre in x)]

    if genre_matches.empty:
        return "No books found for this genre"

    
    genre_book_names = genre_matches["Book Name"].str.lower().tolist()

    
    author_books_cleaned = df_cleaned[df_cleaned["Book Name"].str.lower().isin(genre_book_names)]

   
    if len(author_books_cleaned) >= n:
        return author_books_cleaned.sort_values("Rating", ascending=False).head(n)[
            ["Book Name", "Author", "Genres", "Rating", "Price"]
        ]

   
    needed = n - len(author_books_cleaned)

    other_books = df_cleaned[~df_cleaned["Book Name"].str.lower().isin(genre_book_names)]
    filler = other_books.sample(needed)

   
    final_result = pd.concat([
        author_books_cleaned[["Book Name", "Author", "Genres", "Rating", "Price"]],
        filler[["Book Name", "Author", "Genres", "Rating", "Price"]]
    ], ignore_index=True)

    return final_result

#model 2 cluster based
def Recommend_cluster_based(title, n=5):
    matches = df_clustered[df_clustered["Book Name"]==title.lower()]
    if len(matches)==0:
        return "Book Not Found"
    idx = matches.index[0]
    cluster_Id = df_clustered.loc[idx, "Clusters"]
    clustered_book_idx = df_clustered[df_clustered["Clusters"]==cluster_Id].index
    return df_cleaned.loc[clustered_book_idx].sample(n)[["Book Name","Author","Language","Rating","Price"]]


#Model 3 Hybrid recommendation
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

    mode = st.radio("Choose Recommendation Mode: ",['Book-Based', 'Author-Based', 'Genre-Based'])

    if mode =='Book-Based':
        st.subheader("🔎 Recommendation Based on Book Names")
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

    if mode == 'Author-Based':
        st.subheader("🔎 Recommendation Based on Author Names")  
        name = st.selectbox("Select an Author :", df_cleaned['Author'].tolist())
        if st.button("Recommend Author Books"):
            st.subheader(f"Books by or similar to **{name}**")

            result = recommend_books_Author(name)
            
            if isinstance(result, str):   # "Author not in list"
                st.error(result)
            else:
                st.dataframe(result)

    if mode == 'Genre-Based':
        st.subheader("🔎 Recommendation Based on Genre")  
        genre = st.selectbox("Select an Genre :", all_genres)
        if st.button("Recommend Books by Genre"):
            st.subheader(f"Books by or similar to **{genre}**")

            result = recommend_by_genre(genre)
            
            if isinstance(result, str):   # "Author not in list"
                st.error(result)
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