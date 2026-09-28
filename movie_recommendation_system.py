# The movie and rating datasets were loaded, cleaned and analyzed to find similarities among movies using user-rating patterns.
# A movie recommendation system was developed with an interactive graphical interface to display recommendations, similarity scores, and explanations.

# importing data analysis libraries
import pandas as pd
import numpy as np
import tkinter as tk

# importing interface components
from tkinter import ttk, messagebox

# importing similarity calculation tools
from sklearn.metrics.pairwise import cosine_similarity

# defining dataset file locations
movies_path = r"E:\ml-latest-small\movies.csv"
ratings_path = r"E:\ml-latest-small\ratings.csv"

# loading movie and rating datasets
try:
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # displaying dataset dimensions
    print("Movie Dataset Shape:", movies.shape)
    print("Ratings Dataset Shape:", ratings.shape)

    # displaying sample movie records
    print("\nMovie Dataset:")
    print(movies.head())

    # displaying sample rating records
    print("\nRatings Dataset:")
    print(ratings.head())

    print("\nDataset loaded successfully.")

# handling missing dataset files
except FileNotFoundError as error:
    raise FileNotFoundError(
        "Dataset files were not found. "
        "Please verify the specified file paths."
    ) from error

# handling other loading errors
except Exception as error:
    raise RuntimeError(
        "An error occurred while loading the dataset."
    ) from error

# removing duplicate movie and rating records
movies = movies.drop_duplicates(subset="movieId")
ratings = ratings.drop_duplicates(subset=["userId", "movieId"])

# removing records with missing values
movies = movies.dropna(subset=["movieId", "title", "genres"])
ratings = ratings.dropna(subset=["userId", "movieId", "rating"])

# converting identification columns into integer format
movies["movieId"] = movies["movieId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)
ratings["userId"] = ratings["userId"].astype(int)

# retaining valid rating values
ratings = ratings[
    (ratings["rating"] >= 0.5) &
    (ratings["rating"] <= 5.0)
].copy()

# identifying valid movie identification numbers
valid_movie_ids = set(movies["movieId"])

# removing ratings for unavailable movies
ratings = ratings[
    ratings["movieId"].isin(valid_movie_ids)
].copy()

# displaying cleaned dataset dimensions
print("\nData preprocessing completed.")
print("Cleaned Movie Dataset Shape:", movies.shape)
print("Cleaned Ratings Dataset Shape:", ratings.shape)

# creating a matrix of user ratings for each movie
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# replacing missing ratings with zero
user_movie_matrix = user_movie_matrix.fillna(0)

# displaying matrix dimensions
print("\nUser-Movie Matrix Shape:")
print(user_movie_matrix.shape)

# arranging movies as rows and users as columns
movie_user_matrix = user_movie_matrix.T

# calculating cosine similarity between movies
similarity_matrix = cosine_similarity(movie_user_matrix)

# organizing similarity scores into a labeled table
movie_similarity = pd.DataFrame(
    similarity_matrix,
    index=movie_user_matrix.index,
    columns=movie_user_matrix.index
)

print("\nMovie similarity matrix created.")

# calculating average ratings and rating counts
movie_statistics = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

# combining movie details with rating statistics
movie_information = movies.merge(
    movie_statistics,
    on="movieId",
    how="left"
)

# replacing missing statistics with zero
movie_information["average_rating"] = (
    movie_information["average_rating"].fillna(0)
)

movie_information["rating_count"] = (
    movie_information["rating_count"].fillna(0)
)

# creating movie title and identification mappings
title_to_id = dict(
    zip(
        movie_information["title"].str.lower().str.strip(),
        movie_information["movieId"]
    )
)

movie_id_to_title = dict(
    zip(
        movie_information["movieId"],
        movie_information["title"]
    )
)

# defining the recommendation process
def recommend_movies(movie_title, number_of_recommendations=10):

    # standardizing the entered movie title
    movie_title = movie_title.strip().lower()

    # checking for empty input
    if not movie_title:
        return None

    # checking whether the movie exists
    if movie_title not in title_to_id:
        return None

    # identifying the selected movie
    selected_movie_id = title_to_id[movie_title]

    # checking whether similarity scores are available
    if selected_movie_id not in movie_similarity.index:
        return None

    # retrieving similarity scores for the selected movie
    similarity_scores = movie_similarity[
        selected_movie_id
    ].drop(
        selected_movie_id,
        errors="ignore"
    )

    # arranging movies by similarity
    similarity_scores = similarity_scores.sort_values(
        ascending=False,
        kind="mergesort"
    )

    # selecting the requested number of recommendations
    top_recommendations = similarity_scores.head(
        number_of_recommendations
    )

    # preparing recommendation results
    recommendations = pd.DataFrame({
        "movieId": top_recommendations.index,
        "similarity": top_recommendations.values
    })

    # adding movie details and rating statistics
    recommendations = recommendations.merge(
        movie_information[
            [
                "movieId",
                "title",
                "genres",
                "average_rating",
                "rating_count"
            ]
        ],
        on="movieId",
        how="left",
        sort=False
    )

    # arranging recommendations by similarity score
    recommendations = recommendations.sort_values(
        by="similarity",
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    # selecting final display columns
    recommendations = recommendations[
        [
            "title",
            "genres",
            "average_rating",
            "rating_count",
            "similarity"
        ]
    ]

    return recommendations

# creating the main application window
root = tk.Tk()

# configuring window appearance
root.title("AI Movie Recommendation System")
root.geometry("1050x700")
root.minsize(850, 600)
root.configure(bg="#20232A")

# defining interface colors
BACKGROUND = "#20232A"
PANEL_BACKGROUND = "#2D333B"
TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#28A745"
SECONDARY_COLOR = "#61AFEF"
STATUS_COLOR = "#D6D6D6"

# configuring table appearance
style = ttk.Style()
style.theme_use("clam")

style.configure(
    "Treeview",
    background="#FFFFFF",
    foreground="#222222",
    rowheight=30,
    fieldbackground="#FFFFFF",
    font=("Arial", 10)
)

style.configure(
    "Treeview.Heading",
    font=("Arial", 10, "bold"),
    background="#D9E2F3",
    foreground="#222222"
)

style.map(
    "Treeview",
    background=[("selected", "#B7D7F0")],
    foreground=[("selected", "#222222")]
)

# creating the header section
header_frame = tk.Frame(root, bg=BACKGROUND)

header_frame.pack(
    fill="x",
    pady=(20, 10)
)

# displaying the application title
heading = tk.Label(
    header_frame,
    text="AI Movie Recommendation System",
    font=("Arial", 24, "bold"),
    bg=BACKGROUND,
    fg=TEXT_COLOR
)

heading.pack()

# displaying the application subtitle
subtitle = tk.Label(
    header_frame,
    text="Discover movies based on similar user-rating patterns",
    font=("Arial", 12),
    bg=BACKGROUND,
    fg=STATUS_COLOR
)

subtitle.pack(pady=5)

# creating the input panel
input_frame = tk.Frame(
    root,
    bg=PANEL_BACKGROUND,
    padx=20,
    pady=20
)

input_frame.pack(
    fill="x",
    padx=25,
    pady=15
)

# adding the movie title label
movie_label = tk.Label(
    input_frame,
    text="Enter Movie Title:",
    font=("Arial", 12, "bold"),
    bg=PANEL_BACKGROUND,
    fg=TEXT_COLOR
)

movie_label.grid(
    row=0,
    column=0,
    sticky="w",
    padx=10,
    pady=10
)

# creating the movie title input field
movie_entry = ttk.Entry(
    input_frame,
    width=45,
    font=("Arial", 12)
)

movie_entry.grid(
    row=0,
    column=1,
    padx=10,
    pady=10
)

# adding the recommendation count label
count_label = tk.Label(
    input_frame,
    text="Number of Recommendations:",
    font=("Arial", 11),
    bg=PANEL_BACKGROUND,
    fg=TEXT_COLOR
)

count_label.grid(
    row=1,
    column=0,
    sticky="w",
    padx=10,
    pady=10
)

# creating the recommendation count selector
count_selector = ttk.Combobox(
    input_frame,
    values=[5, 10, 15, 20],
    width=12,
    state="readonly"
)

# setting the default recommendation count
count_selector.set(10)

count_selector.grid(
    row=1,
    column=1,
    sticky="w",
    padx=10,
    pady=10
)

# displaying the application status
status_label = tk.Label(
    root,
    text="System ready. Enter a movie title to begin.",
    font=("Arial", 10),
    bg=BACKGROUND,
    fg=STATUS_COLOR
)

status_label.pack(pady=5)

# creating the results section
results_frame = tk.Frame(root, bg=BACKGROUND)

results_frame.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=10
)

# defining result table columns
columns = (
    "Movie Title",
    "Genres",
    "Average Rating",
    "Rating Count",
    "Similarity"
)

# creating the recommendation table
result_table = ttk.Treeview(
    results_frame,
    columns=columns,
    show="headings",
    height=12
)

# defining table column widths
column_widths = {
    "Movie Title": 260,
    "Genres": 280,
    "Average Rating": 130,
    "Rating Count": 120,
    "Similarity": 120
}

# configuring table headings and alignment
for column in columns:
    result_table.heading(column, text=column)

    result_table.column(
        column,
        width=column_widths[column],
        anchor="center"
    )

# creating the vertical scrollbar
scrollbar = ttk.Scrollbar(
    results_frame,
    orient="vertical",
    command=result_table.yview
)

# connecting the scrollbar to the results table
result_table.configure(
    yscrollcommand=scrollbar.set
)

# displaying the results table
result_table.pack(
    side="left",
    fill="both",
    expand=True
)

# displaying the scrollbar
scrollbar.pack(
    side="right",
    fill="y"
)

# defining the recommendation display process
def display_recommendations():

    # retrieving the entered movie title
    selected_title = movie_entry.get().strip()

    # validating the movie title input
    if not selected_title:
        messagebox.showwarning(
            "Input Required",
            "Please enter a movie title."
        )

        status_label.config(
            text="Waiting for a valid movie title."
        )

        return

    # validating the recommendation count
    try:
        selected_count = int(count_selector.get())

    except ValueError:
        messagebox.showwarning(
            "Invalid Selection",
            "Please select a valid recommendation count."
        )

        return

    # updating the application status
    status_label.config(
        text="Generating recommendations..."
    )

    root.update_idletasks()

    # generating movie recommendations
    results = recommend_movies(
        selected_title,
        selected_count
    )

    # handling unavailable movies or empty results
    if results is None or results.empty:
        messagebox.showerror(
            "Movie Not Found",
            "The entered movie was not found in the dataset.\n\n"
            "Please check the spelling or include the release year."
        )

        status_label.config(
            text="No recommendations generated."
        )

        return

    # clearing previous recommendations
    for item in result_table.get_children():
        result_table.delete(item)

    # inserting new recommendations into the table
    for _, row in results.iterrows():
        result_table.insert(
            "",
            "end",
            values=(
                row["title"],
                row["genres"],
                round(row["average_rating"], 2),
                int(row["rating_count"]),
                round(row["similarity"], 3)
            )
        )

    # displaying recommendation completion status
    status_label.config(
        text=(
            f"{len(results)} recommendations generated "
            f"for {selected_title}."
        )
    )

# defining the explanation display process
def show_explanation():

    # retrieving the selected recommendation
    selected_items = result_table.selection()

    # checking whether a recommendation is selected
    if not selected_items:
        messagebox.showinfo(
            "Select a Movie",
            "Select a recommended movie from the results table."
        )

        return

    # retrieving selected movie details
    selected_values = result_table.item(
        selected_items[0],
        "values"
    )

    # extracting recommendation information
    recommended_title = selected_values[0]
    similarity_score = selected_values[4]
    average_rating = selected_values[2]

    # preparing the recommendation explanation
    explanation = (
        f"Recommended Movie: {recommended_title}\n\n"
        f"Average Rating: {average_rating}/5\n"
        f"Similarity Score: {similarity_score}\n\n"
        "Reason for Recommendation:\n"
        "This movie was identified because its user-rating "
        "pattern is similar to the movie selected by the user. "
        "The similarity score is calculated using cosine "
        "similarity within an item-based collaborative "
        "filtering approach.\n\n"
        "The recommendation reflects patterns in historical "
        "user ratings and does not guarantee individual preference."
    )

    # displaying the recommendation explanation
    messagebox.showinfo(
        "Recommendation Explanation",
        explanation
    )

# defining the application reset process
def reset_application():

    # clearing the movie title input
    movie_entry.delete(0, tk.END)

    # restoring the default recommendation count
    count_selector.set(10)

    # removing all displayed recommendations
    for item in result_table.get_children():
        result_table.delete(item)

    # restoring the initial status message
    status_label.config(
        text="System reset. Enter a movie title to begin."
    )

# creating the button section
button_frame = tk.Frame(root, bg=BACKGROUND)

button_frame.pack(pady=15)

# creating the recommendation button
recommend_button = tk.Button(
    button_frame,
    text="Get Recommendations",
    command=display_recommendations,
    font=("Arial", 12, "bold"),
    bg=ACCENT_COLOR,
    fg="white",
    activebackground="#218838",
    activeforeground="white",
    padx=20,
    pady=10,
    cursor="hand2"
)

recommend_button.grid(
    row=0,
    column=0,
    padx=10
)

# creating the explanation button
explanation_button = tk.Button(
    button_frame,
    text="Why This Recommendation?",
    command=show_explanation,
    font=("Arial", 11, "bold"),
    bg=SECONDARY_COLOR,
    fg="#20232A",
    activebackground="#4D9BD6",
    padx=15,
    pady=10,
    cursor="hand2"
)

explanation_button.grid(
    row=0,
    column=1,
    padx=10
)

# creating the reset button
reset_button = tk.Button(
    button_frame,
    text="Reset",
    command=reset_application,
    font=("Arial", 11, "bold"),
    bg="#DC3545",
    fg="white",
    activebackground="#C82333",
    activeforeground="white",
    padx=20,
    pady=10,
    cursor="hand2"
)

reset_button.grid(
    row=0,
    column=2,
    padx=10
)

# connecting the Enter key to recommendation generation
movie_entry.bind(
    "<Return>",
    lambda event: display_recommendations()
)

# displaying application readiness
print("\nAI Movie Recommendation System is ready.")

# starting the graphical application
root.mainloop()