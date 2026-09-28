## AI-Based Movie Recommendation System

### Project Overview

The AI-Based Movie Recommendation System is a Python based program that recommends movies based on their historical ratings similarity. The system employs item-based collaborative filtering and cosine similarity to find movies that have similar rating patterns. The graphical user interface (GUI) is designed using Tkinter and offers an interactive and simple search for movies and recommendations.

### Project Objectives

The main goal is to show how AI and machine learning can be effectively used in personalized movie recommendations. The system is designed to analyze movie rating information, discover similarities, make good suggestions, and display the results in a user-friendly fashion.

### Technologies Used

The most common programming language used for applications is Python. Both pandas and NumPy provide the ability to load, clean and perform matrix operations on data. Scikit-learn contains the cosine similarity calculator and Tkinter is used for making the graph user interface. The MovieLens Latest Small dataset provides ratings and movie data.

### Dataset Description

The MovieLens Latest Small dataset includes 9,742 movies and 100,836 ratings on them from users. The dataset consists of two CSV files: movies.csv, which includes movie identifiers, titles, and genres, and ratings.csv, which contains user identifiers, movie identifiers, ratings, and timestamps. These records contain some of the data necessary to determine relationships between movies.

### System Features

Application includes the movie title input and a number of recommendations to choose. The results table shows movie titles, movie genre, average movie ratings, count of ratings, and similarity scores. There are also other settings such as explanations for movie recommendations, a reset button, input validation and informative messages for missing or unavailable movie names.

### Recommendation Methodology

The system uses item-based collaborative filtering to find movie association based on users' rating patterns. The rating records are used to build a user-movie matrix and the cosine similarity between movie rating vectors is computed. The higher the similarity score, the more similar the movies are, based on the rating information provided. The average ratings and number of ratings will give you some context of the recommended films.

### Installation and Setup

Python must be installed before running the application. Installation of required libraries can be done by using the command below:

pip install pandas numpy scikit-learn

Tkinter comes pre-loaded with many common Python distributions. The MovieLens Latest Small must be downloaded and movies.csv and ratings.csv must be located in the directory where the application is run. Update file paths in the dataset if it is saved in a new location.

### Execution Instructions

The application can be launched from a terminal or command prompt using the following command:

python movie_recommendation.py

The application window is displayed when the data set is loaded and the similarity matrix is ready. A movie title can be inputted and the necessary number of recommendations can be chosen, and the recommendation button can be used to show the results.

### Testing and Results

Toy Story (1995) was used to test the application and ten movie recommendations were successfully generated. The results included Toy Story 2 (1999), Jurassic Park (1993), and Independence Day (a.k.a. ID4) (1996). Other tests were carried out for blank input and a movie title not included in the data set. In both situations, suitable warning messages were there; in both cases the application was still running and could be used for further input.

### Limitations

Historical rating information that is available and consistent is key to the quality of recommendations. Less similar scores may be generated for movies that have fewer ratings. Correct spelling may also be necessary for exact title matching, as well as the addition of the release year. The current one is for movies and not for individual user profiles and real-time viewing behavior.

### Future Enhancements

Suggestions for further enhancements include personalised recommendations, partial title matching, the incorporation of movie descriptions and genres, and creation of a web-based interface. Other evaluation criteria may also be added to evaluate the accuracy of the recommendations and overall system performance.

### Dataset Acknowledgment

The MovieLens Latest Small dataset from GroupLens Research is employed in the project. The data set should be used for research and education and should be used in compliance with the terms and conditions applied to it.
