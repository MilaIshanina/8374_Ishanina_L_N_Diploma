import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import heapq
import matplotlib.pyplot as plt
#Данные
user_ids = []
movie_ids = []
ratings = []
timestamps = []
with open("u.data", 'rt') as file1:
    for line in file1.readlines():
        a = line.split()
        user_ids.append(a[0])
        movie_ids.append(a[1])
        ratings.append(a[2])
        timestamps.append(a[3])
rating_df = pd.DataFrame({'user_id': user_ids, 'movie_id': movie_ids, 'rating': ratings, 'timestamp': timestamps})
#rating_df.head()
movie_ids = []
movie_titles = []
release_dates = []

with open("u.item", 'rt', encoding='latin-1') as file2:
    for line in file2.readlines():
        a = line.split("|")
        movie_ids.append(a[0])
        #print(a[1])
        movie_titles.append(a[1])
        release_dates.append(a[2])

item_df = pd.DataFrame({'movie_id': movie_ids, 'movie_title': movie_titles, 'release_date': release_dates})
#item_df.head()

from matplotlib import pyplot as plt
import seaborn as sns
rating_df.groupby('rating').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

rating_df['rating'] = rating_df['rating'].astype(float)
rating_df_agg = rating_df.groupby('movie_id').agg(avg_rating=('rating', 'mean'), count=('rating', 'count')).reset_index()
rating_df_agg['avg_rating'] = np.round(rating_df_agg['avg_rating'], 2)
rating_df_agg_10 = rating_df_agg[rating_df_agg['count']>10]

rating_df_agg_10_merged = pd.merge(rating_df_agg_10, item_df, on='movie_id', how='inner')

rating_df_agg_10_merged.sort_values('avg_rating').tail(10).\
                    plot.barh(x='movie_title', y='avg_rating', label='Avg. rating', figsize=(10,6), color='#42C5FE')
plt.grid(visible=True, color='black')
plt.title("Highly rated movies", fontsize=15)
plt.xlabel("Movie Name", fontsize=10)
plt.ylabel("Avg. rating", fontsize=10)
plt.xlim([0, 6])

#!pip install scikit-surprise
from surprise import Dataset
from surprise import Reader

pandas_df_reader = Reader()
ratings_dataset = Dataset.load_from_df(rating_df[['user_id', 'movie_id', 'rating']], pandas_df_reader)

fullTrainSet = ratings_dataset.build_full_trainset()

from surprise import KNNBasic
from collections import defaultdict

model = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
model.fit(fullTrainSet)
simMatrix = model.compute_similarities()

#print(simMatrix.shape)
#print(simMatrix)

test_customer = '100'
k = 15

test_customer_inner_uid = fullTrainSet.to_inner_uid(test_customer)
#print(test_customer_inner_uid)
simData = simMatrix[test_customer_inner_uid]
#print(simData.shape)

simUsers_without_self = []
for innerID, score in enumerate(simData):
    if (innerID != test_customer_inner_uid):
        simUsers_without_self.append( (innerID, score) )

kNeighbors = heapq.nlargest(k, simUsers_without_self, key=lambda t: t[1])

#print("kNeighbors selected based on cosine similarity are: ")
#print(kNeighbors)

candidates = defaultdict(float)

for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = fullTrainSet.ur[innerID]
    for rating in theirRatings:
        #print(innerID, " ", rating)
        # rating[0] is the movie inner id which becomes the key of candidates dict
        # rating[1] is the rating given
        # The R.H.S is how we calculate score for each rating given
        # If a movie occurs more than once, the scores are added on
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
#print(candidates)

from collections import Counter
from operator import itemgetter
import time
def calculateNeighborPercentage(itemID, kNeighbors, fullTrainSet):
    start_time = time.time()
    neighbor_ratings = []
    for similarUser in kNeighbors:
        innerID = similarUser[0]
        theirRatings = fullTrainSet.ur[innerID]
        neighbor_rating = next((rating[1] for rating in theirRatings if rating[0] == itemID), None)
        if neighbor_rating is not None:
            neighbor_ratings.append(neighbor_rating)
    if neighbor_ratings:
        rating_four_or_five_count = sum(1 for rating in neighbor_ratings if rating >= 4)
        neighbor_rating_percentage = (rating_four_or_five_count / len(neighbor_ratings)) * 100
    else:
        neighbor_rating_percentage = 0
    #print(f"Время, затраченное на выполнение метода calculateNeighborPercentage: {(time.time() - start_time):.6f} сек.")
    return neighbor_rating_percentage


def calculateAllUsersPercentage(itemID, kNeighbors, fullTrainSet):
    start_time = time.time()
    all_ratings = [rating for (_, rating) in fullTrainSet.ir[itemID]]
    all_rating_four_or_five_count = sum(1 for rating in all_ratings if rating >= 4)
    all_rating_percentage = (all_rating_four_or_five_count / len(all_ratings)) * 100
    #print(f"Время, затраченное на выполнение метода calculateAllUsersPercentage: {(time.time() - start_time):.6f} сек.")
    return all_rating_percentage

def calculateAverageRating(itemID, fullTrainSet):
    start_time = time.time()
    all_ratings = [rating for (_, rating) in fullTrainSet.ir[itemID]]
    if not all_ratings:
        return 0
    average_rating = sum(all_ratings) / len(all_ratings)
    #print(f"Время, затраченное на выполнение метода calculateAverageRating: {(time.time() - start_time):.6f} сек.")
    return average_rating


watched = {}
for itemID, rating in fullTrainSet.ur[test_customer_inner_uid]:
    watched[itemID] = 1

def getMovieName(movie_id, item_df):
      movie_name = item_df[item_df['movie_id']==movie_id]['movie_title'].reset_index(drop=True)[0]
      return movie_name

#Функция для получения и печати рейтингов соседей для рекомендованного фильма
def print_neighbor_ratings(itemID, kNeighbors, fullTrainSet):
    neighbor_ratings = []
    print(f"\nNeighbor ratings for movie ID {itemID}:")
    for similarUser in kNeighbors:
        innerID = similarUser[0]
        theirRatings = fullTrainSet.ur[innerID]
        neighbor_rating = next((rating[1] for rating in theirRatings if rating[0] == itemID), None)
        if neighbor_rating is not None and neighbor_rating >= 3:
            print(f"Neighbor {innerID} gave a rating of {neighbor_rating}")
            neighbor_ratings.append(neighbor_rating)
    return neighbor_ratings

#Функция для визуализации рейтингов соседей
def histogramNeighbors(neighbor_ratings, kNeighbors):
        start_time = time.time()
        #l = [f'{similarUser[0]}' for similarUser in kNeighbors[:len(neighbor_ratings)]]
        #print(l)
        plt.figure()
        plt.bar([f'{similarUser[0]}' for similarUser in kNeighbors[:len(neighbor_ratings)]], neighbor_ratings, color='skyblue')
        plt.xlabel('Neighbor ID')
        plt.ylabel('Rating')
        plt.title('Neighbor Ratings')
        plt.show()
        #print(f"Время, затраченное на выполнение метода plot_neighbor_ratings: {(time.time() - start_time):.6f} сек.")


#Функция для визуализации частоты рейтингов
def groupingHistograms(neighbor_ratings):
        start_time = time.time()
        rating_counts = Counter(neighbor_ratings)
        unique_ratings = sorted(rating_counts.keys())
        counts = [rating_counts[rating] for rating in unique_ratings]

        plt.figure()
        plt.bar(unique_ratings, counts, color='skyblue', width=0.4)
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.title('Neighbor Rating Frequencies')
        plt.xlim(0, 5.5)
        plt.show()
        #print(f"Время, затраченное на выполнение метода plot_rating_frequencies: {(time.time() - start_time):.6f} сек.")


#Основная функция для обработки и вывода рекомендованных фильмов
def recommendations(candidates, watched, kNeighbors, fullTrainSet, item_df):
        pos = 0
        for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if itemID not in watched:
                movieID = fullTrainSet.to_raw_iid(itemID)
                movieName = getMovieName(movieID, item_df)
                print("movieID")
                print(movieID)
                print("movieName")
                print(movieName)

                neighbor_ratings = print_neighbor_ratings(itemID, kNeighbors, fullTrainSet)
                histogramNeighbors(neighbor_ratings, kNeighbors)

                if neighbor_ratings:
                    groupingHistograms(neighbor_ratings)
                    neighbor_rating_percentage = calculateNeighborPercentage(itemID, kNeighbors, fullTrainSet)
                    all_rating_percentage = calculateAllUsersPercentage(itemID, kNeighbors, fullTrainSet)
                    all_rating = calculateAverageRating(itemID, fullTrainSet)
                    if neighbor_rating_percentage and all_rating_percentage > 0:
                        print(f"\n {neighbor_rating_percentage:.2f}% пользователей, которые похожи на вас, положительно оценили фильм '{movieName}'")
                        print(f" {all_rating_percentage:.2f}% пользователей положительно оценили фильм '{movieName}'")
                        print(f"Средний рейтинг фильма: {all_rating:.2f}")

                pos += 1
                if pos > 4:
                    break

recommendations(candidates, watched, kNeighbors, fullTrainSet, item_df)

def item_based_recommendations(testUser, k, fullTrainSet, item_df):
    sim_options = {'name': 'cosine', 'user_based': False}
    model = KNNBasic(sim_options=sim_options)
    model.fit(fullTrainSet)
    item_similarity_matrix = model.compute_similarities()
    testUserInnerID = fullTrainSet.to_inner_uid(testUser)
    testUserRatings = fullTrainSet.ur[testUserInnerID]
    kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])
    candidates = defaultdict(float)
    influence_dict = defaultdict(list)
    for itemID, rating in kNeighbors:
        similarityRow = item_similarity_matrix[itemID]
        for innerID, score in enumerate(similarityRow):
            candidates[innerID] += score * (rating / 5.0)
            movieID = fullTrainSet.to_raw_iid(itemID)
            influence_dict[innerID].append((movieID, rating))

    watched = {}
    for itemID, rating in fullTrainSet.ur[testUserInnerID]:
        watched[itemID] = 1

    top_recommended_movies = []
    top_influencing_movies = []

    top = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            top += 1
            if top > 5:
                break
            top_recommended_movies.append(itemID)
            top_influencing_movies.append(influence_dict[itemID]) 
    start_time = time.time()
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 20))
    for i, (recommended_movie_id, influencing_movies_list) in enumerate(zip(top_recommended_movies, top_influencing_movies)):
        recommended_movie_name = item_df[item_df['movie_id'] == fullTrainSet.to_raw_iid(recommended_movie_id)]['movie_title'].reset_index(drop=True)[0]
        axes[i].set_title(f"Recommended Movie: {recommended_movie_name}")
        influences = []
        movie_names = []
        for influ_movie_id, influ_rating in influencing_movies_list:
            influ_movie_name = item_df[item_df['movie_id'] == influ_movie_id]['movie_title'].reset_index(drop=True)[0]
            movie_names.append(influ_movie_name)
            influences.append(influ_rating)
        y_pos = np.arange(len(movie_names))
        axes[i].barh(y_pos, influences, align='center', alpha=0.5)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(movie_names)
        axes[i].invert_yaxis()

    plt.tight_layout()
    plt.show()
    #print(f"Время, затраченное на выполнение метода item_based_recommendations: {(time.time() - start_time):.6f} сек.")
#k=5
item_based_recommendations(test_customer, k, fullTrainSet, item_df)
