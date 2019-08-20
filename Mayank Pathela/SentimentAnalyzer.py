import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import csv


def main():
    # twitter api credentials - you need these to gain access to API
    consumer_key = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
    consumer_secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxx'
    access_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    access_token_secret = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

    # instantiate the api
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    # string to search on twitter
    query = 'Samsung'
    # open/create a csv file to append data
    csvFile = open(query + '_data.csv', 'w', encoding='utf-8')
    # use csv Writer
    csvWriter = csv.writer(csvFile)
    # get data from twitter
    tweet_num = 0
    for tweet in tweepy.Cursor(api.search, q=query, count=10000000, lang="en").items(200000):
        if tweet.place is not None:
            try:
                # not entirely necessary but you can inspect what is being written to file
                # print('tweet number: {}'.format(tweet_num),
                #       tweet.text, tweet.place.full_name)
                # write data to csv
                csvWriter.writerow([tweet.created_at,
                                    tweet.user.location,
                                    tweet.user.followers_count,
                                    tweet.user.friends_count,
                                    tweet.text,
                                    tweet.place.bounding_box.coordinates,
                                    tweet.place.full_name,
                                    tweet.place.country,
                                    tweet.place.country_code,
                                    tweet.place.place_type])
                tweet_num += 1

            except Exception:
                pass


if __name__ == "__main__":
    main()
