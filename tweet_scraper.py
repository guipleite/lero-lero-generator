import sys
import pandas as pd
from Scweet.scweet import scrape


def get_tweets(start_date, end_date, account=None, words=None):
    """
    Returns a pandas dataframe with tweets.

    Keyword arguments:
    start_date -- oldest tweet date to search for, must be in YYYY-MM-DD format
    start_date -- newest tweet date to search for, must be in YYYY-MM-DD format
    account -- account to scrape tweets from, if is None will search for tweets of any account (default None)
    words -- list with word to search for, if it is empty will not won't search for any specific (default None)
    """

    print("Getting tweets ...")
    if words is None:
        data = scrape(
            since=start_date,
            until=end_date,
            from_account=account,
            interval=10,
            display_type="Top",
            save_images=False,
        )
    else:
        data = scrape(
            words=words,
            since=start_date,
            until=end_date,
            from_account=account,
            interval=1,
            headless=True,
            display_type="Top",
            save_images=False,
            resume=False,
            filter_replies=True,
            proximity=True,
        )
    tweets_df = pd.DataFrame(data)
    tweets_df.to_csv('./data/tweets.csv')
    return tweets_df


if __name__ == "__main__":

    data = get_tweets(
        start_date=sys.argv[1],
        max_date=sys.argv[2],
        account=sys.argv[3],
        words=sys.argv[4],
    )

    print(data)