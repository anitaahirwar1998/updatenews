import datetime
import gspread
from gspread_dataframe import set_with_dataframe
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import multiprocessing
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
creds = ServiceAccountCredentials.from_json_keyfile_name('./gssep-399015-22a26cd898e7.json')
client = gspread.authorize(creds)
gs = client.open('Stock Database Sep23')
sheet_pos=gs.worksheet('Positive_Words')
sheet_neg=gs.worksheet('Negative_Words')
all_record_pos=sheet_pos.get_all_records()
pos_df=pd.DataFrame(all_record_pos)
all_record_neg=sheet_neg.get_all_records()
neg_df=pd.DataFrame(all_record_neg)
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound'] * 100
def calculate_positivity_score(positive_count, negative_count):
    total_count = positive_count + negative_count
    positivity_score = (positive_count - negative_count) / total_count * 100 if total_count != 0 else 0
    return round(positivity_score, 2)
def predict_stock_sentiment(sentence):
    import re

    def remove_html_tags(text):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)

    def remove_url(text):
        pattern = re.compile(r'https?://\S+|www\.\S+')
        return pattern.sub(r'', text)

    import string
    exclude = string.punctuation

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', exclude))

    from nltk.corpus import stopwords

    def remove_stopwords(text):
        new_text = []
        for word in text.split():
            if word not in stopwords.words('english'):
                new_text.append(word)
        return ' '.join(new_text)

    clean_word = remove_punctuation(remove_html_tags(remove_url(sentence.lower())))
    words = remove_stopwords(clean_word)

    positive_keywords = [WordNetLemmatizer().lemmatize(remove_punctuation(remove_html_tags(word.lower())), pos='v') for word in pos_df['words']]
    negative_keywords = [WordNetLemmatizer().lemmatize(remove_punctuation(remove_html_tags(word.lower())), pos='v') for word in neg_df['words']]
    positive_count = 0
    negative_count = 0

    lemmatizer = WordNetLemmatizer()

    for word in words.split():
        lemma = lemmatizer.lemmatize(word, pos='v')
        cleaned_word = remove_punctuation(remove_html_tags(word.lower()))
        if any(keyword.lower() in cleaned_word for keyword in positive_keywords):
            positive_count += 1
    for phrase in negative_keywords:
        if phrase.lower() in sentence.lower():
            negative_count += 1
    for phrase in positive_keywords:
        if phrase.lower() in sentence.lower():
            positive_count += 1

    positivity_score = calculate_positivity_score(positive_count, negative_count)

    return positivity_score
def short_link(link):
    import urllib.parse
    import urllib.request
    endpoint = 'http://tinyurl.com/api-create.php'
    long_url = link
    params = {'url': long_url}
    encoded_params = urllib.parse.urlencode(params).encode('utf-8')
    response = urllib.request.urlopen(endpoint + '?' + encoded_params.decode('utf-8'))
    if response.status == 200:
        short_url = response.read().decode('utf-8')
        return short_url
    else:
        return 'Error: HTTP'
def get_newsbyticker(url, stock="N/A"):
    import datetime
    import requests
    from bs4 import BeautifulSoup
    import json
    import pandas as pd
    def parse_date(date_string):
        return datetime.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ").date()

    def parse_time(date_string):
        parsed_time = datetime.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ").time()
        formatted_time = parsed_time.strftime("%-I:%M%p")
        return formatted_time

    res = requests.get(url)
    soup = BeautifulSoup(res.content, features="html.parser")
    script_tag = soup.find("script", {"id": "__NEXT_DATA__"})

    if script_tag is None:
        print(f"No script tag with id '__NEXT_DATA__' found on {url}")
        return None

    json_data = json.loads(script_tag.string)

    try:
        news_items = json_data["props"]["pageProps"]["securitySummary"]["news"]["items"]
    except (KeyError, IndexError, TypeError):
        print(f"No News found for {stock}.")
        return None

    if not news_items:
        print(f"No News found for {stock}.")
        return None

    t = []
    d = []
    l = []
    c = []
    dic = []

    for item in news_items:
        t.append(item.get("title", None))
        l.append(item.get("link", None))
        d.append(item.get("date", None))
        dic.append(item.get("description", None))
        c.append(item["publisher"]["name"])

    df = pd.DataFrame({"Title": t, "Date Obj": d, "URL": l, "Channel": c, "Description": dic})
    df['Date'] = df['Date Obj'].apply(parse_date)
    df['Time'] = df['Date Obj'].apply(parse_time)
    df["URL"] = df["URL"].apply(short_link)
    df["Stock"] = stock
    sorted_df = df[["Stock", "Date", "Time", "Title", "URL", "Channel", "Description"]].copy()
    return sorted_df
def fetch_news_data(row):
    name = row["Name"]
    url = row["URL"]
    df = get_newsbyticker(url, name)
    return df
def main():
    creds = ServiceAccountCredentials.from_json_keyfile_name('./gssep-399015-22a26cd898e7.json')
    client = gspread.authorize(creds)
    gs = client.open('Stock Database Sep23')
    sheet = gs.worksheet("Raw")
    sheet_all_record = sheet.get_all_records()
    sheet_df = pd.DataFrame(sheet_all_record)
    sort_df = sheet_df[["Name", "URL"]].copy()
    filtered_df = sort_df[sort_df["URL"] != ''].copy()
    start_time = time.time()
    if __name__ == "__main__":
        pool = multiprocessing.Pool(processes=12)
        rows = filtered_df.to_dict(orient='records')
        dfs = pool.map(fetch_news_data, rows)
        pool.close()
        pool.join()
        final_df = pd.concat(dfs, ignore_index=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
    print(final_df.shape)
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    seven_days_ago = datetime.datetime.now() - datetime.timedelta(days=7)
    filtered_df_last_30_days = final_df[final_df['Date'] >= thirty_days_ago.date()].reset_index(drop=True)
    filtered_df_last_30_days["Deep Score"] = filtered_df_last_30_days['Description'].apply(predict_stock_sentiment)
    filtered_df_last_30_days['Normal Score'] = filtered_df_last_30_days['Title'].apply(analyze_sentiment)
    sorted_30days = filtered_df_last_30_days[['Stock', 'Date', 'Time', 'Title', 'URL', 'Channel', "Normal Score", "Deep Score"]].copy()
    filtered_df_last_7_days = sorted_30days[sorted_30days['Date'] >= seven_days_ago.date()].reset_index(drop=True)
    out_gs = client.open('News Database Sep 2023')
    out_sheet = out_gs.worksheet("News<30Days")
    out_sheet.clear()
    set_with_dataframe(out_sheet, sorted_30days)
    out_sheet1 = out_gs.worksheet("News<7Days")
    out_sheet1.clear()
    set_with_dataframe(out_sheet1, filtered_df_last_7_days)

if __name__ == "__main__":
    main()
