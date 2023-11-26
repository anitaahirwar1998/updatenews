import datetime
import gspread
import pandas as pd
import multiprocessing
import time
import nltk
import requests
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import re
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Define a function to remove HTML tags from text
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

# Define a function to remove URLs from text
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

# Define a function to remove punctuation from text
def remove_punctuation(text):
    exclude = string.punctuation
    return text.translate(str.maketrans('', '', exclude))

# Define a function to remove stopwords from text
def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word not in stopwords.words('english'):
            new_text.append(word)
    return ' '.join(new_text)

# Initialize Google Sheets
creds = ServiceAccountCredentials.from_json_keyfile_name('./gssep-399015-22a26cd898e7.json')
client = gspread.authorize(creds)
gs = client.open('Stock Database Sep23')
sheet_pos = gs.worksheet('Positive_Words')
sheet_neg = gs.worksheet('Negative_Words')
all_record_pos = sheet_pos.get_all_records()
pos_df = pd.DataFrame(all_record_pos)
all_record_neg = sheet_neg.get_all_records()
neg_df = pd.DataFrame(all_record_neg)

# Initialize the Sentiment Intensity Analyzer
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment of a text
def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound'] * 100

# Function to calculate positivity score
def calculate_positivity_score(positive_count, negative_count):
    total_count = positive_count + negative_count
    positivity_score = (positive_count - negative_count) / total_count * 100 if total_count != 0 else 0
    return round(positivity_score, 2)

# Function to predict stock sentiment
def predict_stock_sentiment(sentence):
    try:
        clean_text = remove_punctuation(remove_html_tags(remove_url(sentence.lower())))
        words = remove_stopwords(clean_text)

        positive_keywords = [WordNetLemmatizer().lemmatize(remove_punctuation(remove_html_tags(word.lower())), pos='v') for word in pos_df['words']]
        negative_keywords = [WordNetLemmatizer().lemmatize(remove_punctuation(remove_html_tags(word.lower())), pos='v') for word in neg_df['words']]
        positive_count = 0
        negative_count = 0

        lemmatizer = WordNetLemmatizer()

        for word in words.split():
            try:
                lemma = lemmatizer.lemmatize(word, pos='v')
                cleaned_word = remove_punctuation(remove_html_tags(word.lower()))
                if any(keyword.lower() in cleaned_word for keyword in positive_keywords):
                    positive_count += 1
            except Exception as e:
                print(f"Error processing word: {e}")

        for phrase in negative_keywords:
            try:
                if phrase.lower() in sentence.lower():
                    negative_count += 1
            except Exception as e:
                print(f"Error processing negative phrase: {e}")

        for phrase in positive_keywords:
            try:
                if phrase.lower() in sentence.lower():
                    positive_count += 1
            except Exception as e:
                print(f"Error processing positive phrase: {e}")

        positivity_score = calculate_positivity_score(positive_count, negative_count)

        return positivity_score

    except Exception as e:
        print(f"Error in predict_stock_sentiment: {e}")
        return 0  # Return a neutral score or handle the error as needed

# Function to shorten a URL
def short_link(link):
    endpoint = 'http://tinyurl.com/api-create.php'
    long_url = link
    params = {'url': long_url}
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        short_url = response.text
        return short_url
    else:
        return 'Error: HTTP'

# Function to fetch news data for a given stock URL
def get_newsbyticker(url, stock="N/A"):
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
def get_news_contain(url,channel="Times of India"):
    import requests
    from bs4 import BeautifulSoup
    import json
    import re
    try:
      res = requests.get(url)
      soup = BeautifulSoup(res.text, features="html.parser")
      if channel in("Times of India","Economic Times"):
        script_tag=soup.find_all("script",{"type":"application/ld+json"})[1]
        script_data = script_tag.string.strip()
        data = json.loads(script_data)
        return data.get("articleBody")
      if channel in("Bloomberg Quint"):
        if len(soup.find_all("script",{"type":"application/ld+json"}))<3:
          script_tag=soup.find_all("script",{"type":"application/ld+json"})[1]
          script_data = script_tag.string.strip()
          data = json.loads(script_data)
          return data.get("articleBody")
        else:
          script_tag=soup.find_all("script",{"type":"application/ld+json"})[2]
          script_data = script_tag.string.strip()
          data = json.loads(script_data)
          return data.get("articleBody")
      if channel in("News18"):
        script_tag=soup.find_all("script",{"type":"application/ld+json"})[2]
        script_data = script_tag.string.strip()
        data = json.loads(script_data)
        return data.get("articleBody")
      if channel=="Moneycontrol":
        script_elements = soup.find_all("script", {"type": "application/ld+json"})
        script_text = script_elements[2].text.strip()
        script_text = script_text.replace('<script type="application/ld+json">', '').replace('</script>', '')
        script_text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", script_text)
        data = json.loads(script_text)
        article_body = data[0].get("articleBody", "")
        return article_body
      if channel in("The Hindu Businessline","The Hindu"):
        art_text=soup.find("div",{"itemprop":"articleBody"}).text
        if "\ncomments\n" in art_text.lower():
          return art_text[:art_text.lower().index("\ncomments\n")]
        else:
          return art_text
      if channel in("Business Today"):
        art_text=soup.find("div",{"class":"story-with-main-sec"}).text
        if "also read" in art_text.lower():
          return art_text[:art_text.lower().index("also read")]
        else:
          return art_text
      if channel in("Investing.com"):
        art_text=soup.find("section",{"class":"article-item-content"}).text
        if "original post" in art_text.lower():
          return art_text[:art_text.lower().index("original post")]
        else:
          return art_text
      else:
        return None
    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}")
        return None

    except (ValueError, IndexError, AttributeError, KeyError) as e:
        print(f"Error processing the page: {e}")
        return None
# Function to fetch news content in parallel
def fetch_news_content(row):
    url = row["URL"]
    channel = row["Channel"]
    content = get_news_contain(url, channel)
    return content
def fetch_news_data(row):
    name = row["Name"]
    url = row["URL"]
    df = get_newsbyticker(url, name)
    return df
# Main function
def main():
    # Initialize Google Sheets
    creds = ServiceAccountCredentials.from_json_keyfile_name('./gssep-399015-22a26cd898e7.json')
    client = gspread.authorize(creds)
    gs = client.open('Stock Database Sep23')

    # Fetch data from the "Raw" worksheet
    sheet = gs.worksheet("Raw")
    sheet_all_record = sheet.get_all_records()
    sheet_df = pd.DataFrame(sheet_all_record)

    # Filter out rows with empty URLs
    sort_df = sheet_df[["Name", "URL"]].copy()
    filtered_df = sort_df[sort_df["URL"] != ''].copy()

    # Start fetching news data in parallel
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
    print(f"Total execution time for fetching news data: {execution_time} seconds")
    print(f"Shape of the final news data dataframe: {final_df.shape}")
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    filtered_df_last_30_days = final_df[final_df['Date'] >= thirty_days_ago.date()].reset_index(drop=True)
    sorted_30days = filtered_df_last_30_days[['Stock', 'Date', 'Time', 'Title', 'URL', 'Channel']].copy()
    seven_days_ago = datetime.datetime.now() - datetime.timedelta(days=7)
    filtered_df_last_7_days = sorted_30days[sorted_30days['Date'] >= seven_days_ago.date()].reset_index(drop=True)
    start_time = time.time()
    if __name__ == "__main__":
        pool = multiprocessing.Pool(processes=12)
        content_rows = filtered_df_last_7_days.to_dict(orient='records')
        news_content_list = pool.map(fetch_news_content, content_rows)
        pool.close()
        pool.join()
        filtered_df_last_7_days["Content"] = news_content_list
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time for fetching news content: {execution_time} seconds")
    filtered_df_last_7_days["Deep Score"] = filtered_df_last_7_days['Content'].apply(predict_stock_sentiment)
    filtered_df_last_7_days['Normal Score'] = filtered_df_last_7_days['Title'].apply(analyze_sentiment)
    filtered_df_last_7_days=filtered_df_last_7_days[['Stock', 'Date', 'Time', 'Title', 'URL', 'Channel','Normal Score',"Deep Score"]].copy()
    filtered_df_last_7_days_sort=filtered_df_last_7_days.sort_values(by="Date",ascending=False).copy()
    out_gs = client.open('News Database Sep 2023')
    out_sheet = out_gs.worksheet("News<30Days")
    out_sheet.clear()
    set_with_dataframe(out_sheet, sorted_30days)

    out_sheet1 = out_gs.worksheet("News<7Days")
    out_sheet1.clear()
    set_with_dataframe(out_sheet1, filtered_df_last_7_days_sort)

if __name__ == "__main__":
    main()
