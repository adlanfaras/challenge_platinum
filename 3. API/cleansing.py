import pandas as pd
import re


df_alay = pd.read_csv('new_kamusalay.csv', encoding='ISO-8859-1', header=None)
df_alay = df_alay.rename(columns={0: 'alay', 1: 'formal'}) 
df_alay_dict = dict(zip(df_alay['alay'], df_alay['formal']))

def case_folding (text):
    return text.lower()

def clean (text):
    clean1 = re.sub ('\\n','', text)
    clean2 = re.sub ('RT',' ', clean1)
    clean3 = re.sub ('USER', ' ', clean2)
    clean4 = re.sub ('(http|https):\/\/s+', ' ', clean3)
    clean5 = re.sub ('[^0-9a-zA-Z]+', ' ', clean4)
    clean6 = re.sub ('x[a-z0-9]{2}', ' ', clean5)
    clean7 = re.sub ("\d+", ' ', clean6)
    clean8 = re.sub ('  +', '', clean7)
    clean9 = re.sub ('user', ' ', clean8)
    return clean9

def tokenization(text):
    text = re.split('\W+', text)
    return text

def alay_normalization(text):
    newlist = []
    for word in text:
        if word in df_alay_dict:
            text = df_alay_dict[word]
            newlist.append(text)
        else:
            text = word
            newlist.append(text)
    return newlist

#stopwords
stopword_list = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                 'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua', 'ia',
                  'ia', 'seperti', 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'kepada', 'oleh',
                  'saat', 'sementara', 'setelah', 'kami', 'sekitar', 'bagi', 'serta', 'di', 'dari', 'telah',
                  'sebagai', 'masih', 'hal', 'ketika', 'adalah', 'itu', 'dalam', 'bahwa', 'atau', 'kita', 'dengan',
                  'akan', 'juga', 'ada', 'mereka', 'sudah', 'saya', 'terhadap', 'secara', 'agar', 'lain', 'anda',
                  'kalo', 'amp', 'biar', 'bikin', 'bilang', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya','kayak', 'i', 'a',
                  'gak', 'ga', 'krn', 'nya', 'nih', 'sih','jd', 'jgn', 'sdh', 'aja', 'n', 't', 'gue', 'yah', 
                 'begitu', 'mengapa', 'kenapa', 'yaitu', 'yakni', 'daripada', 'itulah', 'lagi', 'maka', 'tentang',
                  'demi', 'dimana', 'kemana', 'pula', 'sambil', 'sebelum', 'sesudah', 'supaya', 'guna', 'kah', 'pun',
                  'sampai', 'sedangkan', 'selagi', 'sementara', 'tetapi', 'apakah', 'kecuali', 'sebab', 'seolah', 'seraya',
                  'seterusnya', 'dsb', 'dst', 'dll', 'dahulu', 'dulunya', 'anu', 'demikian', 'mari', 'nanti', 'oh', 'ok',
                  'setiap', 'sesuatu','saja', 'toh', 'walau', 'amat', 'apalagi', 'dengan', 'bahwa', 'oleh']

stopword_list = set(stopword_list)

def remove_stopwords(text):
    text = [word for word in text if word not in stopword_list]
    return text

def clean_non_existed(text):
    if text == '':
        return None
    else:
        return text

#jadikan satu fungsi cleansing
def cleansing_all(text):
    text = case_folding(text)
    text = clean(text)
    text = tokenization(text)
    text = alay_normalization(text)
    text = remove_stopwords(text)
    text = ' '.join(text)
    text = clean_non_existed(text)
    return text
