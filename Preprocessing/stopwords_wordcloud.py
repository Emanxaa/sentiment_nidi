from wordcloud import STOPWORDS

custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    'yang', 'dan', 'untuk', 'ini', 'itu', 'di', 'ke', 'dari',
    'dengan', 'pada', 'adalah', 'akan', 'sudah', 'ada', 'juga',
    'tidak', 'bisa', 'lebih', 'kita', 'saya', 'kami', 'mereka',
    'atau', 'tapi', 'karena', 'jadi', 'lagi', 'banyak', 'dalam',
    'ya', 'ga', 'yg', 'rb', 'com', 'tampilkan', 'membalas',
    'nya', 'ber', 'ter', 'me', 'kan', 'lah', 'pun', 'ku',
    'si', 'sang', 'para', 'namun', 'walau', 'meski',
    'https', 'http', 'co', 'pic', 'twitter',

    # lokasi
    'sumut', 'sumbar', 'sumsel',
    'riau', 'jambi', 'lampung',

    # kata terlalu umum
    'banjir', 'warga', 'korban',
    'air', 'hujan', 'wilayah'
])
