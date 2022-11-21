import pandas as pd
from io import StringIO
import numpy as np
from jinja2 import Environment, FileSystemLoader


df = pd.read_csv('data/raw/heart_cleveland_upload.csv')

df_describe = df.describe()

output = StringIO()
df.info(buf=output)
df_info = pd.DataFrame(columns=['DF INFO'], data=output.getvalue().split('\n'))

df_target_corr = df.corr()['condition'][:-1]
most_corr_features = df_target_corr[abs(df_target_corr) > 0.3].sort_values(ascending=False)
most_corr_features = pd.DataFrame(np.array(most_corr_features), np.array(most_corr_features.index), )

env = Environment(loader=FileSystemLoader('reports/templates'))

template = env.get_template('report_template.html')

html = template.render(page_title_text='My report',
                       title_text='EDA of dataset from kaggle',
                       text='Hello, welcome to my report!',
                       beg_test='First 5 rows of our data',
                       df=df,
                       info_text='all info about dataset',
                       df_info=df_info,
                       info_text_conc='We see that the dataset is small (297 observations).'
                                      ' There are no gaps in the data. The number of features'
                                      ' is also small - 13.',
                       des_text='Statistics',
                       df_describe=df_describe,
                       des_text_conc='It is worth noting that the data mainly '
                                     'cover people aged: 75% of the observed'
                                     'are older than 48 years. As for gender, it '
                                     'is shifted towards men. You can also notice '
                                     'that the target is distributed fairly balanced,'
                                     ' about half of the people are actually sick.',
                       text_dup='There are no duplicate observations in our dataset.',
                       text_dup_conc='df.duplicated().sum()',
                       dup=df.duplicated().sum(),
                       text_graph="Let's plot graphics",
                       text_ill="Among women, the disease is less common than among men",
                       text_hist="Let's look at the histograms for all numeric features",
                       text_corr="Correlation matrix",
                       text_high_corr="The strongest correlation is observed in the"
                                      " slope and old peak variables",
                       text_most_corr="Let's see which variables are most correlated with the target",
                       most_corr_features=most_corr_features
                       )

with open('eda_report.html', 'w') as f:
    f.write(html)
