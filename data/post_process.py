import os
from os.path import join as opj

import pandas as pd

from sklearn.preprocessing import LabelEncoder


def post_processing(args):

    if args.train=='True':
      data=pd.read_csv(args.path_to_train_data)
    else:
      data=pd.read_csv(args.path_to_test_data)

    data['text_obj']=data['text_obj'].replace('[NAN]','')
    data['text_mthd']=data['text_mthd'].replace('[NAN]','')
    data['text_deal']=data['text_deal'].replace('[NAN]','')

    data=data.fillna('')
    data['text']=data['text_obj']+' '+data['text_mthd']+' '+data['text_deal']
    data['label']=data['digit_1']+'-'+data['digit_2'].apply(str)+'-'+data['digit_3'].apply(str)

    if args.train=='True':
      data=data.drop([231666,278838,342781]).reset_index(drop=True)

      encoder1 = LabelEncoder()
      encoder1.fit(data['digit_1'])
      data['label_1']=encoder1.transform(data['digit_1'])

      encoder2 = LabelEncoder()
      encoder2.fit(data['digit_2'])
      data['label_2']=encoder2.transform(data['digit_2'])

      encoder3 = LabelEncoder()
      encoder3.fit(data['digit_3'])
      data['label_3']=encoder3.transform(data['digit_3'])

      return data, encoder3

    else:
      return data