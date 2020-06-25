import joblib

def weight_rank():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    weight = model.coef_[0].tolist()
    ftrs = vectorizer.get_feature_names()
    weight_ftrs = list(zip(weight, ftrs))
    weight_ftrs.sort()

    print('10 Most Important: \n' + str(weight_ftrs[:10]))
    print('\n10 Least Important: \n' + str(weight_ftrs[:-11:-1]))


if __name__ == '__main__':
    weight_rank()

'''
10 Most Important: 
[(-1.0362164679900348, 'google'), 
 (-1.0293926869361842, 'study'), 
 (-0.8376510923119779, 'facebook'), 
 (-0.7433903836124939, 'ebola'), 
 (-0.7217994609991263, 'mers'), 
 (-0.6884520671290041, 'cases'), 
 (-0.6638454732473879, 'death'), 
 (-0.659393731237185, 'cancer'), 
 (-0.6423247837454914, '2014'), 
 (-0.5972594516021696, 'health')]

10 Least Important: 
[(1.9296417753304298, 'obamacare'), 
(1.5586958767191976, 'billion'), 
(1.5339487132643286, 'pay'), 
(1.4229120067184, 'mcdonald'), 
(1.2642335990977684, 'tax'), 
(1.2336099917092511, 'workers'), 
(1.2024056695101772, 'amazon'), 
(1.094882935888083, 'prices'), 
(1.0671850361532458, 'china'), 
(1.0228215175576776, 'uninsured')]
'''