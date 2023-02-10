import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = [4, 110, 92, 0, 0, 37.6, 0.191, 130]

# changing the input data into numpy array
input_data_as_array = np.asarray(input_data)

# reshaping the array as we are predicting for one instance
input_data_reshape = input_data_as_array.reshape(1, -1)

# predicting the label
prediction = loaded_model.predict(input_data_reshape)
# print(prediction)

if prediction == 0:
    print('The person is not Diabetic')
else:
    print('The person is Diabetic')
