from keras.models import load_model
import pandas as pd
model=load_model('model2.h5')
path="/content/drive/My Drive/2test12.csv"
testdata= pd.read_csv(path)
result=model.predict_classes(testdata)
for i in range(len(result)):
  if result[0][0]<0.5:
     prediction='intruder'
  else:
     prediction='genuine'
print(result[0][0],"certainty of being a",prediction)