from Titanic import *
import csv

test_data = pd.read_csv("/Users/seandoyle/git/TensorflowTesting/titanic/test.csv")
_, X = preProcessData(test_data, False)

model = createModel(len(X[0]))

model.load_weights(checkpoint_path)

predictions = model.predict(X)

print(predictions[:4])

f = open("submission.csv", "w")
writer = csv.writer(f)

header = ["PassengerId", "Survived"]
writer.writerow(header)
currentId = 892
for prediction in predictions:
    if(prediction[0] > 0.5):
        nextLine = [currentId, 1]
    else:
        nextLine = [currentId, 0]
    currentId += 1
    writer.writerow(nextLine)

f.close()
print("All done!")