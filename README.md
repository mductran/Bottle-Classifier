# realsense-cam

# Pickle training data file to load X_train and y_train

file_Name = "training-data.p"
with open(file_Name, mode='rb') as f:
    train = pickle.load(f)
    
X_train, y_train = train['Images'], train['labels']
