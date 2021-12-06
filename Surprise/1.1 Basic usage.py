from surprise import SVD, Dataset, evaluate, Reader, BaselineOnly, accuracy

'''
data=Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

algo=SVD()

perf=evaluate(algo, data, measures=['RMSE','MAE'])

print(perf)
'''

'''file=path='../ml-latest-small/ml-latest-small/tags.csv'

reader=Reader(line_format='userId movieId rating timestamp', sep=',')'''

'''reader=Reader('ml-100k')
file_dir='C:/Users/dmlab/Dropbox/2018-2/산학협력프로젝트/ml-100k'
train_file=file_dir+'/u%d.base'
test_file=file_dir+'/u%d.test'
fold_files=[(train_file % i, test_file % i) for i in range(1,6)]

data=Dataset.load_from_folds(fold_files, reader=reader)

print(data) #<surprise.dataset.DatasetUserFolds object at 0x00000258F39295C0>
'''

data=Dataset.load_builtin('ml-100k')
data.split(n_folds=3)
print(data) #<surprise.dataset.DatasetAutoFolds object at 0x00000213B57397F0>

algo=BaselineOnly()
algo2=KNN

for trainset, testset in data.folds():
    algo.train(trainset)
    pred=algo.test(testset)
    rmse=accuracy.rmse(pred, verbose=True)
    print(rmse)

