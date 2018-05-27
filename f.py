import skimage.io
import skimage.feature
import math
from sklearn import neighbors
import os, re
import pickle, bisect
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

def image_files_in_folder(folder):
	return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def ret_GLCM_feature(path):
	im = skimage.io.imread(path, as_grey=True)
	im = skimage.img_as_ubyte(im) 
	im //= 32
	g = skimage.feature.greycomatrix(im, [5], [0], levels=8, symmetric=True, normed=True)
	con = skimage.feature.greycoprops(g, 'contrast')[0][0]
	cor = skimage.feature.greycoprops(g, 'correlation')[0][0]
	ene = skimage.feature.greycoprops(g, 'energy')[0][0]
	ent = skimage.feature.greycoprops(g, 'dissimilarity')[0][0]
	hom = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
	pro = skimage.feature.greycoprops(g, 'ASM')[0][0]

	return (con, cor, ene, ent, hom, pro)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def knn_train(train_dir, model_save_path=None, n_neighbors=None, verbose=True, knn_algo='ball_tree'):
	X = []
	y = []

	ind = 0
	nameslist = {}
	train_dir = os.path.abspath(train_dir)
	# Loop through each image in the training set
	for class_dir in os.listdir(train_dir):
		if not os.path.isdir(os.path.join(train_dir, class_dir)):
			continue

		nameslist[ind] = class_dir
		# Loop through each training image for the current image
		for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
	
			# Add encoding for current image to the training set
			X.append(ret_GLCM_feature(img_path))
			y.append(class_dir)
			ind += 1

	# Determine how many neighbors to use for weighting in the KNN classifier
	if n_neighbors is None:
		n_neighbors = int(round(math.sqrt(len(X))))
		if verbose: print("Chose n_neighbors automatically:", n_neighbors)

	# Create and train the KNN classifier
	knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
	knn_clf.fit(X, y)

	print(len(y))

	# Save the trained KNN classifier
	if model_save_path is not None:
		with open(model_save_path, 'wb') as f:
			pickle.dump(knn_clf, f)
	with open('ind.bin', 'wb') as f:
		pickle.dump(nameslist, f)

	return knn_clf


def knn_predict(X_img_path, knn_clf=None, model_path=None, neighbors = 2, distance_threshold=0.15):
	if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
		raise Exception("Invalid image path: {}".format(X_img_path))

	if knn_clf is None and model_path is None:
		raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

	# Load a trained KNN model (if one was passed in)
	if knn_clf is None:
		with open(model_path, 'rb') as f:
			knn_clf = pickle.load(f)

	# Find encodings for texture features in the test image
	encodings = [ret_GLCM_feature(X_img_path)]

	# Use the KNN model to find the best matches for the test face
	outind, outdist = [], []
	dist, ind = knn_clf.kneighbors(encodings, n_neighbors=neighbors)
	dist = dist[0]; ind = ind[0]
	for i, d in enumerate(dist):
		if d <= distance_threshold:
			outind.append(ind[i])
			outdist.append(d)
	return outind, outdist

# 40 115
if __name__ == "__main__":
	# STEP 1: Train the KNN classifier and save it to disk
	# Once the model is trained and saved, you can skip this step next time.

	print("\n\n\tTraining KNN classifier...")
	# classifier = knn_train("Training_DataSet", model_save_path="trained_glcm_knn_model.clf")
	print("\tTraining complete!")
	
	# STEP 2: Using the trained classifier, make predictions for unknown images


	print("\tAnalyzing the Test images now...")

	print("Enter the path of the image to classify, enter 'q' to quit")
	with open("ind.bin", 'rb') as f:
		nameslist = pickle.load(f)
	while True:
		print("\n")
		path = input()
		if path == 'q':
			break
		path = 'C:\\Users\\Utkarsh\\Desktop\\MINI PROJECT GLCM\\Test_DataSet\\'+path+ '.jpg'

		if not os.path.isfile(path) or os.path.splitext(path)[1][1:] not in ALLOWED_EXTENSIONS:
			print("Invalid Image path!!: {}\nTry Again.\n".format(path))
			continue

		full_file_path = os.path.abspath(path)
	
		print("Looking for textures in {}".format(os.path.basename(full_file_path)))

		# Find all people in the image using a trained classifier model
		# Note: You can pass in either a classifier file name or a classifier model instance
		indices, dist = knn_predict(full_file_path, model_path="trained_glcm_knn_model.clf")

		name1=""
		if indices:
			# Print results on the console
			for j, ind in enumerate(indices):
				name = ""
				l = sorted(nameslist.keys())
				for i, k in enumerate(l):
					if k > ind:
						if i > 0:
							name = nameslist[l[i - 1]]
							break
				
				if name: 
					if name!=name1:
						print( "Prediction : {}".format(name), ", Distance: ", dist[j])
					j += 1
					name1=name
				else :
					print("Unknown")
		else:
			print("Unknown")

	# print("\tAnalysis completed successfully.")
	# print("\n\tThe following is the summary of Classification done : \n\n\tSr. No.\tCategory\tTrain\tTest\tCorrect(in 1)\t     Accuracy\tCorrect(in 2)\tAccuracy")
	# print("\t1\tAnt\t\t43\t 10\t\t8\t\t80%\t\t9\t90%")
	# print("\t2\tButterfly\t35\t 10\t\t7\t\t70%\t\t9\t90%")
	# print("\t3\tFace\t\t435\t 10\t\t7\t\t70%\t\t8\t80%")
	# print("\t4\tLaptop\t\t94\t 10\t\t7\t\t70%\t\t10\t100%")
	# print("\t5\tPigeon\t\t60\t 10\t\t7\t\t70%\t\t10\t100%")
	# print("\t6\tScissors\t28\t 10\t\t7\t\t70%\t\t10\t100%")
	# print("\t7\tSoccer_Ball\t66\t 10\t\t7\t\t70%\t\t8\t80%")
	# print("\t8\tStarfish\t50\t 10\t\t8\t\t80%\t\t9\t90%")
	# print("\n\t9\tOTHERS\t\t--\t 20\t\t13\t\t65%\t\t15\t75%")
	# print("\n\t\tAverage Accuracy (considering 1st Match) : 71.667%")
	# print("\n\t\tAverage Accuracy (considering Top 2 Matches) : 89.44%\n\nConfusion Matrix\n\n")
	
	# multiclass = np.array([	[8, 0, 0, 0, 0, 1, 0, 0, 0],
 #                       		[1, 7, 0, 0, 0, 0, 0, 0, 1],
 #                       		[0, 1, 7, 0, 1, 0, 0, 0, 0],
 #                       		[0, 2, 0, 7, 1, 0, 3, 0, 4],
 #                       		[1, 0, 1, 2, 7, 2, 0, 1, 1],
 #                       		[0, 0, 0, 0, 1, 7, 0, 1, 0],
 #                       		[0, 0, 1, 1, 0, 0, 7, 0, 2],
 #                       		[0, 0, 1, 0, 0, 0, 0, 8, 1],
 #                       		[0, 0, 0, 0, 0, 0, 0, 0, 13]])

	# fig, ax = plot_confusion_matrix(conf_mat=multiclass)
	# plt.show()