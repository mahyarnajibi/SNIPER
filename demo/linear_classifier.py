import mxnet as mx
import numpy as np
import pickle
import logging

import pdb


DEBUG = False

def get_symbol(logger, num_of_classes):
    """
	Symbol of linear classifier
	"""
    X = mx.sym.Variable('data')
    Y = mx.sym.Variable("softmax_label")
    fc1 = mx.sym.FullyConnected(data=X, name='fc1', num_hidden=num_of_classes)
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=Y, name='softmax')

    model = mx.mod.Module(symbol=softmax, data_names=['data'], label_names=['softmax_label'], 
    						logger=logger, context=mx.gpu())
    return model

def label_assignment(indices, class_to_label, class_names):
	"""
	Assign label (0-x) w.r.t. class name (eg: vr/xyz.jpg)
	"""
	keep = []
	labels = []
	for i in range(len(indices)):
		name = indices[i].split('/')[0]
		if name in class_names:
			keep.append(i)
			labels.append(class_to_label[name])
	return labels, keep

def split_train_val(data, labels, indices, num_of_classes, num_val_per_class):
	"""
	split features and labels to train and val sets.
	num_val_per_class: currently there's roughly 100 pics for each class; this determines the number of pics used
						for evaluation per class.
	"""
	splits = []
	eval_data = []
	eval_labels = []
	removed_list = []
	indices_eval = []
	for i in range(num_of_classes):
		splits.append(100 * i)
	for i in range(num_of_classes):
		for j in range(num_val_per_class):
			eval_data.append(data[splits[i] + j])
			eval_labels.append(labels[splits[i] + j])
			removed_list.append(splits[i] + j)
			indices_eval.append(indices[splits[i] + j])
	for index in sorted(removed_list, reverse=True):
		del data[index]
		del labels[index]

	if DEBUG:
		for i in range(len(eval_data)):
			if any((eval_data[i] == x).all() for x in data):
				print("Number %d duplication!" %(i))

	return data, labels, eval_data, eval_labels, indices_eval, removed_list


def data_iter(train_data, train_labels, eval_data, eval_labels, batch_size):
	train_iter = mx.io.NDArrayIter(train_data, train_labels, batch_size, shuffle=True)
	eval_iter = mx.io.NDArrayIter(eval_data, eval_labels, batch_size, shuffle=False)
	return train_iter, eval_iter


def train_model(class_names, image_number_per_class, batch_size=100, learning_rate=0.0001, momentum=0.9, num_epoch=30):
	# input your class names here
    #class_names = ('gpu', 'ironman', 'ted', 'vr', 'fidgetspinner', \
    #                'garfield', 'bugatti', \
    #                'teslaroadster', 'sadcat', 'yeezy')
    num_of_classes = len(class_names)
    class_to_label = dict(zip(class_names, xrange(num_of_classes)))

    # the pooled features and indices include all classes in folder 10cls_imgs
    with open("./demo/cache/features.pkl", 'rb') as f:
    	features = pickle.load(f)
    with open("./demo/cache/indices.pkl", 'rb') as f:
    	indices = pickle.load(f)

    # based on the 10 classes you choose, assign label
    labels, keep = label_assignment(indices, class_to_label, class_names)
    kept_data = [features[i] for i in keep]
    kept_indices = [indices[i] for i in keep]

    # split train val sets
    train_data, train_labels, eval_data, eval_labels, indices_eval, eval_list = split_train_val(kept_data, labels, kept_indices, num_of_classes, 40)
    train_labels = np.array(train_labels)
    train_data = np.concatenate(train_data)
    eval_labels = np.array(eval_labels)
    eval_data = np.concatenate(eval_data)

    # dump indices for evaluation
    with open("./demo/cache/eval_index.txt", 'w') as f:
    	for one in indices_eval:
    		f.write(one+'\n')

    # get data iters
    train_iter, eval_iter = data_iter(train_data, train_labels, eval_data, eval_labels, batch_size)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    model = get_symbol(logger, num_of_classes)
    model.fit(train_iter, eval_iter, optimizer='sgd', 
                optimizer_params={'learning_rate':learning_rate, 'momentum': momentum},
            	num_epoch=num_epoch, eval_metric='acc', batch_end_callback = mx.callback.Speedometer(batch_size, 2))
    
    return model, eval_list


def classify_rois(model, roipooled_features):
    """
    evaluate the rois 
    """
    rois_classification = []

    for one in roipooled_features:
        rois_iter = mx.io.NDArrayIter([one.reshape((one.shape[0], -1))], np.zeros((one.shape[0])), 100, shuffle=False)
        out = model.predict(rois_iter)
        rois_classification.append(out.asnumpy())
    return rois_classification



if __name__ == '__main__':
	train()
