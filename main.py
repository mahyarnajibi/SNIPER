from lib.MNIterator import MNIterator

if __name__=='__main__':
	rec_path = 'train_list.rec'
	list_path = 'train_list.lst'
	mn_iterator = MNIterator(rec_path,list_path,[0,0,0],batch_size=4)
	mn_iterator.next()
