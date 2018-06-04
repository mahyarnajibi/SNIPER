import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def crop_polys(polys, crop, im_shape, im_scale, im_path):
	new_polys = []
	for poly in polys:
		new_segs = []
		for seg in poly:
			new_seg = np.array(seg, dtype=np.float32)
			new_seg[::2] -= crop[0]
			new_seg[1::2] -= crop[1]
			new_seg *= im_scale
			new_segs.append(new_seg)
		new_polys.append(new_segs)
	return new_polys


def poly_encoder(polys, cats, max_poly_len=500, max_n_gts=100):
	all_encoded = -1 * np.ones((max_n_gts, max_poly_len), dtype=np.float32)
	for i, (poly, cat) in enumerate(zip(polys, cats)):
		if i>=max_n_gts:
			break
		n_seg = len(poly)
		encoded = np.array([cat], dtype=np.float32)
		cum_sum = 2 + n_seg
		count = 0
		lens = []
		for seg in poly:
			if cum_sum + len(seg) > max_poly_len:
				break
			count += 1
			cum_sum += len(seg)
			lens.append(len(seg))
		encoded = np.hstack((encoded, count))
		encoded = np.hstack((encoded, lens))

		for j in range(count):
			seg = poly[j]
			encoded = np.hstack((encoded, np.array(seg, dtype=np.float32)))

		all_encoded[i, 0:len(encoded)] = encoded
	return all_encoded



        
        






