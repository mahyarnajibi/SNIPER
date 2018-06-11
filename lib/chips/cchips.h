/*
SNIPER: Efficient Multi-Scale Training
Licensed under The Apache-2.0 License [see LICENSE for details]
by Mahyar Najibi and Bharat Singh
*/
#ifndef C_CHIPS_H
#define C_CHIPS_H
#include <vector>
namespace chips{
    void compute_overlaps(std::vector<std::vector<float> >& boxes1, std::vector<std::vector<float> >& boxes2, int n1, int n2, int ignore_flag, std::vector<std::vector<float> >& overlaps);
    std::vector<std::vector<float> > cgenerate(int width, int height, int chipsize, std::vector<std::vector<float> >& boxes, int num_boxes);
}
#endif