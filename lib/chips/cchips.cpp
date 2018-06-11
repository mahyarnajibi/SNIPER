/*
---------------------------------------------------------------
SNIPER: Efficient Multi-Scale Training
Licensed under The Apache-2.0 License [see LICENSE for details]
by Mahyar Najibi and Bharat Singh
---------------------------------------------------------------
*/
#include <algorithm>
#include <set>
#include <vector>
#include <stdlib.h>
#include <iostream>
namespace chips{
    void compute_overlaps(std::vector<std::vector<float> >& boxes1, std::vector<std::vector<float> >& boxes2, int n1, int n2, int ignore_flag, std::vector<std::vector<float> >& overlaps)
    {
      float x1, y1, x2, y2;
      float xx1, yy1, xx2, yy2;
      float area1, area2, uarea;
      float iw, ih;

      overlaps.resize(n1);
      for (int i = 0; i < n1; i++) {
        overlaps[i].resize(n2, 0);
      }

      for (int i = 0; i < n1; i++) {
        x1 = boxes1[i][0];
        y1 = boxes1[i][1];
        x2 = boxes1[i][2];
        y2 = boxes1[i][3];
        area1 = (x2 - x1 + 1) * (y2 - y1 + 1);

        for (int j = 0; j < n2; j++) {
          xx1 = boxes2[j][0];
          yy1 = boxes2[j][1];
          xx2 = boxes2[j][2];
          yy2 = boxes2[j][3];
          area2 = (xx2 - xx1 + 1) * (yy2 - yy1 + 1);
          iw = std::min(x2, xx2) - std::max(x1, xx1) + 1;
          if (iw > 0) {
        ih = std::min(y2, yy2) - std::max(y1, yy1) + 1;
        if (ih > 0) {
          if (ignore_flag == 1) {
            overlaps[i][j] = iw*ih / area2;
          } else {
            uarea = area1 + area2 - ih*iw;
            overlaps[i][j] = ih*iw / uarea;
          }
        }
          }
        }
      }
    }
    std::vector<std::vector<float> > cgenerate(int width, int height, int chipsize, std::vector<std::vector<float> >& boxes, int num_boxes)
    {
       if (boxes.size() == 0 || boxes[0].size() == 0)
            return std::vector<std::vector<float> >();
      std::vector<std::vector<float> > chips;
      std::vector<float> tmp(4);

      //handle corners
      tmp[0] = std::max(width - chipsize, 0);
      tmp[1] = 0;
      tmp[2] = width-1;
      tmp[3] = std::min(chipsize, height-1);

      chips.push_back(tmp);

      tmp[0] = 0;
      tmp[1] = std::max(height - chipsize, 0);
      tmp[2] = std::min(chipsize, width-1);
      tmp[3] = height-1;

      chips.push_back(tmp);

      tmp[0] = std::max(width - chipsize, 0);
      tmp[1] = std::max(height - chipsize, 0);
      tmp[2] = width-1;
      tmp[3] = height-1;

      chips.push_back(tmp);

      int stride = 32;

      for (int i = 0; i < width - chipsize; i = i + stride) {
        for (int j = 0; j < height - chipsize; j = j + stride) {
          tmp[0] = i;
          tmp[1] = j;
          tmp[2] = i + chipsize - 1;
          tmp[3] = j + chipsize - 1;
          chips.push_back(tmp);
        }
      }

      //handle edges
      for (int i = 0; i < height - chipsize; i = i + stride) {
        tmp[0] = std::max(width - chipsize - 1, 0);
        tmp[1] = i;
        tmp[2] = width - 1;
        tmp[3] = i + chipsize - 1;
        chips.push_back(tmp);
      }

      for (int i = 0; i < width - chipsize; i = i + stride) {
        tmp[0] = i;
        tmp[1] = std::max(height - chipsize - 1, 0);
        tmp[2] = i + chipsize - 1;
        tmp[3] = height - 1;
        chips.push_back(tmp);
      }

      int num_chips = chips.size();

      std::vector<int> ids(num_chips);
      for (int i = 0; i < num_chips; i++) {
        ids[i] = i;
      }

      random_shuffle(ids.begin(), ids.end());

      std::vector<std::vector<float> > vchips(num_chips);
      for (int i = 0; i < num_chips; i++) {
        vchips[i].resize(4);
        vchips[i][0] = chips[ids[i]][0];
        vchips[i][1] = chips[ids[i]][1];
        vchips[i][2] = chips[ids[i]][2];
        vchips[i][3] = chips[ids[i]][3];
      }

      std::vector<std::vector<float> > overlaps;
      compute_overlaps(vchips, boxes, num_chips, num_boxes, 1, overlaps);
      std::vector<std::set<int> > chip_matches;

      for (int i = 0; i < num_chips; i++) {
        std::set<int> matches;
        for (int j = 0; j < num_boxes; j++) {
          if (overlaps[i][j] == 1) {
        matches.insert(j);
          }
        }
        chip_matches.push_back(matches);
      }

      int total_matches = 0, max_matches, mid, num_match;
      num_chips = chip_matches.size();
      std::vector<int> finalids;

      while (1) {
        max_matches = 0;
        mid = 0;
        for (int i = 0; i < num_chips; i++) {
          num_match = chip_matches[i].size();
          if (num_match > max_matches) {
        max_matches = num_match;
        mid = i;
          }
        }
        if (max_matches == 0)
          break;
        finalids.push_back(mid);
        std::set<int> tmpd = chip_matches[mid];
        for (int i = 0; i < chip_matches.size(); i++) {
          std::set<int> result;
          std::set_difference(chip_matches[i].begin(), chip_matches[i].end(),
                 tmpd.begin(), tmpd.end(),
                 inserter(result, result.end()));
          chip_matches[i] = result;
        }
      }
      num_chips = finalids.size();
      std::vector<std::vector<float> >fchips(num_chips, std::vector<float>(4, 0));
      for (int i = 0; i < num_chips; i++) {
        fchips[i][0] = vchips[finalids[i]][0];
        fchips[i][1] = vchips[finalids[i]][1];
        fchips[i][2] = vchips[finalids[i]][2];
        fchips[i][3] = vchips[finalids[i]][3];
      }
      return fchips;
    }
}


