//
// Created by user on 3/21/25.
//

#ifndef INFER_YOLO_DEF_H
#define INFER_YOLO_DEF_H

#include <vector>

// Make Yolo a polymorphic base class by adding at least one virtual function.
// Even a virtual destructor is enough for this purpose and good practice.
struct Yolo {
    int lx, ly, rx, ry;
    float conf;
    int cls;

    // A virtual destructor is crucial for proper cleanup when
    // deleting derived class objects through a base class pointer.
    virtual ~Yolo() = default;
};

struct YoloPoint {
    int x, y;
    float conf;
};

// YoloPose publicly inherits from Yolo
struct YoloPose : Yolo {
    // lx, ly, rx, ry, conf, and cls are inherited from Yolo
    std::vector<YoloPoint> pts;
};

#endif //INFER_YOLO_DEF_H
