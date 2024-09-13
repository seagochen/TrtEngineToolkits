//
// Created by vipuser on 9/16/24.
//
#include "common/engine/inference.h"

// YoloPoint のシリアライズ関数
void to_json(const YoloPoint& p, nlohmann::json& j) {
    j = nlohmann::json{{"x", p.x}, {"y", p.y}, {"conf", p.conf}};
}

// YoloResult のシリアライズ関数
void to_json(const YoloResult& r, nlohmann::json& j) {
    // Convert r.keypoints to nlohmann::json
    std::vector<nlohmann::json> keypoints;
    for (const auto &p : r.key_pts) {
        nlohmann::json j_p;
        to_json(p, j_p);
        keypoints.push_back(j_p);
    }

    // Convert r to nlohmann::json
    j = nlohmann::json{
            {"lx", r.lx}, {"ly", r.ly}, {"rx", r.rx}, {"ry", r.ry},
            {"cls", r.cls}, {"conf", r.conf}, {"keypoints", keypoints}
    };
}

// YoloResult のデシリアライズ関数
void from_json(const std::string &str, YoloResult &r) {
    nlohmann::json j = nlohmann::json::parse(str);
    r.lx = j["lx"];
    r.ly = j["ly"];
    r.rx = j["rx"];
    r.ry = j["ry"];
    r.cls = j["cls"];
    r.conf = j["conf"];
    r.key_pts = j["keypoints"].get<std::vector<YoloPoint>>();
}

// YoloPoint のデシリアライズ関数
void from_json(const std::string &str, YoloPoint &p) {
    nlohmann::json j = nlohmann::json::parse(str);
    p.x = j["x"];
    p.y = j["y"];
    p.conf = j["conf"];
}

// vector<YoloResult> のシリアライズ関数
std::string to_json(const std::vector<YoloResult>& results) {
    // Convert results to nlohmann::json
    std::vector<nlohmann::json> j_results;
    for (const auto &r : results) {
        nlohmann::json j_r;
        to_json(r, j_r);
        j_results.push_back(j_r);
    }

    // Convert j_results to string
    nlohmann::json j = j_results;
    return j.dump();
}

// vector<YoloResult> のデシリアライズ関数
void from_json(const std::string &str, std::vector<YoloResult> &results) {
    nlohmann::json j = nlohmann::json::parse(str);
    results = j.get<std::vector<YoloResult>>();
}