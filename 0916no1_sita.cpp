// g++ 0916no1_fix.cpp -std=c++17 -fopenmp \
//     -I/usr/local/include/opencv4 -I. -I/workspace/code/nanoflann/include \
//     -L/usr/local/lib \
//     -lopencv_core -lopencv_imgproc -lopencv_cudaarithm -lopencv_cudafilters \
//     -lproj -lpdalcpp -o run

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <limits>
#include <cmath>
#include <proj.h>
#include <pdal/pdal.hpp>
#include <pdal/PointView.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/io/LasWriter.hpp>

namespace fs = std::filesystem;

struct PointXYZ {
    double x, y, z;  // 内部表現: x=lon, y=lat, z=高さ
};

/// === XYZファイル読み込み ===
/// 入力 (lat, lon, z) → 内部 (x=lon, y=lat, z=z)
std::vector<PointXYZ> loadXYZFiles(const std::string& dir) {
    std::vector<PointXYZ> points;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".xyz") {
            std::ifstream infile(entry.path());
            if (!infile.is_open()) continue;

            double lat, lon, z;
            while (infile >> lat >> lon >> z) {
                if (std::isnan(lat) || std::isnan(lon) || std::isnan(z)) continue;

                // 内部表現に変換 (lon, lat, z)
                points.push_back({lon, lat, z});
            }
        }
    }
    return points;
}

int main() {
    std::string input_dir  = "/workspace/fulldata/floor_sita_xyz/";
    std::string output_las = "/workspace/output/0916no1_suidoubasi_floor_sita.las";

    // === [1] XYZ読み込み ===
    auto geo_points = loadXYZFiles(input_dir);
    std::cout << "✅ 元点数: " << geo_points.size() << std::endl;

    if (geo_points.empty()) {
        std::cerr << "❌ 有効な点群がありません" << std::endl;
        return 1;
    }

    // === [2] PROJ 初期化 ===
    PJ_CONTEXT* C = proj_context_create();
    PJ* P = proj_create_crs_to_crs(
        C,
        "EPSG:4326",   // 入力: WGS84 (lon,lat)
        "EPSG:32654",  // 出力: UTM Zone 54N
        nullptr);

    if (!P) {
        std::cerr << "❌ PROJ 初期化失敗" << std::endl;
        return 1;
    }

    // === [3] UTM変換 ===
    std::vector<PointXYZ> utm_points;
    utm_points.reserve(geo_points.size());

    for (size_t i = 0; i < geo_points.size(); i++) {
        // そのまま渡す (x=lon, y=lat)
        PJ_COORD coord  = proj_coord(geo_points[i].x, geo_points[i].y, geo_points[i].z, 0);
        PJ_COORD result = proj_trans(P, PJ_FWD, coord);

        if (proj_errno(P) != 0) {
            std::cerr << "❌ PROJ変換エラー: point[" << i << "] "
                      << "lon=" << geo_points[i].x << ", lat=" << geo_points[i].y
                      << " → " << proj_errno_string(proj_errno(P)) << std::endl;
            continue;
        }

        utm_points.push_back({result.xy.x, result.xy.y, result.xyz.z});

        // 最初の数点だけ確認用に表示
        if (i < 5) {
            std::cout << "DEBUG UTM point[" << i << "]: "
                      << "E=" << result.xy.x << ", N=" << result.xy.y
                      << ", Z=" << result.xyz.z << std::endl;
        }
    }

    proj_destroy(P);
    proj_context_destroy(C);

    std::cout << "✅ UTM変換後の点数: " << utm_points.size() << std::endl;

    // === [4] LAS保存 (UTM座標系で出力) ===
    pdal::PointTable table;
    pdal::PointLayoutPtr layout = table.layout();
    layout->registerDim(pdal::Dimension::Id::X);
    layout->registerDim(pdal::Dimension::Id::Y);
    layout->registerDim(pdal::Dimension::Id::Z);

    pdal::PointViewPtr view(new pdal::PointView(table));
    for (auto& p : utm_points) {
        pdal::PointId id = view->size();
        view->setField(pdal::Dimension::Id::X, id, p.x);
        view->setField(pdal::Dimension::Id::Y, id, p.y);
        view->setField(pdal::Dimension::Id::Z, id, p.z);
    }

    pdal::Options opt;
    opt.add("filename", output_las);
    opt.add("scale_x", 0.001); // 1mm 精度
    opt.add("scale_y", 0.001);
    opt.add("scale_z", 0.001);
    opt.add("offset_x", 0.0);
    opt.add("offset_y", 0.0);
    opt.add("offset_z", 0.0);
    opt.add("a_srs", "EPSG:32654");  // 出力CRSをUTMに指定

    pdal::LasWriter writer;
    writer.setOptions(opt);

    pdal::PointViewSet viewSet;
    viewSet.insert(view);
    writer.prepare(table);
    writer.execute(table);

    std::cout << "🎉 LAS出力完了: " << output_las << std::endl;
    return 0;
}
