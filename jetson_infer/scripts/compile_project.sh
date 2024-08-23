#!/bin/bash

# もし build ディレクトリが存在しなければ作成し、存在する場合は内容をクリア
if [ ! -d build ]; then
  mkdir build
else
  rm -rf build/*
fi

# cumatrix ディレクトリに移動してコンパイル
cd cumatrix || { echo "cumatrix ディレクトリへの移動に失敗しました"; exit 1; }
make || { echo "cumatrix のコンパイルに失敗しました"; exit 1; }

# cnn_toolkits ディレクトリに移動してコンパイル
cd ../cnn_toolkits || { echo "cnn_toolkits ディレクトリへの移動に失敗しました"; exit 1; }
make || { echo "cnn_toolkits のコンパイルに失敗しました"; exit 1; }

# build ディレクトリに移動
cd ../build || { echo "build ディレクトリへの移動に失敗しました"; exit 1; }
cmake .. || { echo "CMake の設定に失敗しました"; exit 1; }
make || { echo "プロジェクトのビルドに失敗しました"; exit 1; }

# 生成された jetson_infer を一つ上のディレクトリにコピー
cp jetson_infer ../ || { echo "jetson_infer のコピーに失敗しました"; exit 1; }

echo "ビルドとコピーが正常に完了しました。"
