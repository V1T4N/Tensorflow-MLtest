
===============================================
サンプルデータをお使いになる前に必ずお読みください
===============================================

●サンプルデータについて
　サンプルデータには『現場で使える！ TensorFlow開発入門　Kerasによる深層学習モデル構築手法』の本文で解説した第3章から第12章のサンプルを用意しています。

●本書のサンプルのテスト環境とGPUへの対応
　本書のサンプルは以下の環境で、問題なく動作することを確認しています。
　なお第2部のサンプルを実機で実行するにはGPU（NVIDIA社のGeForce GTX1080Ti相当）が必須です。GPUが搭載されていないマシンで第2部のサンプルを動作させるとCPUに大きな負荷がかかってしまいますので、推奨いたしません。
　そこで、マシンスペックを満たさない方に向けて、本書ではGCP（Google Compute Platform）によるGPUの環境構築について解説した「補足資料」（hozoku.zip）を用意しています。GCPによるGPUの環境構築、TensorFlow-GPUのインストール方法等は、「補足資料」（hozoku.zip）をダウンロードして確認してください。

・第1部
OS：Windows 10
CPU：Intel Core i5 3.00GHz、4コア
メモリ：8GB
GPU：なし
Python：3.5.4/3.5.5
Anaconda：5.0.1
TensorFlow：1.5.0

・第2部（GPU：NVIDIA社のGeForce GTX1080Ti相当が必要）
OS：Ubuntu 16.04.4 LTS
CPU：Intel Xeon E5-1650 v4 3.60GH、6コア
メモリ：64GB
GPU：GeForce GTX1080Ti
Python：3.5.4/3.5.5
TensorFlow-GPU：1.5.0


●サンプルデータの一覧
　サンプルデータのフォルダ構成は次の通りです。zipファイルを解凍して利用してください。
　また書籍の図表のうち、カラーで確認できるものを「colorpicture」フォルダにまとめています。

sample.zip
    +-- chapter3【第3章のサンプル】
    +-- chapter4【第4章のサンプル】
    +-- chapter5【第5章のサンプル】
    +-- chapter6【第6章のサンプル】
    +-- chapter7【第7章のサンプル】
    +-- chapter8【第8章のサンプル】
    +-- chapter9【第9章のサンプル】
    +-- chapter10【第10章のサンプル】
    +-- chapter11【第11章のサンプル】
    +-- chapter12【第12章のサンプル】
    +-- sanko【Part2のサンプルを実行するのに必要な各種インポート群をまとめたもの】
    +-- colorpicture【第6章、第7章、第8章、第9章、第11章の参考カラー画像】
    +-- README.txt

●免責事項について
・本書に記載されたURLなどは予告なく変更される場合があります。
・本書の出版にあたっては正確な記述につとめましたが、著者や出版社などのいずれも、本書の内容に対して何らかの保証をするものではなく、内容やサンプルにもとづくいかなる運用結果に関してもいっさいの責任を負いません。
・本書に記載されている会社名、製品名はそれぞれ各社の商標および登録商標です。
・本書の内容は、2018年3月執筆時点のものです。

2018年4月　株式会社翔泳社 編集部