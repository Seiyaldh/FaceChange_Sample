# FaceChange_Sample

顔検出をするために使用したライブラリと分類器の用途

・OpenCV
OpenCV は画像に対する処理のためにも使っていますが、顔を検出する面では、入力画像 内の複数人の顔を検出し、その顔領域を別々に抽出するために使っています。検出にはカス ケード分類器を使って、”cascade.detectMultiScale”メソッドで検出を行っています。

・dlib
dlib は OpenCV で抽出した顔に対して顔検出を行い、そこで検出した顔の目や鼻などの特 徴点を抽出するために使っています。
